from debug import debug
import torch.nn as nn
import torch.optim

import torch.nn.functional as F

import segmentation_models_pytorch as smp


def get_model(model, in_channels, out_channels):
    if model == 'Unet':
        return Unet(in_channels, [64, 128], out_channels, block_depth=2, bottleneck_depth=2)
    if model == 'Unet_smp':
        return smp.Unet(
            encoder_depth=3,
            in_channels=in_channels,
            decoder_channels=(256, 128, 64),
            classes=out_channels,
            activation=None,
        )
    if model == 'UnetPlusPlus':
        return smp.UnetPlusPlus(
            encoder_depth=3,
            in_channels=in_channels,
            decoder_channels=(256, 128, 64),
            classes=out_channels,
            activation=None,
        )
    if model == 'BaselineColorMatrix':
        return TransferBaselineColorMatrix(in_channels, out_channels)
    if model == 'BaselineConv':
        return TransferBaselineConv(in_channels, out_channels)

    raise Exception('invalid model')


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,
                  kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, linear_head=True):
        super().__init__()
        block_features = [block_features[0]] + block_features
        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = linear_head

        if linear_head:
            self.head = nn.Linear(block_features[-1], num_classes)
        else:
            self.head = nn.Conv2d(
                block_features[-1], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head:
            # completely reduce spatial dimension
            x = F.avg_pool2d(x, x.shape[-1])
            x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


class Unet(nn.Module):
    def __init__(self, in_channels, down_features, num_classes, block_depth=1, bottleneck_depth=2, pooling=False):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels, down_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(down_features[0]),
        )

        self.pooling = pooling

        down_stride = 2 if not pooling else 1

        # if pooling:
        #     self.downs = nn.ModuleList([
        #         nn.Sequential(*[conv_block(ins, outs, stride=2 if i == block_repeat else 1)
        #                         for i in range(block_repeat)])
        #         for ins, outs in zip(down_features, down_features[1:])
        #     ])

        self.downs = nn.ModuleList([
            nn.Sequential(*[conv_block(ins, outs if i == block_depth - 1 else ins, stride=down_stride if i == block_depth - 1 else 1)
                            for i in range(block_depth)])
            for ins, outs in zip(down_features, down_features[1:])])

        self.bottleneck = nn.Sequential(
            *[ResBlock(down_features[-1], down_features[-1]) for i in range(bottleneck_depth)])

        up_features = down_features[::-1]
        self.ups = nn.ModuleList([
            nn.Sequential(*[conv_block(ins + outs if i == 0 else outs, outs)
                            for i in range(block_depth)])
            for ins, outs in zip(up_features, up_features[1:])])

        self.head = nn.Conv2d(down_features[0], num_classes, kernel_size=1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expand(x)

        x_skips = []

        for down in self.downs:
            x_skips.append(x)
            x = down(x)
            if self.pooling:
                x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up, x_skip in zip(self.ups, reversed(x_skips)):
            x = torch.cat([self.upsample(x), x_skip], dim=1)
            x = up(x)

        x = self.head(x)

        return x


class DistortionModelConv(nn.Module):
    def __init__(self, input_shape, lambd=0.1):
        super().__init__()

        kernel_size = 3
        nch = input_shape[0]

        self.conv1 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.noise = torch.randn(input_shape).unsqueeze(0)

        self.conv1.weight.data.normal_()
        self.conv2.weight.data.normal_()
        self.conv1.weight.data *= lambd
        self.conv2.weight.data *= lambd
        for f in range(nch):
            self.conv1.weight.data[f][f][1][1] += 1
            self.conv2.weight.data[f][f][1][1] += 1

        self.conv1.bias.data.normal_()
        self.conv2.bias.data.normal_()
        self.conv1.bias.data *= lambd
        self.conv2.bias.data *= lambd

        self.noise.data *= lambd

    @ torch.no_grad()
    def forward(self, inputs):
        outputs = inputs
        outputs = outputs + self.noise
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class TransferBaselineColorMatrix(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.color_matrix = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.color_matrix.weight.data = torch.eye(3).reshape(3, 3, 1, 1)
        # self.color_matrix.bias.data = torch.zeros(3)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.color_matrix(x)
        x = self.bn(x)

        return x


class TransferBaselineConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x


if __name__ == '__main__':
    from datasets import get_dataset

    dataset = get_dataset('Cytomorphology')

    x = dataset[0][0].unsqueeze(0)

    import matplotlib.pyplot as plt
    # import torchvision.utils
    from torchvision.utils import make_grid

    plt.imshow(make_grid(x, normalize=True).permute(1, 2, 0))

    model = TransferBaselineColorMatrix(3, 3)
    print(model.color_matrix.weight.shape)
    x = model(x)
    plt.imshow(make_grid(x, normalize=True).permute(1, 2, 0))

    # net = Unet(3, [64, 128], 3, block_depth=2, bottleneck_depth=2)
    # # from utils import get_layers
    # # conv_layers = get_layers(net, nn.Conv2d)
    # from debug import debug

    # debug(net)

    # x = torch.randn((8, 3, 32, 32))
    # y = net(x)
    # debug(y)
