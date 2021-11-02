from torchvision.transforms.functional import rgb_to_grayscale
import torchvision.transforms as T
from collections import defaultdict
import os
import shutil
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import calculate_mean_and_std, num_params, test_accuracy, pretty_plot, get_bn_layers, accuracy
from datasets import Subset, get_dataset, get_transfer_mapping_classes, get_transfer_mapping_labels, CrossEntropyTransfer

from models import ResNet, Unet
import segmentation_models_pytorch as smp

from debug import debug
from torchvision.utils import make_grid, save_image

import argparse

os.makedirs('transfer', exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_to', choices=[
                    'PBCBarcelona', 'PBCBarcelona_2x', 'PBCBarcelona_4x',
                    'Cytomorphology', 'Cytomorphology_2x', 'Cytomorphology_4x',
                    'CIFAR10', 'MNIST', 'SVHN'
                    ], default='CIFAR10')
parser.add_argument(
    '--network', choices=['Unet', 'Unet_smp', 'UnetPlusPlus'], default='Unet')
parser.add_argument('--model_from', default='models/model.ckpt',
                    help='Model checkpoint for saving/loading.')
parser.add_argument('--ckpt', default='auto',
                    help='Model checkpoint for saving/loading.')

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--size', type=int, default=4096,
                    help='Sample used for transfer.')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--f_stats', type=float,
                    default=0.01, help='stats multiplier')
parser.add_argument('--f_reg', type=float, default=0,
                    help='regularization multiplier')

parser.add_argument('--unsupervised', action='store_true',
                    help='Don\'t use label information.')

parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true',
                    help='Save only the best models (measured in valid accuracy).')
args = parser.parse_args()

device = 'cuda'  # if args.cuda else 'cpu'

if args.ckpt == 'auto':
    save_loc = args.model_from.replace(
        '.ckpt', f'_to_{args.dataset_to}_lr={args.lr:1.0e}_f-st={args.f_stats:1.0e}')
    save_loc = save_loc.replace('models', 'transfer')

if os.path.exists(save_loc) and args.reset:
    shutil.rmtree(save_loc)

os.makedirs(save_loc, exist_ok=True)

model_ckpt = f'{save_loc}/model.ckpt'
plot_loc = f'{save_loc}/metrics.png'
log_loc = f'{save_loc}/log.txt'


def log(msg):
    print(msg)
    with open(log_loc, 'a') as f:
        f.write(msg + '\n')


log('\n' + '\n'.join(f'{k}={v}' for k, v in vars(args).items()) + '\n')

# XXX: enable augmentation
dataset_to = get_dataset(args.dataset_to, train_augmentation=False)
train_set = Subset(dataset_to.train_set, range(args.size)
                   ) if args.size > 0 else Subset(dataset_to.train_set)
train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
valid_loader = DataLoader(
    dataset_to.valid_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(
    dataset_to.test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

classifier_state_dict = torch.load(args.model_from, map_location=device)
classifier = classifier_state_dict['model']
classifier_input_shape = classifier_state_dict['input_shape']
log(
    f"Loaded model {args.model_from} ({classifier_state_dict['epoch']} epochs), valid acc {classifier_state_dict['acc']:.3f}")

log(f'input shape: {classifier_input_shape}')

##################################### Transfer Model #####################################

classes_from = classifier_state_dict['classes']
classes_to = dataset_to.classes
transfer_map, _ = get_transfer_mapping_labels(classes_from, classes_to)
log(f'Transfer map: {get_transfer_mapping_classes(classes_from, classes_to)}')

loss_fn = CrossEntropyTransfer(classes_from, classes_to)


# Transfer model input / output channels
in_channels = dataset_to.in_channels
out_channels = classifier_input_shape[0]

if os.path.exists(model_ckpt) and not args.reset:
    state_dict = torch.load(model_ckpt, map_location=device)
    transfer_model = state_dict['model']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    log(f"Loading transfer model {model_ckpt} ({init_epoch} epochs), valid acc {best_acc:.3f}")
else:
    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    if args.network == 'Unet':
        transfer_model = Unet(
            in_channels, [64, 128], out_channels, block_depth=2, bottleneck_depth=2)
    if args.network == 'Unet_smp':
        transfer_model = smp.Unet(
            encoder_depth=3,
            in_channels=in_channels,
            decoder_channels=(256, 128, 64),
            classes=out_channels,
            activation=None,
        )
    if args.network == 'UnetPlusPlus':
        transfer_model = smp.UnetPlusPlus(
            encoder_depth=3,
            in_channels=3,
            decoder_channels=(256, 128, 64),
            classes=3,
            activation=None,
        )

    transfer_model.to(device)

    optimizer = torch.optim.Adam(transfer_model.parameters(), lr=args.lr)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transfer_model = transfer_model
        self.classifier = classifier

        for p in classifier.parameters():   # freeze classifier
            p.requires_grad = False

    def forward(self, x):
        x = self.transfer_model(x)
        x = self.classifier(x)
        return x

    def train(self, mode=True):
        self.transfer_model.train(mode)
        self.classifier.train(False)    # freeze classifier batchnorm layers


full_model = FullModel()

grayscale_to_rgb = T.Lambda(lambda x: x.repeat_interleave(3, 1))

transform = None
if dataset_to.in_channels == 1 and classifier_input_shape[0] == 3:
    transform = grayscale_to_rgb
elif dataset_to.in_channels == 3 and classifier_input_shape[0] == 1:
    transform = rgb_to_grayscale


# # XXX: should check whether images are normalized after distorting, could be a source of error
# # also for running test_accuracy without the transfer model, grayscale_torgb will change normalization
# # calculate_mean_and_std()
test_accuracy(classifier, valid_loader, transform=transform, transfer_map=transfer_map,
              name='no transfer valid', device=device)
valid_acc = test_accuracy(full_model, valid_loader, transfer_map=transfer_map,
                          name='valid', device=device)


##################################### Training #####################################
# SETUP hooks

bn_layers = get_bn_layers(classifier)

bn_losses = [None] * len(bn_layers)


def layer_hook_wrapper(idx):
    def hook(module, inputs, _outputs):
        x = inputs[0]
        bn_losses[idx] = ((module.running_mean - x.mean(dim=[0, 2, 3])).norm(2)
                          + (module.running_var - x.var(dim=[0, 2, 3])).norm(2))
    return hook


for l, layer in enumerate(bn_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def normalize(x):
    return x.mean(dim=[0, 2, 3]).norm() + (1 - x.var(dim=[0, 2, 3])).norm()


def total_variation(x):
    tv = ((x[:, :, :, :-1] - x[:, :, :, 1:]).norm()
          + (x[:, :, :-1, :] - x[:, :, 1:, :]).norm()
          + (x[:, :, 1:, :-1] - x[:, :, :-1, 1:]).norm()
          + (x[:, :, :-1, :-1] - x[:, :, 1:, 1:]).norm())
    return tv


_zero = torch.zeros([1], device=device)

if not os.path.exists(model_ckpt) or args.resume_training or args.reset:

    log('Training transfer model ' f'{args.network}, '
        f'params:\t{num_params(transfer_model) / 1000:.2f} K')

    full_model.to(device)

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        full_model.train()

        step_start = epoch * len(train_loader)
        for step, (x, y) in enumerate(train_loader, start=step_start):
            x, y = x.to(device), y.to(device)

            logits = full_model(x)

            loss_crit = loss_fn(logits, y)
            loss_bn = args.f_stats * \
                sum(bn_losses) if args.f_stats != 0 else _zero
            loss_reg = args.f_reg * \
                (normalize(x) + total_variation(x)) if args.f_reg != 0 else _zero

            loss = loss_bn + loss_reg

            y_pred = logits.argmax(dim=1) == y
            acc = accuracy(y_pred, y, transfer_map=transfer_map)

            # acc = sum(correct) / len(correct)
            # acc = (logits.argmax(dim=1) == y).float().mean().item()

            metrics = {'acc': acc, 'loss_bn': loss_bn.item(),
                       'loss_reg': loss_reg.item()}

            if not args.unsupervised:
                loss += loss_crit
                metrics['loss_crit'] = loss_crit.item()

            for m, v in metrics.items():
                logs[m].append(v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        full_model.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(
            full_model, valid_loader, transfer_map=transfer_map, device=device)
        metrics['valid_acc'] = valid_acc
        interpolate_valid_acc = torch.linspace(
            valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)

        log(f'[{epoch + 1}/{init_epoch + args.num_epochs}] ' +
            ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        view_batch = next(iter(valid_loader))[0][:32].to(device)
        save_image(make_grid(view_batch.cpu(), normalize=True),
                   f'{save_loc}/sample_input.png')
        samples = transfer_model(view_batch)

        if (epoch + 1) % 10 == 0:
            save_image(make_grid(samples.cpu(), normalize=True),
                       f'{save_loc}/sample_{epoch + 1:02d}.png')

        # if not args.save_best or valid_acc > best_acc:
            pretty_plot(logs, steps_per_epoch=len(train_loader),
                        smoothing=50, save_loc=plot_loc)
            best_acc = valid_acc
            save_image(make_grid(samples.cpu(), normalize=True),
                       f'{save_loc}/sample_best.png')

            log(f'Saving transfer_model to {model_ckpt}')
            torch.save({'model': transfer_model, 'optimizer': optimizer, 'epoch': epoch + 1,
                       'acc': best_acc, 'logs': logs}, model_ckpt)

    # if args.save_best:
    #     state_dict = torch.load(model_ckpt, map_location=device)
    #     log(f"Loading best model {model_ckpt} ({state_dict['epoch']} epochs), valid acc {best_acc:.3f}")


pretty_plot(logs, smoothing=50)
