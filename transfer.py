from collections import defaultdict
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import num_params, test_accuracy, pretty_plot, get_bn_layers
from datasets import get_dataset

from models import DistortionModelConv, resnet18, resnet34
import segmentation_models_pytorch as smp

from debug import debug

import argparse

os.makedirs('models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_to', choices=['PBCBarcelona', 'CIFAR10', 'CIFAR10Distorted'], default='CIFAR10')
parser.add_argument('--network', choices=['Unet', 'UnetPlusPlus'], default='Unet')
parser.add_argument('--model_from', default='models/model.ckpt', help='Model checkpoint for saving/loading.')
parser.add_argument('--ckpt', default='auto', help='Model checkpoint for saving/loading.')

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--unsupervised', action='store_true', help='Don\'t use label information.')

parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true', help='Save only the best models (measured in valid accuracy).')
args = parser.parse_args()

device = 'cuda'  # if args.cuda else 'cpu'

if args.ckpt == 'auto':
    args.ckpt = args.model_from.replace('.ckpt', f'_transfer-to_{args.dataset_to}.ckpt')

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

dataset_to = get_dataset(args.dataset_to)
train_loader = DataLoader(dataset_to.train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
valid_loader = DataLoader(dataset_to.valid_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(dataset_to.test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

classifier_state_dict = torch.load(args.model_from, map_location=device)
classifier = classifier_state_dict['model']
print(
    f"Loading model {args.model_from} ({classifier_state_dict['epoch']} epochs), valid acc {classifier_state_dict['acc']:.3f}")

##################################### Train Model #####################################

plot_loc = args.ckpt.split('.')[0] + '.png'

loss_fn = nn.CrossEntropyLoss()

if os.path.exists(args.ckpt) and not args.reset:
    state_dict = torch.load(args.ckpt, map_location=device)
    transfer_model = state_dict['model']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    print(f"Loading model {args.ckpt} ({init_epoch} epochs), valid acc {best_acc:.3f}")
else:
    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    if args.network == 'Unet':
        transfer_model = smp.UnetPlusPlus(
            encoder_depth=3,
            in_channels=3,
            decoder_channels=(256, 128, 64),
            classes=3,
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

        for p in classifier.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.transfer_model(x)
        x = self.classifier(x)
        return x

    def train(self, mode=True):
        self.transfer_model.train(mode)
        self.classifier.train(False)    # freeze bn


full_model = FullModel()

valid_acc = test_accuracy(full_model, valid_loader, name='valid', device=device)


# SETUP hooks

bn_layers = get_bn_layers(classifier)

layer_activations = [None] * len(bn_layers)


def layer_hook_wrapper(idx):
    def hook(_module, inputs, _outputs):
        layer_activations[idx] = inputs[0]
    return hook


for l, layer in enumerate(bn_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def get_bn_loss():
    return sum([
        ((bn.running_mean - x.mean(dim=[0, 2, 3])) ** 2).sum()
        + ((bn.running_var - x.var(dim=[0, 2, 3])) ** 2).sum()
        for bn, x in zip(bn_layers, layer_activations)
    ])


if not os.path.exists(args.ckpt) or args.resume_training or args.reset:

    print('Training transfer model ' f'{args.network}, '
          f'params:\t{num_params(transfer_model) / 1000:.2f} K')

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        full_model.train()
        step_start = epoch * len(train_loader)
        for step, (x, y) in enumerate(train_loader, start=step_start):
            x, y = x.to(device), y.to(device)

            logits = full_model(x)
            loss_bn = get_bn_loss()

            acc = (logits.argmax(dim=1) == y).float().mean().item()

            if args.unsupervised:
                loss = loss_bn
                metrics = {'acc': acc, 'loss_bn': loss_bn.item()}
            else:
                loss_crit = loss_fn(logits, y)
                loss = loss_crit + loss_bn
                metrics = {'acc': acc, 'loss_bn': loss_bn.item(), 'loss_crit': loss_crit.item()}

            for m, v in metrics.items():
                logs[m].append(v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % len(train_loader) % 50 == 0:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(train_loader):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        full_model.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(full_model, valid_loader, name='valid', device=device)
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)

        if not args.save_best or valid_acc > best_acc:
            pretty_plot(logs, steps_per_epoch=len(train_loader), smoothing=50, save_loc=plot_loc)
            best_acc = valid_acc

            print(f'Saving transfer_model to {args.ckpt}')
            torch.save({'model': transfer_model, 'optimizer': optimizer, 'epoch': epoch + 1,
                       'acc': best_acc, 'logs': logs}, args.ckpt)

    if args.save_best:
        state_dict = torch.load(args.ckpt, map_location=device)
        print(f"Loading best model {args.ckpt} ({state_dict['epoch']} epochs), valid acc {best_acc:.3f}")


pretty_plot(logs, smoothing=50)
