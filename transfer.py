from collections import defaultdict
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import num_params, test_accuracy, pretty_plot
from datasets import get_dataset

from models import DistortionModelConv, resnet18, resnet34

import argparse

os.makedirs('models', exist_ok=True)

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_from', choices=['PBCBarcelona', 'CIFAR10'], default='CIFAR10')
parser.add_argument('--dataset_to', choices=['PBCBarcelona', 'CIFAR10'], default='CIFAR10')
parser.add_argument('--network', choices=['Unet', 'UnetPlusPlus'], default='Unet')
parser.add_argument('--model_from', default='models/model.ckpt', help='Model checkpoint for saving/loading.')
parser.add_argument('--ckpt', default='auto', help='Model checkpoint for saving/loading.')

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--unsupervised', action='store_true', help='Don\'t use label information.')
parser.add_argument('--distortion_test', action='store_true')

parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true', help='Save only the best models (measured in valid accuracy).')
args = parser.parse_args()

device = 'cuda'  # if args.cuda else 'cpu'

if args.ckpt == 'auto':
    args.ckpt = args.model_from.replace('.ckpt', f'_transfer_{args.dataset_to}.ckpt')

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

dataset_to = get_dataset(args.dataset_to)
train_loader = DataLoader(dataset_to.train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
valid_loader = DataLoader(dataset_to.valid_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(dataset_to.test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

in_channels = dataset_to.in_channels
num_classes = dataset_to.num_classes


##################################### Train Model #####################################

plot_loc = args.ckpt.split('.')[0] + '.png'

loss_fn = nn.CrossEntropyLoss()

if os.path.exists(args.ckpt) and not args.reset:
    state_dict = torch.load(args.ckpt, map_location=device)
    model = state_dict['model']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    print(f"Loading model {args.ckpt} ({init_epoch} epochs), valid acc {best_acc:.3f}")
else:
    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    if args.network == 'resnet18':
        model = resnet18(in_channels, num_classes)
    if args.network == 'resnet34':
        model = resnet34(in_channels, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


valid_acc = test_accuracy(model, valid_loader, name='valid', device=device)


# SETUP hooks

from utils import get_bn_layers
from debug import debug

bn_layers = get_bn_layers(model)

layer_activations = [None] * len(bn_layers)


def layer_hook_wrapper(idx):
    def hook(_module, inputs, _outputs):
        layer_activations[idx] = inputs
    return hook


for l, layer in enumerate(bn_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def get_bn_loss():
    return ([
        ((bn.running_mean - input.mean(dim=[0, 2, 3])) ** 2).sum()
        for bn, input in zip(bn_layers, layer_activations)
    ])


distortion = None   # used for testing when args.distortion_test

if not os.path.exists(args.ckpt) or args.resume_training:

    print('Training transfer model ' f'{model},'
          f'params:\t{num_params(model) / 1000:.2f} K')

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        model.train()
        step_start = epoch * len(train_loader)
        for step, (x, y) in enumerate(train_loader, start=step_start):
            x, y = x.to(device), y.to(device)

            if args.distortion_test:
                if distortion is None:
                    distortion = DistortionModelConv(input_shape=x.shape, lambd=0.1)
                x = distortion(x)

            logits = model(x)
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

        model.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(model, valid_loader, name='valid', device=device)
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)

        if not args.save_best or valid_acc > best_acc:
            pretty_plot(logs, steps_per_epoch=len(train_loader), smoothing=50, save_loc=plot_loc)
            best_acc = valid_acc

            print(f'Saving model to {args.ckpt}')
            torch.save({'model': model, 'optimizer': optimizer, 'epoch': epoch + 1,
                       'acc': best_acc, 'logs': logs}, args.ckpt)

    if args.save_best:
        state_dict = torch.load(args.ckpt, map_location=device)['model']
        print(f"Loading best model {args.ckpt} ({state_dict['epoch']} epochs), valid acc {best_acc:.3f}")


pretty_plot(logs, smoothing=50)
