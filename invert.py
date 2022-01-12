import json
from collections import defaultdict
import os
import shutil
import random

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import num_params, test_accuracy, pretty_plot, get_bn_layers
from datasets import Subset, get_dataset

from debug import debug
from torchvision.utils import make_grid, save_image

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/model.ckpt',
                    help='Model checkpoint for saving/loading.')

parser.add_argument('--num_epochs', type=int, default=3,
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--f_stats', type=float,
                    default=1e-2, help='Stats multiplier')
parser.add_argument('--f_reg', type=float, default=0,
                    help='Regularization multiplier')

parser.add_argument('--unsupervised', action='store_true',
                    help='Don\'t use label information.')

parser.add_argument('--device', default='cuda')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true',
                    help='Save only the best models (measured in valid accuracy).')

parser.add_argument('--save_loc', default='auto',
                    help='Location for saving/loading.')
parser.add_argument('--experiment', type=str, help='Experiment name')
args = parser.parse_args()

device = args.device

if args.save_loc == 'auto':
    save_loc = args.model.replace(
        '.ckpt', f'_lr={args.lr:1.0e}_f-st={args.f_stats:1.0e}_f-reg={args.f_reg:1.0e}')
    save_loc = save_loc.replace('models', 'invert')
else:
    save_loc = args.save_loc

if os.path.exists(save_loc) and args.reset:
    shutil.rmtree(save_loc)

os.makedirs(save_loc, exist_ok=True)

inputs_ckpt = f'{save_loc}/inputs.ckpt'
plot_loc = f'{save_loc}/metrics.png'
log_loc = f'{save_loc}/log.txt'


def log(msg):
    print(msg)
    with open(log_loc, 'a') as f:
        f.write(msg + '\n')


log(json.dumps(vars(args)))


args_log = '\n'.join(f'{k}={v}' for k, v in vars(args).items())
log(args_log + '\n')

torch.manual_seed(4)

classifier_state_dict = torch.load(args.model, map_location=device)
classifier = classifier_state_dict['model']
classifier_input_shape = classifier_state_dict['input_shape']
log(
    f"Loading model {args.model} input {tuple(classifier_input_shape)} ({classifier_state_dict['epoch']} epochs), valid acc {classifier_state_dict['acc']:.3f}")

##################################### Train Model #####################################

loss_fn = nn.CrossEntropyLoss()

if os.path.exists(inputs_ckpt) and not args.reset:
    state_dict = torch.load(inputs_ckpt, map_location=device)
    x, y = state_dict['inputs']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    log(f"Loading inputs {inputs_ckpt} ({init_epoch} epochs), valid acc {best_acc:.3f}")
else:
    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    x = torch.randn((args.batch_size,) + classifier_input_shape)
    y = torch.arange(0, args.batch_size) % 10

    x, y = x.to(device), y.to(device)
    x.requires_grad = True

    optimizer = torch.optim.Adam([x], lr=args.lr)

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
    return x.mean(dim=[0, 2, 3]).norm() + (1 - x.std(dim=[0, 2, 3])).norm()


def total_variation(x):
    tv = ((x[:, :, :, :-1] - x[:, :, :, 1:]).norm()
          + (x[:, :, :-1, :] - x[:, :, 1:, :]).norm()
          + (x[:, :, 1:, :-1] - x[:, :, :-1, 1:]).norm()
          + (x[:, :, :-1, :-1] - x[:, :, 1:, 1:]).norm())
    return tv


def jitter(x):
    off1 = random.randint(-3, 3)
    off2 = random.randint(-3, 3)
    return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


_zero = torch.zeros([1], device=device)

# if not os.path.exists(inputs_ckpt) or args.resume_training or args.reset:

log(f'Inverting inputs for {args.model}')

classifier.eval()

x_original = x

for epoch in range(init_epoch, args.num_epochs):

    x = jitter(x_original)

    logits = classifier(x)

    loss_crit = loss_fn(logits, y)
    loss_bn = args.f_stats * sum(bn_losses) if args.f_stats != 0 else _zero
    loss_reg = args.f_reg * \
        (normalize(x) + total_variation(x)) if args.f_reg != 0 else _zero

    loss = loss_bn + loss_reg

    acc = (logits.argmax(dim=1) == y).float().mean().item()

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

    x = x_original

    if (epoch + 1) % 10 == 0:
        log(f'[{epoch}/{args.num_epochs}] ' +
            ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

    if acc > best_acc:
        best_acc = acc
        save_image(make_grid(x[:64].cpu(), normalize=True,
                             scale_each=True), f'{save_loc}/sample_best.png')

    if (epoch + 1) % 100 == 0:
        pretty_plot(logs, smoothing=50, save_loc=plot_loc)

        log(f'Saving inputs to {inputs_ckpt}')
        torch.save({'inputs': (x, y), 'optimizer': optimizer, 'epoch': epoch + 1,
                    'acc': best_acc, 'logs': logs}, inputs_ckpt)

    if (epoch + 1) % 500 == 0:
        save_image(make_grid(x[:64].cpu(), normalize=True, scale_each=True),
                   f'{save_loc}/sample_{epoch + 1:02d}.png')


pretty_plot(logs, smoothing=50)
