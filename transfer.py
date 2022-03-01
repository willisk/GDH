import json
import sys
from torchvision.transforms.functional import rgb_to_grayscale
import torchvision.transforms as T
from collections import defaultdict
import os
import shutil
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import calculate_mean_and_std, clamp, num_params, str2bool, test_accuracy, pretty_plot, get_bn_layers, accuracy
from datasets import Subset, get_dataset, get_transfer_mapping_classes, get_transfer_mapping_labels, CrossEntropyTransfer

from models import get_model

from debug import debug
from torchvision.utils import make_grid, save_image

import argparse

os.makedirs('transfer', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_to', default='CIFAR10')
parser.add_argument('--network', default='Unet')
parser.add_argument('--model_from', default='models/model.ckpt',
                    help='Model checkpoint for saving/loading.')

parser.add_argument('--size', type=int, default=4096,
                    help='Sample used for transfer.')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--f_stats', type=float,
                    default=0, help='Stats multiplier')
parser.add_argument('--f_reg', type=float, default=0,
                    help='Regularization multiplier')

parser.add_argument('--unsupervised', type=str2bool,
                    help='Don\'t use label information.')
parser.add_argument('--fine_tune', type=str2bool,
                    help='Fine tune classifier head')
parser.add_argument('--retrain_baseline', type=str2bool,
                    help='Retrain classification model balseline.')
parser.add_argument('--shuffle', type=str2bool, default=True,
                    help='Shuffle training dataset.')

parser.add_argument('--device', default='cuda')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true',
                    help='Save only the best models (measured in valid accuracy).')

parser.add_argument('--save_loc', default='auto',
                    help='Location for saving/loading.')
parser.add_argument('--experiment', type=str, help='Experiment name')
args = parser.parse_args()

device = args.device


base_dir = 'transfer' + (f'/{args.experiment}' if args.experiment else '')

if args.save_loc == 'auto':
    append_loc_sci = [f"{a.replace('_', '-')}={getattr(args, a):1.0e}"
                      for a in ['lr', 'f_stats', 'f_reg']
                      if sum([f'--{a}' in arg for arg in sys.argv])]

    append_loc_v = [f"{a.replace('_', '-')}={getattr(args, a)}"
                    for a in ['size', 'unsupervised', 'fine_tune', 'retrain_baseline']
                    if sum([f'--{a}' in arg for arg in sys.argv])]

    append_args_to_loc = '_'.join(append_loc_sci + append_loc_v)

    # print('sys argv TRANSFER', sys.argv)

    model_from = args.model_from.split('/')[-1].split('.')[0]

    save_loc = f'{base_dir}/{args.network}_{model_from}_to_{args.dataset_to}_{append_args_to_loc}'
else:
    save_loc = args.save_loc

model_ckpt = f'{save_loc}/model.ckpt'
plot_loc = f'{save_loc}/metrics.png'
log_loc = f'{save_loc}/log.txt'
args_loc = f'{save_loc}/args.json'

if os.path.exists(save_loc) and args.reset:
    shutil.rmtree(save_loc)

os.makedirs(save_loc, exist_ok=True)

print('using root directory', save_loc)


# json.dumps()

with open(args_loc, 'w') as f:
    f.write(json.dumps(vars(args)))


def log(msg):
    print(msg)
    with open(log_loc, 'a') as f:
        f.write(msg + '\n')


args_log = '\n'.join(f'{k}={v}' for k, v in vars(args).items())
log(args_log + '\n')

torch.manual_seed(4)

# XXX: enable augmentation
dataset_to = get_dataset(args.dataset_to, train_augmentation=False)
train_size = min(args.size, len(dataset_to.train_set)
                 ) if args.size > 0 else len(dataset_to.train_set)
train_set = Subset(dataset_to.train_set, range(train_size))
train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=16)
valid_loader = DataLoader(
    dataset_to.valid_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(
    dataset_to.test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

classifier_state_dict = torch.load(args.model_from, map_location=device)
classifier = classifier_state_dict['model']
classifier_input_shape = classifier_state_dict['input_shape']
log(
    f"Loaded model {args.model_from} ({classifier_state_dict['epoch']} epochs), valid acc {classifier_state_dict['acc']:.3f}")

log(f'Dataset {args.dataset_to} [{train_size}/{len(dataset_to.train_set)}], input shape {classifier_input_shape}')

##################################### Transfer Model #####################################

classes_from = classifier_state_dict['classes']
classes_to = dataset_to.classes

transfer_map = get_transfer_mapping_labels(classes_from, classes_to)
log(f'Transfer map: {get_transfer_mapping_classes(classes_from, classes_to)}\n')

loss_fn = CrossEntropyTransfer(classes_from, classes_to)


# Transfer model input / output channels
in_channels = dataset_to.in_channels
out_channels = classifier_input_shape[0]

if os.path.exists(model_ckpt) and not args.reset:
    print('loading', model_ckpt)
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

    transfer_model = get_model(args.network, in_channels, out_channels)

    optimizer = None

if args.retrain_baseline:
    transfer_model = classifier

transfer_model.to(device)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transfer_model = transfer_model
        self.classifier = classifier

        for p in classifier.parameters():   # freeze classifier
            p.requires_grad = False

        if args.fine_tune:
            for p in classifier.head.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.transfer_model(x)
        x = self.classifier(x)
        return x

    def train(self, mode=True):
        self.transfer_model.train(mode)
        self.classifier.train(False)    # freeze classifier batchnorm layers


if args.retrain_baseline:
    full_model = classifier
else:
    full_model = FullModel()

if optimizer is None:
    parameters = list(transfer_model.parameters())
    if args.fine_tune and not args.retrain_baseline:
        parameters += list(classifier.head.parameters())

    optimizer = torch.optim.Adam(parameters, lr=args.lr)


grayscale_to_rgb = T.Lambda(lambda x: x.repeat_interleave(3, 1))

transform = None
if dataset_to.in_channels == 1 and classifier_input_shape[0] == 3:
    transform = grayscale_to_rgb
elif dataset_to.in_channels == 3 and classifier_input_shape[0] == 1:
    transform = rgb_to_grayscale


no_transfer_valid_acc = test_accuracy(classifier, valid_loader, transform=transform, transfer_map=transfer_map,
                                      name='no transfer valid', device=device)
valid_acc = test_accuracy(full_model, valid_loader, transfer_map=transfer_map,
                          name='valid', device=device)

logs['no_transfer_acc'] = [no_transfer_valid_acc]

##################################### Training #####################################

bn_layers = get_bn_layers(classifier)

bn_losses = [None] * len(bn_layers)


def layer_hook_wrapper(idx):
    def hook(module, inputs, _outputs):
        x = inputs[0]
        bn_losses[idx] = ((module.running_mean - x.mean(dim=[0, 2, 3])).norm(2)
                          + (module.running_var - x.var(dim=[0, 2, 3])).norm(2))
    return hook


if args.f_stats != 0:
    print('adding hooks')
    for l, layer in enumerate(bn_layers):
        layer.register_forward_hook(layer_hook_wrapper(l))


def normalize(x):
    return x.mean(dim=[0, 2, 3]).norm() + (1 - x.std(dim=[0, 2, 3])).norm()


_zero = torch.zeros([1], device=device)

log('Training transfer model ' f'{args.network}, '
    f'params:\t{num_params(transfer_model) / 1000:.2f} K')


full_model.to(device)

for epoch in range(init_epoch, args.num_epochs):
    full_model.train()

    step_start = epoch * len(train_loader)
    for step, (x, y) in enumerate(train_loader, start=step_start):
        x, y = x.to(device), y.to(device)

        # logits = full_model(x)
        x_transfer = transfer_model(x) if not args.retrain_baseline else x

        logits = classifier(x_transfer)
        loss_crit = loss_fn(logits, y)

        loss_bn = args.f_stats * sum(bn_losses) if args.f_stats != 0 else _zero
        loss_reg = args.f_reg * (normalize(x_transfer) + (x - x_transfer).norm()) if args.f_reg != 0 else _zero

        loss = loss_bn + loss_reg

        y_pred = logits.argmax(dim=1)
        acc = accuracy(y_pred, y, transfer_map=transfer_map)

        metrics = {'acc': acc, 'loss_bn': loss_bn.item(), 'loss_reg': loss_reg.item()}

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
    valid_acc = test_accuracy(full_model, valid_loader, transfer_map=transfer_map, device=device)
    metrics['val_acc'] = valid_acc
    interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
    logs['val_acc'].extend(interpolate_valid_acc)
    logs['no_transfer_acc'] = [no_transfer_valid_acc] * ((epoch + 1) * len(train_loader))

    log(f'[{epoch + 1}/{args.num_epochs}] ' +
        ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

    save_model = not args.save_best or valid_acc > best_acc
    save_samples = (epoch + 1) % 10 == 0

    if save_model or save_samples:
        view_batch = next(iter(valid_loader))[0][:32].to(device)
        samples = transfer_model(view_batch)

        if save_samples:
            save_image(make_grid(view_batch.cpu(), normalize=True),
                       f'{save_loc}/sample_input.png')

            save_image(make_grid(samples.cpu(), normalize=True),
                       f'{save_loc}/sample_{epoch + 1:02d}.png')

        if save_model:
            best_acc = valid_acc

            pretty_plot(logs, steps_per_epoch=len(train_loader),
                        smoothing=200, save_loc=plot_loc)
            save_image(make_grid(samples.cpu(), normalize=True),
                       f'{save_loc}/sample_best.png')

            log(f'Saving transfer_model to {model_ckpt}')
            torch.save({'model': transfer_model, 'optimizer': optimizer, 'epoch': epoch + 1,
                        'acc': best_acc, 'logs': logs}, model_ckpt)


# print(logs)
pretty_plot(logs, smoothing=200)
