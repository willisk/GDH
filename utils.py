import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


@torch.no_grad()
def test_accuracy(model, data_loader, name='test', device='cuda'):
    num_total = 0
    num_correct = 0
    model.eval()
    model.to(device)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    print(f'{name} accuracy: {acc:.3f}')
    return acc


def transpose_dict(d):
    if isinstance(d, dict):
        return [{k: v for k, v in zip(d.keys(), vals)} for vals in zip(*d.values())]
    elif isinstance(d, list):
        return {k: v for k, v in zip(d[0].keys(), zip(*[e.values() for e in d]))}


def pretty_plot(logs, steps_per_epoch=1, smoothing=0, save_loc=None):
    vals = torch.Tensor(list(logs.values()))
    if smoothing:
        vals = F.conv1d(vals.reshape((len(vals), 1, -1)), torch.ones((1, 1, smoothing)) / smoothing).squeeze()
    y_max = max(vals.mean(dim=1) + vals.std(dim=1) * 1.5)
    x = torch.arange(vals.shape[1]) / steps_per_epoch

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(14, 8))
    plt.xlabel('epoch')
    main_axis = plt.gca()
    main_axis.set_ylim([0, y_max])
    scaled_axis = main_axis.twinx()
    scaled_axis.set_ylabel('%')
    scaled_axis.set_ylim([-10, 110])

    for i, (m, values) in enumerate(zip(logs, vals)):
        axis = main_axis
        if 'acc' in m:
            values *= 100
            axis = scaled_axis
        axis.plot(x, values, label=m, c=color_cycle[i])

    legend = main_axis.legend(loc=2)
    legend.remove()
    scaled_axis.legend(loc=1)
    scaled_axis.add_artist(legend)

    if save_loc is not None:
        plt.savefig(save_loc)
        # plt.savefig(save_loc, transparent=True)
        # plt.savefig(save_loc.replace('.pdf', '.png'))

    plt.show()


def get_file_name(path):
    return path.split('/')[-1].split('.')[0]


def get_bn_layers(net):
    # ignore_types = ['activation', 'loss', 'container', 'pooling']
    all_layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0:
            if 'batchnorm' in layer.__module__:
                all_layers.append(layer)
        else:
            all_layers += get_bn_layers(layer)
    return all_layers


@torch.no_grad()
def calculate_mean_and_std(data_loader):
    mean = 0
    std = 0
    for x, y in data_loader:
        mean += x.mean(dim=[0, 2, 3])
        std += x.std(dim=[0, 2, 3])
    mean /= len(data_loader)
    std /= len(data_loader)
    print(f'mean: {mean}')
    print(f'std: {std}')
