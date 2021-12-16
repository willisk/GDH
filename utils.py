from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from itertools import product
# from datasets import equivalence_classes


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def labels_correct(y_pred, y, transfer_map=None):
    if transfer_map is None:
        return (y_pred == y).tolist()
    y_mapped = [transfer_map[yi] for yi in y.tolist()]
    return [pred in mapped
            for mapped, pred in zip(y_mapped, y_pred.tolist())
            if len(mapped) > 0]


def accuracy(y_pred, y, transfer_map=None):
    if transfer_map is None:
        return (y_pred == y).float().mean().item()
    correct = [pred in transfer_map[true_label]
               for true_label, pred in zip(y.tolist(), y_pred.tolist())
               if len(transfer_map[true_label]) > 0]
    return sum(correct) / len(correct) if len(correct) > 0 else 0


@torch.no_grad()
def test_accuracy(model, data_loader, transform=None, transfer_map=None, name=None, device='cuda'):
    num_total = 0
    num_correct = 0
    model.eval()
    model.to(device)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if transform is not None:
            x = transform(x)
        out = model(x)
        predictions = out.argmax(dim=1)
        correct = labels_correct(predictions, y, transfer_map=transfer_map)
        num_correct += sum(correct)
        num_total += len(correct)

    acc = num_correct / num_total if num_total > 0 else 0   # this shouldn't happen
    if name is not None:
        print(f'{name} accuracy: {acc:.3f}')
    return acc


def transpose_dict(d):
    if isinstance(d, dict):
        return [{k: v for k, v in zip(d.keys(), vals)} for vals in zip(*d.values())]
    elif isinstance(d, list):
        return {k: v for k, v in zip(d[0].keys(), zip(*[e.values() for e in d]))}


# def rbg_to_luminance(x):
#     assert x.ndim == 4
#     return (x[:, 0, None, :, :] * 0.2126 +
#             x[:, 1, None, :, :] * 0.7152 +
#             x[:, 2, None, :, :] * 0.0722)


def pretty_plot(logs, steps_per_epoch=1, smoothing=0, save_loc=None):

    max_len = max([len(v) for v in logs.values()])
    dashed = [k for k, v in logs.items() if len(v) == 1]
    logs = {k: v * max_len if len(v) == 1 else v for k, v in logs.items()}

    vals = torch.Tensor(list(logs.values()))
    if smoothing and len(vals) > smoothing:
        vals = F.conv1d(vals.reshape((len(vals), 1, -1)),
                        torch.ones((1, 1, smoothing)) / smoothing, padding='same').squeeze()
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
        axis.plot(x, values, label=m, c=color_cycle[i],
                  linestyle='dashed' if m in dashed else None)

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


def get_layers(net, layer_type):
    all_layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0:
            if isinstance(layer, layer_type):
                all_layers.append(layer)
        else:
            all_layers += get_layers(layer, layer_type)
    return all_layers


def get_bn_layers(net):
    return get_layers(net, nn.BatchNorm2d)


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


def total_variation(x):
    tv = ((x[:, :, :, :-1] - x[:, :, :, 1:]).norm()
          + (x[:, :, :-1, :] - x[:, :, 1:, :]).norm()
          + (x[:, :, 1:, :-1] - x[:, :, :-1, 1:]).norm()
          + (x[:, :, :-1, :-1] - x[:, :, 1:, 1:]).norm())
    return tv


def dict_product(grid):
    return [dict(zip(grid.keys(), v)) for v in product(*grid.values())]


# def list_all_files(_dir):
# files = sum([
#     [
#         os.path.join(root, file)
#         for file in files
#         if 'args.json' in file
#     ]
#     for root, dirs, files in os.walk(_di)
# ], [])

def clamp(x, _min, _max):
    return max(min(_max, x), _min)


def str2bool(x):
    return False if x == '0' or x == 'False' or not x else True
