import torch.nn.functional as F
import os
import sys
import json
import argparse
import importlib

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import dict_product


rename_label = {
    'acc': 'Accuracy',
    'val_acc': 'Validation Accuracy',
}


def parse_json(file):
    file_name = file.split('/')[-1].split('.')[0]
    with open(file, 'r') as f:
        return {'experiment': file_name, **json.loads(f.read())}


# should all be used through the json
default_args = [
    # '--experiment=sizes'
]

# args.json overrides this!
transfer_base_args = [
    # '--dataset_to=PBCBarcelona-4x',
    # '--network=Unet',
    # '--model_from=models/Cytomorphology-4x_Resnet34.ckpt',
    # '--lr=1e-3',
    # '--f_stats=1e-7',
    # '--size=4096',
    # '--batch_size=64',
    # '--num_epochs=30',
    # '--cuda',
]

# args.json overrides this!
param_grid = {
    # # 'lr': [1e-5],
    # 'network': ['Unet', 'UnetSmp', 'BaselineColorMatrix', 'BaselineConv'],
    # 'size': [128, 512, 2048, 8192, 32768],
}


sys.argv = [sys.argv[0]] + default_args + sys.argv[1:]

parser = argparse.ArgumentParser()
# parser.add_argument('--reversed', action='store_true')
parser.add_argument('--reload_results', action='store_true')
parser.add_argument('--reversed', action='store_true')
parser.add_argument('--experiment', type=str, help='experiment name')
parser.add_argument('--json', type=parse_json, help='json experiment settings')
args, transfer_args = parser.parse_known_args()

if args.json is not None:
    transfer_base_args += args.json['transfer_base_args']
    param_grid = args.json['param_grid']

    if args.experiment is None:
        args.experiment = args.json['experiment']


transfer_args = [sys.argv[0], f'--experiment={args.experiment}'] + \
    transfer_base_args + transfer_args


def get_transfer_results(params):
    global transfer_module
    sys.argv = sum([[f'--{k}={v}']
                   for k, v in params.items()], transfer_args)
    if 'transfer_module' not in globals():
        import transfer as transfer_module
    else:
        importlib.reload(transfer_module)
    transfer_results = transfer_module.logs
    transfer_results['args'] = transfer_module.args
    transfer_results['args_log'] = transfer_module.args_log
    return transfer_results


# param_dict_product = dict_product(param_grid)
# if args.reversed:
#     param_dict_product = reversed(param_dict_product)

def get_param_identifier(params):
    return ' '.join([f'{p}={v}' for p, v in params.items()])


param_ids = {get_param_identifier(params): params
             for params in dict_product(param_grid)}

base_dir = f'transfer/{args.experiment}'
results_loc = f'{base_dir}/results.pt' if args.experiment else f'transfer/results_{hash(tuple(param_ids.values()))}.pt'


def load_results():  # useful when running multiple gpus that are accessing the same file
    if os.path.exists(results_loc):
        return torch.load(results_loc)
    return {}


results = load_results()


# ----------- validate results -----------

runs = reversed(param_ids.items()) if args.reversed else param_ids.items()

for i, (id, params) in enumerate(runs):

    if not id in results or args.reload_results:
        print('==========')
        print(f"Run id '{id}' [{i + 1}/{len(runs)}]")
        print('\nParams: \n' +
              '\n'.join(f'{k}={v}' for k, v in params.items()) + '\n')
        print('----------')

        results_id = get_transfer_results(params)
        results = load_results()
        # results[id] = parse_logs(transfer_module.logs)
        results[id] = results_id
        # results['num_epochs'] = results_id.args.num_epochs // not clean
        torch.save(results, results_loc)
    else:
        print(f"Run with id '{id}' found, skipping")

# ----------- plot results -----------

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 15

# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# mpl.rcParams.update({'font.size': MEDIUM_SIZE})


def format_label(label):
    return rename_label[label] if label in rename_label else label.title()


label_param = args.json['plot']['label_param']
x_param = args.json['plot']['x_param']
y_param = args.json['plot']['y_param']

if label_param == 'IDs':
    labels = param_ids.keys()
    use_filter = False

    first_param_values = next(iter(param_grid.values()))

    param_ids_first_param = [next(iter(params.values()))
                             for params in dict_product(param_grid)]

    param_ids_style_types = [first_param_values.index(v)
                             for v in param_ids_first_param]

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    style_cycle = ['solid', 'dotted', 'dashed', 'dashdot',
                   (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5))]
    cycle_length = min(len(style_cycle),
                       len(labels) // len(first_param_values))

    param_ids_styles = [(color_cycle[k], style_cycle[i % cycle_length])
                        for i, k in enumerate(param_ids_style_types)]

else:
    labels = param_grid[label_param]
    use_filter = True
    param_ids_styles = None

# print(labels)


def parse_logs(values):
    return values[-1]
    # return {k: max(v) for k, v in logs.items()}


next = next(iter(results.values()))
no_transfer_acc = parse_logs(next['no_transfer_acc'])
epochs = range(len(next['acc']))


def smoothen(x, smoothing=11):
    if smoothing > 1:
        # x = F.pad(x, (), mode=self.padding_mode),
        # return F.conv1d(torch.Tensor(x).reshape((1, 1, -1)), torch.ones((1, 1, smoothing)) / smoothing, padding='same', padding_mode='reflect').squeeze()
        avg = torch.nn.Conv1d(1, 1, smoothing,
                              bias=False, padding='same', padding_mode='reflect')
        avg.weight.data = torch.ones_like(avg.weight) / smoothing
        return avg(torch.Tensor(x).reshape((1, 1, -1))).squeeze().detach()
    return x


x = epochs if x_param == 'epoch' else param_grid[x_param]

for i, label in enumerate(labels):
    if x_param == 'epoch':
        y = smoothen(results[label][y_param], smoothing=201)
        x_ticks_every = 10  # every 10 epochs
        num_epochs = results[label]['args'].num_epochs
        batch_size = results[label]['args'].batch_size
        runs_per_epoch = len(x) // num_epochs
        skip = runs_per_epoch * x_ticks_every
        x_ticks = list(x[::skip]) + [x[-1]]
        x_ticks_label = [(xi // runs_per_epoch)
                         for xi in x][::skip] + [(x[-1] + 1) // runs_per_epoch]  # inexact, but whatever
        plt.xticks(x_ticks, x_ticks_label)
    else:
        filtered_ids = [id
                        for id, params in param_ids.items()
                        if (params[label_param] == label if use_filter else True)]

        y = [parse_logs(results[id][y_param]) for id in filtered_ids]

        plt.xticks(x)
        plt.xscale('log')

    # print(label, len(x), len(y), len(results[label][y_param]))
    color, linestyle = (None, None) if param_ids_styles is None \
        else param_ids_styles[i]

    plt.plot(x, y, color=color, linestyle=linestyle, label=label)


plt.plot(x, [no_transfer_acc] * len(x),
         label='No Transfer Accuracy', color='black', linestyle='dashed')


title = f"Experiment '{format_label(args.experiment)}'"
# if x_param == 'epoch':
#     num_epochs = results[id]['args'].num_epochs
#     title = title + f' ({num_epochs} epochs)'

plt.title(title)
plt.ylabel(format_label(y_param))
plt.xlabel(format_label(x_param))

# plt.xlim((0, 20))

plt.legend(
    title=format_label(label_param), loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig(f'{base_dir}/results.png', bbox_inches='tight')
plt.savefig(f'experiments/{args.experiment}.png', bbox_inches='tight')
plt.show()
