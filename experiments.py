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
    forced_combinations = args.json['forced_combinations']

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

def get_param_identifier(params):
    return ' '.join([f'{p}={v}' for p, v in params.items()])


def skip_if_forced(params):
    for forced in forced_combinations:
        if list(forced.items())[0] in params.items() and not forced.items() <= params.items():
            return False
    return True


# ids to parameters mapping
# all the transfer.py parameter combinations to be run

# filter by enforcing parameter combinations (because they would have no effect on the outcome)
# if first parameter configuration in element of forced_combination is found, the rest must follow
# otherwise it will be skipped
ids_to_params = {
    get_param_identifier(params): params
    for params in dict_product(param_grid)
    if skip_if_forced(params)
}


base_dir = f'transfer/{args.experiment}'
results_loc = f'{base_dir}/results.pt' if args.experiment else f'transfer/results_{hash(tuple(ids_to_params.values()))}.pt'


def load_results():  # useful when running multiple gpus that are accessing the same file
    if os.path.exists(results_loc):
        return torch.load(results_loc)
    return {}


results = load_results()


# ----------- validate results / run transfer.py experiment -----------

runs = reversed(ids_to_params.items()) if args.reversed else ids_to_params.items()

for i, (id, params) in enumerate(runs):

    if not id in results or args.reload_results:
        print('==========')
        print(f"Run id '{id}' [{i + 1}/{len(runs)}]")
        print('\nParams: \n' +
              '\n'.join(f'{k}={v}' for k, v in params.items()) + '\n')
        print('----------')

        results_id = get_transfer_results(params)
        results = load_results()
        results[id] = results_id
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

def parse_logs(values):
    return values[-1]
    # return {k: max(v) for k, v in logs.items()}


def smoothen(x, smoothing=11):
    if smoothing > 1:
        # x = F.pad(x, (), mode=self.padding_mode),
        # return F.conv1d(torch.Tensor(x).reshape((1, 1, -1)), torch.ones((1, 1, smoothing)) / smoothing, padding='same', padding_mode='reflect').squeeze()
        avg = torch.nn.Conv1d(1, 1, smoothing,
                              bias=False, padding='same', padding_mode='reflect')
        avg.weight.data = torch.ones_like(avg.weight) / smoothing
        return avg(torch.Tensor(x).reshape((1, 1, -1))).squeeze().detach()
    return x



label_param = args.json['plot']['label_param']
x_param = args.json['plot']['x_param']
y_param = args.json['plot']['y_param']

all_ids = ids_to_params.keys()

x = param_grid[x_param]

# labels to ids mapping
# labels are grouped by param_grid[x_param],
# labels do not contain x_param in identifier (or simply contain label_param, if it is != ''), 
# but map to full run id (including x_param)
labels_to_ids = {xi: {
    get_param_identifier({k: v for k, v in params.items() if k != x_param and (k == label_param or label_param == '')}): id
    for id, params in ids_to_params.items() if params[x_param] == xi
} for xi in x}

# same as labels_to_ids mapping, but transposed, to enable iteration
# i.e. x_param is not grouped first, but last
from collections import defaultdict
labels_to_ids_transposed = defaultdict(dict)

for xi, grouped_labels in labels_to_ids.items():
    for label, id in grouped_labels.items():
        labels_to_ids_transposed[label][xi] = id


for i, (label, ids_by_x_param) in enumerate(labels_to_ids_transposed.items()):

    y = [parse_logs(results[ids_by_x_param[xi]][y_param])  for xi in x]

    # plt.xscale('log')
    plt.xticks(x)



    plt.plot(x, y, label=label)

    # # print(label, len(x), len(y), len(results[label][y_param]))
    # color, linestyle = (None, None) if param_ids_styles is None \
    #     else param_ids_styles[i]

    # plt.plot(x, y, color=color, linestyle=linestyle, label=label)



no_transfer_acc = parse_logs(next(iter(results.values()))['no_transfer_acc'])
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
