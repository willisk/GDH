import json
import sys
from torchvision.transforms.functional import rgb_to_grayscale
import torchvision.transforms as T
from collections import defaultdict
import os
import shutil
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from utils import calculate_mean_and_std, clamp, num_params, str2bool, test_accuracy, pretty_plot, get_bn_layers, accuracy, labels_correct
from datasets import Subset, get_dataset, get_transfer_mapping_classes, get_transfer_mapping_labels, CrossEntropyTransfer

from models import get_model

from debug import debug
from torchvision.utils import make_grid, save_image

import argparse

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from numpy import savetxt
from sklearn.metrics import plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score

number_workers = 6

os.makedirs('transfer', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--predic', type=int, default=0, help='0: beide, 1: nur model_from, 2: nur model_to')
parser.add_argument('--dataset_from', default='CIFAR10')
parser.add_argument('--dataset_to', default='CIFAR10')
parser.add_argument('--model_from', default='models/model.ckpt',
                    help='Model checkpoint for saving/loading.')
parser.add_argument('--model_to', default='models/model.ckpt',
                    help='Model checkpoint for saving/loading.')
parser.add_argument('--size', type=int, default=4096,
                    help='Sample used for transfer.')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--f_stats', type=float,
                    default=1e-3, help='Stats multiplier')
parser.add_argument('--f_reg', type=float, default=0,
                    help='Regularization multiplier')

parser.add_argument('--unsupervised', type=str2bool,
                    help='Don\'t use label information.')
parser.add_argument('--fine_tune', type=str2bool,
                    help='Fine tune classifier head')
parser.add_argument('--shuffle', type=str2bool, default=True,
                    help='Shuffle training dataset.')

parser.add_argument('--device', default='cuda')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--save_best', action='store_true',
                    help='Save only the best models (measured in valid accuracy).')

parser.add_argument('--save_loc', default='auto',
                    help='Location for saving/loading.')
parser.add_argument('--experiment', type=str, help='Experiment name')
parser.add_argument('--confusion_matrix', default=True)
if 'ipykernel' in sys.argv[0]:
    args = parser.parse_args([
        '--predic', '0',
        '--dataset_from', 'Cytomorphology-4x',
        '--dataset_to', 'PBCBarcelona-4x',
        '--model_from', 'models/Cytomorphology-4x_Resnet18.ckpt',
        '--model_to', 'transfer/Unet_Cytomorphology-4x_Resnet18_to_PBCBarcelona-4x_lr=1e-03_f-stats=1e-07/model.ckpt',
        '--size', '64'
    ])
else:
    args = parser.parse_args()
device = args.device
device = 'cuda:0' 
#checkpoint_path='models/Cytomorphology-4x_Resnet18.ckpt'
classifier_state_dict = torch.load(args.model_from, map_location=device)
classifier = classifier_state_dict['model']
classifier_input_shape = classifier_state_dict['input_shape']

#checkpoint = torch.load(checkpoint_path)
#model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
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
def classification_complete_report5(y_true, y_pred ,labels = None  ): 
    report = classification_report(y_true, y_pred, labels = labels)
    print(report)
    print(15*"----")
    print("matthews correlation coeff: %.4f" % (matthews_corrcoef(y_true, y_pred)) )
    print("Cohen Kappa score: %.4f" % (cohen_kappa_score(y_true, y_pred)) )
    print("Accuracy: %.4f & balanced Accuracy: %.4f" % (accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)) )
    print("macro F1 score: %.4f & micro F1 score: %.4f" % (f1_score(y_true, y_pred, average = "macro"), f1_score(y_true, y_pred, average = "micro")) )
    print("macro Precision score: %.4f & micro Precision score: %.4f" % (precision_score(y_true, y_pred, average = "macro"), precision_score(y_true, y_pred, average = "micro")) )
    print("macro Recall score: %.4f & micro Recall score: %.4f" % (recall_score(y_true, y_pred, average = "macro"), recall_score(y_true, y_pred, average = "micro")) )
    cm = confusion_matrix(y_true, y_pred,labels= labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10)) #plot size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.title(Title)
    savefig(plotname)
    plt.show()
    print(15*"----")
def test_accuracy1(model, data_loader, transform=None, transfer_mapping=None, name=None, device='cuda'):
    i=0
    n=len(data_loader)
    y_pred = np.zeros(n).astype(int)
    y_true = np.zeros(n).astype(int)
    num_total = 0
    num_correct = 0
    model.eval()
    model.to(device)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if transform is not None:
            x = transform(x)
        out = model(x)
        predictions = out.argmax(dim=1).tolist()[0]
        y_true[i]=y.tolist()[0]
        y_pred[i] = predictions
        i=i+1
        #correct = labels_correct(predictions, y, transfer_map=transfer_map, i)
        #num_correct += sum(correct)
        #num_total += len(correct)

    #acc = num_correct / num_total if num_total > 0 else 0   # this shouldn't happen
    #if name is not None:
        #print(f'{name} accuracy: {acc:.3f}')
    return y_true, y_pred
def test_accuracy2(model, data_loader, transform=None, transfer_mapping=None, name=None, device='cuda'):
    i=0
    n=len(data_loader)
    y_pred = np.zeros(n).astype(int)
    y_true = np.zeros(n).astype(int)
    num_total = 0
    num_correct = 0
    model.eval()
    model.to(device)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if transform is not None:
            x = transform(x)
        out = model(x)
        predictions = out.argmax(dim=1).tolist()[0]
        y_true[i]=y.tolist()[0]
        y_pred[i] = list(transfer_mapping[predictions])[0]
        i=i+1
        #correct = labels_correct(predictions, y, transfer_map=transfer_map, i)
        #num_correct += sum(correct)
        #num_total += len(correct)

    #acc = num_correct / num_total if num_total > 0 else 0   # this shouldn't happen
    #if name is not None:
        #print(f'{name} accuracy: {acc:.3f}')
    return y_true, y_pred
if not args.predic==1:
    if not args.model_to==None:
        transfer_state_dict = torch.load(args.model_to, map_location=device)
        transfer_model = transfer_state_dict['model']
        model=FullModel()
        model_to=FullModel()
        model_to.eval()
        pred_to_dataset = get_dataset(args.dataset_to, train_augmentation=False)
        pred_to_data=pred_to_dataset.test_set
        pred_to_loader = DataLoader(
            pred_to_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
        classes_to = pred_to_dataset.classes
        classes_from = classifier_state_dict['classes']
        reverse_map = get_transfer_mapping_labels(classes_to, classes_from)
        reverse_map[3]={7}

        y_true_to, y_pred_to = test_accuracy2(
            model_to, pred_to_loader, transfer_mapping=reverse_map, device=device)
        #y_true, y_pred = test_accuracy2(
        #full_model, pred_loader, transfer_mapping=mapping, device=device)
        label_list=[*classes_to]
        np.savetxt('y_pred_to.csv', y_pred_to, delimiter=',')
        np.savetxt('y_true_to.csv', y_true_to, delimiter=',')
        Title = 'Confusion Matrix'
        plotname='confusion_matrix_to'
        classification_complete_report5(y_true_to, y_pred_to)

if not args.predic==2:
    model_from=classifier
    model_from.eval()
    pred_from_dataset = get_dataset(args.dataset_from, train_augmentation=False)
    pred_from_data=pred_from_dataset.test_set
    pred_from_loader = DataLoader(
    pred_from_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    y_true_from, y_pred_from = test_accuracy1(
        model_from, pred_from_loader, transfer_mapping=None, device=device)
    classes_from = pred_from_dataset.classes
    #y_true, y_pred = test_accuracy2(
    #full_model, pred_loader, transfer_mapping=mapping, device=device)
    label_list=[*classes_from]
    np.savetxt('y_pred_from.csv', y_pred_from, delimiter=',')
    np.savetxt('y_true_from.csv', y_true_from, delimiter=',')
    Title = 'Confusion Matrix'
    plotname='confusion_matrix_from'
    classification_complete_report5(y_true_from, y_pred_from)