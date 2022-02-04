import torch
from torch.utils.data import DataLoader
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from datasets import get_dataset #seed should be set to 4
from numpy import savetxt
from sklearn.metrics import plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score

'''settings: here you have to make your spezifications'''

#prediction='tensor_pred.pt'
prediction=None # or path to load prediction from, eg.: 'tensor_pred.pt'
dataset='Cytomorphology-4x'
model_ckpt='models/Cytomorphology-4x_Resnet34.ckpt'
device = 'cuda:0'


'''main'''
state_dict = torch.load(model_ckpt, map_location=device)
model = state_dict['model']
optimizer = state_dict['optimizer']
init_epoch = state_dict['epoch']
logs = state_dict['logs']
best_acc = state_dict['acc']

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).to(device)
    for batch in loader:
        images, labels = batch
        #images, labels = images.to(device), labels.to(device)

        preds = model(images.to(device))
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds
dataset = get_dataset(dataset, train_augmentation=False)
test_data=dataset.test_set
#test_data=dataset
#test_data=dataset.train_set
test_loader = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
model.to(device)
if prediction==None:
    with torch.no_grad():
        train_preds = get_all_preds(model, test_loader)
        torch.save(train_preds, 'tensor_pred_test.pt')
else: train_preds = torch.load(prediction) 
a=dataset.test_set
n=len(a)
print(n)
ar = np.zeros(n).astype(int)
i=0
while i <= (n-1):
    ar[i] = a[i][1]
    i=i+1

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

preds_correct = get_num_correct(train_preds.to(device), torch.as_tensor(ar).to(device))
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(a))
y_pred=np.array(train_preds.argmax(dim=1).tolist())
np.savetxt('y_pred.csv', y_pred, delimiter=',')
y_true=np.array(ar)
np.savetxt('y_true.csv', y_true, delimiter=',')
Title = 'Confusion Matrix'
plotname='confusion_matrix'
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.title(Title)
    savefig(plotname)
    plt.show()
    print(15*"----")

classification_complete_report5(y_true, y_pred)