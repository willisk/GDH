import torch.nn.functional as F
from debug import debug
import tarfile
import gdown
from copy import copy
import zipfile
import urllib
import shutil
import os
import torch
import numpy as np
from torch import nn

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as T

from models import DistortionModelConv

from PIL import Image
import tifffile as tiff

import importlib
import debug
importlib.reload(debug)


IMAGE_FILE_TYPES = ['jpg', 'png', 'tif', 'tiff', 'pt']
os.makedirs('data', exist_ok=True)

INVALID_CLASS = '__INVALID__'

EQUIVALENCE_CLASSES = {
    'BAS': 'basophil',
    'EBO': 'erythroblast',
    'EOS': 'eosinophil',
    'KSC': INVALID_CLASS,
    'LYA': 'lymphocyte',
    'LYT': 'lymphocyte',
    'MMZ': 'ig',
    'MOB': 'monocyte',
    'MON': 'monocyte',
    'MYB': 'ig',
    'MYO': 'ig',
    'NGB': 'neutrophil',
    'NGS': 'neutrophil',
    'PMB': 'ig',
    'PMO': 'ig',
}


def get_dataset(dataset, train_augmentation=False):
    if dataset == 'PBCBarcelona':
        return PBCBarcelona()
    if dataset == 'PBCBarcelona-2x':
        return PBCBarcelona(reduce=2)
    if dataset == 'PBCBarcelona-4x':
        return PBCBarcelona(reduce=4)

    if dataset == 'Cytomorphology':
        return Cytomorphology()
    if dataset == 'Cytomorphology-2x':
        return Cytomorphology(reduce=2)
    if dataset == 'Cytomorphology-4x':
        return Cytomorphology(reduce=4)

    if dataset == 'Cytomorphology-4x-PBC':
        return CytomorphologyPBC(reduce=4)

    if dataset == 'MNIST':
        return MNISTWrapper(train_augmentation)
    if dataset == 'SVHN':
        return SVHNWrapper(train_augmentation)
    elif dataset == 'CIFAR10':
        return CIFAR10Wrapper(train_augmentation)
    elif dataset == 'CIFAR10Distorted':
        return CIFAR10Distorted(train_augmentation)

    raise Exception(f"invalid dataset '{dataset}'")


class TorchDatasetWrapper():
    def __init__(self, train_set, test_set, mean, std, train_augmentation=False, transform=None):
        self.normalize = T.Normalize(mean, std)
        self.unnormalize = NormalizeInverse(mean, std)

        self.augment = T.Compose([
            # T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.RandomApply([T.RandomRotation(15)], p=0.3),
            # T.RandomApply([T.RandomAdjustSharpness(0.45)], p=0.3),
            # T.RandomApply([T.ColorJitter(brightness=.1, hue=.15)], p=0.2),
        ])

        if transform is None:
            self.transform = self.normalize
        else:
            self.transform = T.Compose([transform, self.normalize])

        # no augmentation; needed for distorted dataset creation
        self.full_set = ConcatDataset([copy(train_set), copy(test_set)])

        self.train_set, self.valid_set = random_split_frac(
            train_set, [0.8, 0.2], seed=0)
        self.test_set = Subset(test_set)

        self.train_set.transform = T.Compose(
            [self.augment, self.transform]) if train_augmentation else self.transform
        self.valid_set.transform = self.transform
        self.test_set.transform = self.transform

        labels = train_set.labels if hasattr(
            train_set, 'labels') else train_set.targets
        self.classes = sorted(set([label.item() if isinstance(
            label, torch.Tensor) else label for label in labels]))
        self.input_shape = self.train_set[0][0].shape
        self.in_channels = self.input_shape[0]
        self.num_classes = len(self.classes)


class SVHNWrapper(TorchDatasetWrapper):
    def __init__(self, train_augmentation=False):
        train_set = SVHN(root='data', split='train',
                         transform=T.ToTensor(), download=True)
        test_set = SVHN(root='data', split='test',
                        transform=T.ToTensor(), download=True)
        super().__init__(train_set, test_set,
                         mean=[0.4373, 0.4434, 0.4724], std=[0.1955, 0.1985, 0.1943],
                         train_augmentation=train_augmentation)


class CIFAR10Wrapper(TorchDatasetWrapper):
    def __init__(self, train_augmentation=False):
        train_set = CIFAR10(root='data', train=True,
                            transform=T.ToTensor(), download=True)
        test_set = CIFAR10(root='data', train=False,
                           transform=T.ToTensor(), download=True)
        super().__init__(train_set, test_set,
                         mean=[0.4914, 0.4822, 0.4465], std=[0.2464, 0.2428, 0.2608], train_augmentation=train_augmentation)


class MNISTWrapper(TorchDatasetWrapper):
    def __init__(self, train_augmentation=False):
        train_set = MNIST(root='data', train=True, download=True)
        train_set = MNIST(root='data', train=True,
                          transform=T.ToTensor(), download=True)
        test_set = MNIST(root='data', train=False,
                         transform=T.ToTensor(), download=True)

        transform = nn.ZeroPad2d(2)  # ensure 32x32

        super().__init__(train_set, test_set, mean=[0.1308], std=[0.3081],
                         train_augmentation=train_augmentation, transform=transform)


class ImageFolderDataset(Dataset):
    """Creates a dataset of images in `img_dir` and corresponding masks in `mask_dir`.
    Corresponding mask files need to contain the filename of the image.
    Files are expected to be of the same filetype.

    Args:
        img_dir (str): path to image folder
        mask_dir (str): path to mask folder
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """

    def __init__(self, img_dir, in_channels=3, labels=None, folder_labels=False, transform=None, bits=1):

        self.in_channels = in_channels

        self.img_dir = img_dir
        self.images = list_images_in_dir(img_dir, recursive=True)

        if labels is None:
            assert folder_labels, 'No labels provided and not using folder labels.'
            self.class_labels = [img.split('/')[-2] for img in self.images]
        else:
            self.class_labels = labels

        assert len(self.images) == len(self.class_labels)

        self.classes = sorted(set(self.class_labels))
        self.labels = [self.classes.index(clss) for clss in self.class_labels]
        self.num_classes = len(self.classes)

        self.transform = transform
        self.bits = bits

        self.input_shape = self[0][0].shape

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{len(self.images)}]"
        for n, (img, label) in enumerate(zip(self.images, self.labels)):
            rep += f'\nimage: {img}\tlabel: {label}'
            if n > 10:
                rep += '\n...'
                break
        return rep

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        label = self.labels[idx]

        img = load_image(self.images[idx])
        img = img / (2**self.bits - 1)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class Subset(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"Subset [{len(self)}] of " + repr(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class NormalizeInverse(T.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


class CIFAR10Distorted(ImageFolderDataset):  # , CIFAR10Wrapper):

    # # distorted mean, std
    # mean = [0.9079, 0.7855, 0.4956]
    # std = [0.4956, 0.3020, 0.3052]

    # normalize = T.Normalize(mean, std)
    # unnormalize = NormalizeInverse(mean, std)

    # augment = T.Compose([
    #     T.RandomCrop(32, padding=4),
    #     T.RandomHorizontalFlip(),
    #     T.RandomVerticalFlip(),
    # ])

    # transform_augment = T.Compose([augment, normalize])
    # transform = T.Compose([normalize])

    def __init__(self, train_augmentation=True):

        strength = 1e-1

        img_dir = f'data/CIFAR10_distorted_{strength:1.0e}'

        if not os.path.exists(img_dir):
            create_distorted_dataset(
                'CIFAR10', folder_out=img_dir, strength=strength)

        super().__init__(img_dir, in_channels=3, folder_labels=True)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(
            self, [0.7, 0.15, 0.15], seed=0)

        self.train_set.transform = self.transform_augment if train_augmentation else self.transform
        self.train_set.transform = self.transform
        self.valid_set.transform = self.transform
        self.test_set.transform = self.transform


# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958
class Cytomorphology(ImageFolderDataset):

    def __init__(self, transform=None, reduce=1):

        img_dir = f'data/Cytomorphology'

        if not os.path.exists(img_dir):
            download_Cytomorphology_dataset()

        transform = T.Compose([T.ToTensor(),
                               T.CenterCrop(360),
                               T.Resize((360 // reduce, 360 // reduce)),
                               T.Normalize(mean=[0.8092, 0.7088, 0.8340],
                                           std=[0.1815, 0.2745, 0.1029])
                               ])
        super().__init__(img_dir=img_dir, in_channels=3,
                         folder_labels=True, bits=8, transform=transform)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(
            self, [0.7, 0.15, 0.15], seed=0)


class CytomorphologyPBC(Cytomorphology):
    def __init__(self, transform=None, reduce=1):
        super().__init__(transform, reduce)

        # self.class_labels = [
        #     EQUIVALENCE_CLASSES[clss]
        #     for clss in self.class_labels
        #     if clss in EQUIVALENCE_CLASSES
        # ]
        self.images, self.class_labels = list(zip(*[
            (image, EQUIVALENCE_CLASSES[clss])
            for image, clss in zip(self.images, self.class_labels)
            if clss in EQUIVALENCE_CLASSES
            and EQUIVALENCE_CLASSES[clss] != INVALID_CLASS
        ]))

        self.classes = sorted(set(self.class_labels))
        self.labels = [self.classes.index(clss) for clss in self.class_labels]
        self.num_classes = len(self.classes)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(
            self, [0.7, 0.15, 0.15], seed=0)

        # print('len images', len(self.images))
        # print('len clss labels', len(self.class_labels))
        # print('len labels', len(self.labels))


# https://data.mendeley.com/datasets/snkd93bnjr/1
class PBCBarcelona(ImageFolderDataset):

    def __init__(self, transform=None, reduce=1):

        img_dir = f'data/PBC_Barcelona'

        if not os.path.exists(img_dir):
            download_PBCBarcelona_dataset()

        transform = T.Compose([T.ToTensor(),
                               #    T.CenterCrop(360),
                               T.Resize((360 // reduce, 360 // reduce)),
                               T.Normalize(mean=[0.8734, 0.7481, 0.7215],
                                           std=[0.1593, 0.1864, 0.0801])
                               ])
        super().__init__(img_dir=img_dir, in_channels=3,
                         folder_labels=True, bits=8, transform=transform)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(
            self, [0.7, 0.15, 0.15], seed=0)


class Subset(Dataset):
    """Define a subset of a dataset by only selecting given indices.

    Args:
        dataset (Dataset): full dataset
        indices (list): subset indices
    """

    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"Subset [{len(self)}] of " + repr(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def create_distorted_dataset(dataset, folder_out='auto', strength=0.1, batch_size=64, force=False):

    if folder_out == 'auto':
        folder_out = f'data/{dataset}_distorted_{strength:1.0e}'

    if os.path.exists(folder_out) and not force:
        print(
            f'SKIPPING. Folder "{folder_out}" already exists. Use --force to overwrite.')
    else:
        if os.path.exists(folder_out):
            shutil.rmtree(folder_out)
        print(
            f'Creating distorted version of {dataset} dataset in "{folder_out}". strength={strength}')
        os.makedirs(folder_out, exist_ok=True)

        dataset = get_dataset(dataset)
        data_loader = DataLoader(
            dataset.full_set, batch_size=batch_size, shuffle=False, num_workers=16)

        distortion = None
        counter = 0

        mean = 0
        std = 0

        for x, y in data_loader:
            if distortion is None:
                distortion = DistortionModelConv(
                    input_shape=x.shape[1:], lambd=strength)

            x = distortion(x)

            mean += x.mean(dim=[0, 2, 3])
            std += x.std(dim=[0, 2, 3])

            for img, label in zip(x, y):
                label_dir = os.path.join(folder_out, dataset.classes[label])
                os.makedirs(label_dir, exist_ok=True)
                torch.save(img, os.path.join(label_dir, f'{counter:04d}.pt'))
                counter += 1

        mean /= len(data_loader)
        std /= len(data_loader)

        print('Note: shouldn\'t be taking mean, std from full set')
        print(f'mean: {mean}')
        print(f'std: {std}')


def random_split_frac(dataset, fracs, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    lengths = [int(len(dataset) * frac) for frac in fracs]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    delim_indices = [sum(lengths[:i])
                     for i, l in enumerate(lengths)] + [len(dataset)]
    rand_indices = list(torch.randperm(len(dataset)))

    return [Subset(dataset, rand_indices[delim_indices[i]:delim_indices[i + 1]])
            for i, _ in enumerate(lengths)]


def load_image(path):
    file_type = path.split('.')[-1].lower()
    if file_type == 'pt':
        return torch.load(path)
    # if file_type == 'dng':
    #     return rawpy.imread(path).raw_image_visible
    if file_type == 'tiff' or file_type == 'tif':
        return np.array(tiff.imread(path), dtype=np.float32)[..., :3]
    else:
        return np.array(Image.open(path), dtype=np.float32)


def list_images_in_dir(path, recursive=False):
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    images = [image for image in files
              if image.split('.')[-1].lower() in IMAGE_FILE_TYPES
              and image.split('/')[-1][0] != '.']
    if recursive:
        folders = [folder for folder in files
                   if os.path.isdir(folder) and not folder.split('/')[-1].startswith('.')]
        return images + sum([list_images_in_dir(folder, recursive=recursive) for folder in folders], [])
    return images


# imgs = list_images_in_dir('data/PBC_Barcelona', recursive=True)


def extract_recursive(zip_file, delete_after=True):
    print(f"Extracting '{zip_file}'..")
    data_dir = zip_file.split('.')[0]
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    if delete_after:
        os.remove(zip_file)
    for _file in os.listdir(data_dir):
        if _file.endswith('.zip'):
            extract_recursive(os.path.join(data_dir, _file))


# gdown https://drive.google.com/uc?id=1mJrlBl2vU_qCCKV0f33cc58wCFQcQhle


def download_Cytomorphology_dataset():
    # if os.path.exists('data/Cytomorphology'):
    #     return
    tar_file = 'data/Cytomorphology.tar.gz'
    if not os.path.exists(tar_file):
        print('Downloading Cytomorphology dataset..')
        gdown.download(
            'https://drive.google.com/uc?id=1c4qLxASvtSX8PKLeqnJviGVspS2G7vLM', tar_file)

    if not os.path.exists('data/Cytomorphology'):
        print('Extracting..')
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall('data/Cytomorphology')
    # os.rename('data/PBC_Barcelona_archive/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB',
    #           'data/PBC_Barcelona')
    # shutil.rmtree('data/PBC_Barcelona_archive')


def download_PBCBarcelona_dataset():
    if os.path.exists('data/PBC_Barcelona'):
        return
    zip_file = 'data/PBC_Barcelona_archive.zip'
    if not os.path.exists(zip_file):
        print('Downloading PBC Barcelona dataset..')
        urllib.request.urlretrieve(
            'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/snkd93bnjr-1.zip', zip_file)
    extract_recursive(zip_file)
    os.rename('data/PBC_Barcelona_archive/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB',
              'data/PBC_Barcelona')
    shutil.rmtree('data/PBC_Barcelona_archive')


def identity_map(x):
    return x


def get_transfer_mapping_labels(from_classes, to_classes):

    if (set(from_classes) == set(to_classes)):   # equal sets, return identity mapping
        return {label: {from_classes.index(clss_to)} for label, clss_to in enumerate(to_classes)}

    # assume from_classes < to_classes: unique mapping exists
    transfer_map = {label: {from_classes.index(EQUIVALENCE_CLASSES[clss_to])}
                    if clss_to in EQUIVALENCE_CLASSES and EQUIVALENCE_CLASSES[clss_to] != INVALID_CLASS else set()
                    for label, clss_to in enumerate(to_classes)}

    # if len({*transfer_map.values()}) > 1:
    if len(set().union(*transfer_map.values())) > 1:
        return transfer_map

    # from_classes > to_classes
    transfer_map = {label: {i for i, clss_from in enumerate(from_classes)
                            if EQUIVALENCE_CLASSES[clss_from] == clss_to}
                    for label, clss_to in enumerate(to_classes)}

    assert len(set().union(*transfer_map.values())
               ) > 1, 'error in equivalence classes'

    return transfer_map


def get_transfer_mapping_classes(from_classes, to_classes):

    transfer_map = get_transfer_mapping_labels(from_classes, to_classes)

    transfer_map_classes = {to_classes[k]: {from_classes[v] for v in values}
                            for k, values in transfer_map.items()}
    return transfer_map_classes


class CrossEntropyTransfer():
    def __init__(self, from_classes, to_classes):
        # mask true labels from 'to class' that don't have corresponding pre-image in 'from class'
        # since the model has never seen this / does not have the expressivity (index) to even learn this

        # if from_classes < (subset) to_classes, this is easy: transform to_classes -> from_classes
        # if from_classes > to_classes, equally boost corresponding from_classes with averaged weight

        self.transfer_map = get_transfer_mapping_labels(
            from_classes, to_classes)

        self.unique_mapping = len(
            [v for v in self.transfer_map.values() if len(v) == 1]) == len(self.transfer_map)

    def __call__(self, x, y):
        labels = y.tolist()
        mask = [len(self.transfer_map[label]) > 0 for label in labels]
        labels = [l for l, m in zip(labels, mask) if m]
        x = x[mask]

        if self.unique_mapping:  # from_classes < to_classes; simply map labels
            y = torch.LongTensor([list(self.transfer_map[l])[0]
                                  for l, m in zip(labels, mask) if m]).to(x.device)
            loss = -1 * F.log_softmax(x, 1).gather(1, y.unsqueeze(1))
        else:   # from_classes > to_classes
            y = torch.zeros_like(x)

            for i in range(len(labels)):
                targets = list(self.transfer_map[labels[i]])
                y[i, targets] = 1 / len(targets)

            loss = -1 * F.log_softmax(x, 1) * y
        return loss.mean()


# if __name__ == '__main__':

    # dataset_from = get_dataset('Cytomorphology_4x')
    # dataset_to = get_dataset('PBCBarcelona_4x')

    # from_classes = dataset_from.classes
    # to_classes = dataset_to.classes

    # N = 5
    # torch.manual_seed(0)

    # # from_classes, to_classes = to_classes, from_classes
    # loss_fn = CrossEntropyTransfer(from_classes, to_classes)
    # transfer_map = loss_fn.transfer_map

    # N = 16
    # x = torch.randint(0, len(from_classes), (N, ))
    # y = torch.randint(0, len(to_classes), (N, ))

    # transfer_map, _ = get_transfer_mapping_labels(from_classes, to_classes)
    # # print(f'Transfer map: {transfer_map}')
    # print(
    #     f'Transfer map: {get_transfer_mapping_classes(from_classes, to_classes)}')
    # for true_label, pred in zip(y.tolist(), x.tolist()):
    #     print('(prediction)', from_classes[pred], 'in',
    #           f'{ {from_classes[v] for v in transfer_map[true_label]} } <= {to_classes[true_label]}', '(true_label): ',  pred in transfer_map[true_label])

    # correct = [pred in transfer_map[true_label]
    #            for true_label, pred in zip(y.tolist(), x.tolist())
    #            if len(transfer_map[true_label]) > 0]

    # print(f'{sum(correct)} / {len(correct)}')

    # print(sum(correct) / len(correct))

    # from utils import accuracy
    # print(accuracy(x, y, loss_fn.transfer_map))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as F
    from debug import debug

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = get_dataset('Cytomorphology-4x')

    # img_dir = 'data/CIFAR10_distorted_1e-01'
    # dataset = ImageFolderDataset(img_dir=img_dir, folder_labels=True)
    # debug(dataset)

    train_loader = DataLoader(
        dataset.train_set, batch_size=64, shuffle=True, num_workers=16)
    valid_loader = DataLoader(
        dataset.valid_set, batch_size=64, shuffle=False, num_workers=16)
    test_loader = DataLoader(
        dataset.test_set, batch_size=64, shuffle=False, num_workers=16)

    # dataset = get_dataset('CIFAR10Distorted')
    # loader = DataLoader(dataset.train_set, batch_size=32)

    labels = sum(
        [label_batch.tolist() for batch, label_batch in test_loader], [])

    print(labels)
    # debug(labels)

    # for x, y in train_loader:
    #     print(type(y))
    #     break
    # print(y)
    #     x = dataset.unnormalize(x)

    #     plt.imshow(make_grid(x, normalize=True).permute(1, 2, 0))
    #     plt.show()

    #     plt.imshow(make_grid(T.ColorJitter(brightness=.1, hue=.15)(x), normalize=True).permute(1, 2, 0))
    #     # plt.imshow(make_grid(F.adjust_sharpness(x, 0.45), normalize=True).permute(1, 2, 0))
    #     plt.show()

    #     break
