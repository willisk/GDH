import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as T

from models import DistortionModelConv

from PIL import Image
import tifffile as tiff

import importlib
import debug
importlib.reload(debug)
from debug import debug

import shutil
import urllib
import zipfile
import shutil
from copy import copy


IMAGE_FILE_TYPES = ['jpg', 'png', 'tif', 'tiff', 'pt']
os.makedirs('data', exist_ok=True)


def get_dataset(dataset):
    if dataset == 'PBCBarcelona':
        return PBCBarcelona()
    if dataset == 'PBCBarcelona_2x':
        return PBCBarcelona(reduce=2)
    if dataset == 'PBCBarcelona_4x':
        return PBCBarcelona(reduce=4)
    if dataset == 'MNIST':
        pass
        # return MNISTWrapper()
    elif dataset == 'CIFAR10':
        return CIFAR10Wrapper()
    elif dataset == 'CIFAR10Distorted':
        return CIFAR10Distorted()


# class MNISTWrapper():
#     def __init__(self):
#         self.in_channels = 1
#         self.num_classes = 10

#         train_transform = T.Compose([T.RandomCrop(28, padding=4),   # XXX unnecessary?
#                                      T.RandomHorizontalFlip(),
#                                      T.ToTensor(),
#                                      T.Normalize(0.1307, 0.3080)])
#         test_transform = T.Compose([T.ToTensor(),
#                                     T.Normalize(0.1307, 0.3080)])

#         train_set = MNIST('data', train=True, transform=train_transform, download=True)
#         test_set = MNIST('data', train=False, transform=test_transform, download=True)

#         self.full_set = ConcatDataset(train_set, test_set)

#         self.train_set, self.valid_set = random_split_frac(train_set, [0.8, 0.2], seed=0)
#         self.test_set = test_set

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

        self.labels = labels

        if labels is None:
            assert folder_labels, 'No labels provided and not using folder labels.'
            self.labels = [img.split('/')[-2] for img in self.images]

        assert len(self.images) == len(self.labels)

        self.classes = list(set(self.labels))
        self.labels = [self.classes.index(label) for label in self.labels]
        self.num_classes = len(self.classes)

        self.transform = transform
        self.bits = bits

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


class CIFAR10Wrapper():
    in_channels = 3
    num_classes = 10

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2464, 0.2428, 0.2608]
    normalize = T.Normalize(mean, std)
    unnormalize = NormalizeInverse(mean, std)

    augment = T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomApply([T.RandomRotation(15)], p=0.3),
        # T.RandomApply([T.RandomAdjustSharpness(0.45)], p=0.3),
        # T.RandomApply([T.ColorJitter(brightness=.1, hue=.15)], p=0.2),
    ])

    transform_augment = T.Compose([augment, normalize])
    transform = normalize

    def __init__(self, augment=True):

        train_set = CIFAR10('data', train=True, transform=T.ToTensor(), download=True)
        test_set = CIFAR10('data', train=False, transform=T.ToTensor(), download=True)

        # no augmentation; needed for distorted dataset creation
        self.full_set = ConcatDataset([copy(train_set), copy(test_set)])

        self.train_set, self.valid_set = random_split_frac(train_set, [0.8, 0.2], seed=0)
        self.test_set = Subset(test_set)

        self.train_set.transform = self.transform_augment if augment else self.transform
        self.valid_set.transform = self.transform
        self.test_set.transform = self.transform

        self.classes = train_set.classes


class CIFAR10Distorted(ImageFolderDataset, CIFAR10Wrapper):

    # distorted mean, std
    mean = [0.9079, 0.7855, 0.4956]
    std = [0.4956, 0.3020, 0.3052]

    normalize = T.Normalize(mean, std)
    unnormalize = NormalizeInverse(mean, std)

    augment = T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])

    transform_augment = T.Compose([augment, normalize])
    transform = T.Compose([normalize])

    def __init__(self, augment=True):

        img_dir = f'data/CIFAR10_distorted_1e-01'

        if not os.path.exists(img_dir):
            create_distorted_dataset('CIFAR10', folder_out=img_dir, strength=0.1)

        super().__init__(img_dir, in_channels=3, folder_labels=True)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(self, [0.7, 0.15, 0.15], seed=0)

        self.train_set.transform = self.transform_augment
        self.valid_set.transform = self.transform
        self.test_set.transform = self.transform


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
        super().__init__(img_dir=img_dir, in_channels=3, folder_labels=True, bits=8, transform=transform)

        self.full_set = self
        self.train_set, self.valid_set, self.test_set = random_split_frac(self, [0.7, 0.15, 0.15], seed=0)


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
        print(f'SKIPPING. Folder "{folder_out}" already exists. Use --force to overwrite.')
    else:
        if os.path.exists(folder_out):
            shutil.rmtree(folder_out)
        print(f'Creating distorted version of {dataset} dataset in "{folder_out}". strength={strength}')
        os.makedirs(folder_out, exist_ok=True)

        dataset = get_dataset(dataset)
        train_loader = DataLoader(dataset.full_set, batch_size=batch_size, shuffle=False, num_workers=16)

        distortion = None
        counter = 0

        mean = 0
        std = 0

        for x, y in train_loader:
            if distortion is None:
                distortion = DistortionModelConv(input_shape=x.shape[1:], lambd=strength)

            x = distortion(x)

            mean += x.mean(dim=[0, 2, 3])
            std += x.std(dim=[0, 2, 3])

            for img, label in zip(x, y):
                label_dir = os.path.join(folder_out, dataset.classes[label])
                os.makedirs(label_dir, exist_ok=True)
                torch.save(img, os.path.join(label_dir, f'{counter:04d}.pt'))
                counter += 1

        mean /= len(train_loader)
        std /= len(train_loader)

        print('Note: shouldn\'t be taking mean, std from full set')
        print(f'mean: {mean}')
        print(f'std: {std}')


def random_split_frac(dataset, fracs, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    lengths = [int(len(dataset) * frac) for frac in fracs]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    delim_indices = [sum(lengths[:i]) for i, l in enumerate(lengths)] + [len(dataset)]
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
        return np.array(tiff.imread(path), dtype=np.float32)
    else:
        return np.array(Image.open(path), dtype=np.float32)


def list_images_in_dir(path, recursive=False):
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    images = [image for image in files
              if image.split('.')[-1].lower() in IMAGE_FILE_TYPES
              and image.split('/')[-1][0] != '.']
    if recursive:
        folders = [folder for folder in files if os.path.isdir(folder)]
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


# def k_fold(dataset, n_splits: int, seed: int, train_size: float):
#     """Split dataset in subsets for cross-validation

#        Args:
#             dataset (class): dataset to split
#             n_split (int): Number of re-shuffling & splitting iterations.
#             seed (int): seed for k_fold splitting
#             train_size (float): should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
#        Returns:
#            idxs (list): indeces for splitting the dataset. The list contain n_split pair of train/test indeces.
#     """
#     if hasattr(dataset, 'labels'):
#         x = dataset.images
#         y = dataset.labels
#     elif hasattr(dataset, 'masks'):
#         x = dataset.images
#         y = dataset.masks

#     sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)

#     idxs = []

#     for idxs_train, idxs_test in sss.split(x, y):
#         idxs.append((idxs_train.tolist(), idxs_test.tolist()))

#     return idxs


# if __name__ == '__main__':

#     from utils import calculate_mean_and_std
#     import matplotlib.pyplot as plt
#     dataset = get_dataset('PBCBarcelona_4x')

#     for img, label in dataset:
#         plt.imshow(img.permute(1, 2, 0))
#         plt.title(f'label {label}, class {dataset.classes[label]}')
#         plt.show()
#         break
    # calculate_mean_and_std(train_loader)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from torchvision.utils import make_grid
#     import torchvision.transforms.functional as F

#     torch.manual_seed(0)
#     np.random.seed(0)

#     dataset = get_dataset('CIFAR10Distorted')

#     # img_dir = 'data/CIFAR10_distorted_1e-01'
#     # dataset = ImageFolderDataset(img_dir=img_dir, folder_labels=True)
#     debug(dataset)

#     # dataset = get_dataset('CIFAR10Distorted')
#     # loader = DataLoader(dataset.train_set, batch_size=32)

#     # for x, y in loader:
#     #     x = dataset.unnormalize(x)

#     #     plt.imshow(make_grid(x, normalize=True).permute(1, 2, 0))
#     #     plt.show()

#     #     plt.imshow(make_grid(T.ColorJitter(brightness=.1, hue=.15)(x), normalize=True).permute(1, 2, 0))
#     #     # plt.imshow(make_grid(F.adjust_sharpness(x, 0.45), normalize=True).permute(1, 2, 0))
#     #     plt.show()

#     #     break
