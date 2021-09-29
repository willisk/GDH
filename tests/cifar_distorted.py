import os
import sys
sys.path.append('..')
# if not os.path.exists('Readme.md'):
#     os.chdir('..')

import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader

from datasets import get_dataset
from debug import debug
from utils import calculate_mean_and_std

dataset = get_dataset('SVHN', train_augmentation=False)
loader = DataLoader(dataset.train_set, batch_size=32)

calculate_mean_and_std(loader)


# for i, img in enumerate(dataset.images):
for x, y in loader:

    debug(x)
    print(x.mean())
    print(x.std())
    # plt.title(dataset.classes[y])
    plt.imshow(make_grid(x, normalize=True).permute(1, 2, 0))
    plt.show()
    print([dataset.classes[label] for label in y])

    break
