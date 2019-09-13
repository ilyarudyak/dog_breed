import numpy as np
from glob import glob

from pathlib import Path
import os
import torch

from torchvision import transforms, datasets

home = str(Path.home())

from fastai.vision import *
from fastai.metrics import error_rate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_files():
    human_files = np.array(glob(os.path.join(home, "data/dog_breed/lfw/*/*")))
    dog_files = np.array(glob(os.path.join(home, "data/dog_breed/dogImages/*/*/*")))
    return human_files, dog_files


def get_loaders(batch_size=128):
    data_dir = os.path.join(home, 'data/dog_breed/dogImages')

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)

    # print(len(train_dataset.classes))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_loaders = {'train': train_loader,
                    'valid': valid_loader,
                    'test': test_loader}

    return data_loaders


def get_classes():
    data_dir = os.path.join(home, 'data/dog_breed/dogImages')

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)

    return train_dataset.classes


def get_data_fastai(bs=128):
    path = Path.home() / 'data/dog_breed/dogImages'
    data = ImageDataBunch.from_folder(path=path,
                                      ds_tfms=get_transforms(),
                                      size=224,
                                      bs=bs,
                                      num_workers=0)
    data.normalize()
    return data


if __name__ == '__main__':
    # human_files, dog_files = get_files()
    # print('There are %d total human images.' % len(human_files))
    # print('There are %d total dog images.' % len(dog_files))

    # loaders = get_loaders()
    # classes = get_classes()
    # print(classes)

    data = get_data_fastai()