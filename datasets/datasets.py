import os

import numpy as np
import torch
from torchvision import datasets, transforms

from datasets.cifar_c import get_CIFAR10_C, get_CIFAR100_C


DATA_PATH = '/data/'
CIFAR_C = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
           'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
           'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
           'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def get_transform(augment_type, dataset):
    if dataset == 'tinyimagenet':
        image_size = 64
    else:
        image_size = 32

    if augment_type == 'base':
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif augment_type == 'ccg':
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(int(image_size / 2)))
    else:
        raise NotImplementedError()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataset(P, dataset, image_size=None, download=False):
    train_transform, test_transform = get_transform(P.augment_type, dataset)

    if P.consistency:
        train_transform = MultiDataTransform(train_transform)

    if dataset == 'cifar10':
        image_size = (3, 32, 32)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (3, 32, 32)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'tinyimagenet':
        image_size = (3, 64, 64)
        n_classes = 200

        train_dir = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'train')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dir = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'val')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    else:
        raise NotImplementedError()

    return train_set, test_set, image_size, n_classes


def get_cifar_c(dataset):

    if dataset == 'cifar10':
        get_corrupt = get_CIFAR10_C
    elif dataset == 'cifar100':
        get_corrupt = get_CIFAR100_C
    else:
        raise NotImplementedError()

    cifar_c_dataset = [get_corrupt(corrupt) for corrupt in CIFAR_C]

    return cifar_c_dataset, CIFAR_C
