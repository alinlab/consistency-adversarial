import os

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms

DATA_PATH = '/data/'


def get_CIFAR10_C(type='brightness'):
    CIFAR10_C_DIR = f'{DATA_PATH}CIFAR-10-C/'
    CIFAR10_C_PATH = os.path.join(CIFAR10_C_DIR, '%s.npy' % type)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    def _load_data_tensor(path):
        npy_data = np.load(path)
        npy_data = np.transpose(npy_data, (0, 3, 1, 2))
        tensor_data = torch.from_numpy(npy_data)
        return tensor_data
    test_images = _load_data_tensor(CIFAR10_C_PATH)
    test_labels = torch.from_numpy(np.load(os.path.join(CIFAR10_C_DIR, 'labels.npy'))).long()
    test_set = TransformTensorDataset([test_images, test_labels], transform=transform)
    return test_set


def get_CIFAR100_C(type='brightness'):
    CIFAR100_C_DIR = f'{DATA_PATH}CIFAR-100-C/'
    CIFAR100_C_PATH = os.path.join(CIFAR100_C_DIR, '%s.npy' % type)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    def _load_data_tensor(path):
        npy_data = np.load(path)
        npy_data = np.transpose(npy_data, (0, 3, 1, 2))
        tensor_data = torch.from_numpy(npy_data)
        return tensor_data
    test_images = _load_data_tensor(CIFAR100_C_PATH)
    test_labels = torch.from_numpy(np.load(os.path.join(CIFAR100_C_DIR, 'labels.npy'))).long()
    test_set = TransformTensorDataset([test_images, test_labels], transform=transform)
    return test_set


class TransformTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
