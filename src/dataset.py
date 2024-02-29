from collections import defaultdict
import numpy as np
import os
import pandas as pd
import pickle
import json
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import CelebA, CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms


def get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'purchase100':
        return 100
    elif dataset == 'texas100':
        return 100
    elif dataset == 'tiny-imagenet-200':
        return 200
    elif 'celeba' in dataset:
        return 2
    else:
        raise ValueError(f'ERROR: Invalid dataset {dataset}.')


class Purchase100(Dataset):
    """
    Purchase100 dataset pre-processed by Shokri et al. 
    (https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz).
    We save the dataset in a .pickle version because it is much faster to load than
    the original file. Check the notebooks/purchase.ipynb notebook for code
    to generate the .pickle file.
    """
    def __init__(self, root, small=False):
        dataset_path = os.path.join(root, 'purchase-100', 'dataset_purchase.pickle')
        if small:
            dataset_path += '_small'
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)['dataset']
        #dataset = np.loadtxt(dataset_path, dtype=int, delimiter=',')
        self.labels = list(dataset[:, 0] - 1)
        self.records = torch.FloatTensor(dataset[:, 1:])
        assert len(self.labels) == len(self.records), f'ERROR: {len(self.labels)} and {len(self.records)}'
        print(f'Successfully loaded the Purchase100 dataset consisting of {len(self.records)} records',
            f'of {len(self.records[0])} attributes each.')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx], self.labels[idx]


class Texas100(Dataset):
    """
    Texas100 dataset pre-processed by Shokri et al. 
    (https://github.com/privacytrustlab/datasets/blob/master/dataset_texas.tgz).
    We save the dataset in a .pickle version because it is much faster to load than
    the original file. Check the notebooks/texas.ipynb notebook for code
    to generate the .pickle file.
    """  
    def __init__(self, root):
        dataset_path = os.path.join(root, 'texas-100', 'texas', '100', 'dataset.pickle')
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.labels = list(dataset['labels'] - 1)
        self.records = torch.FloatTensor(dataset['features'])
        assert len(self.labels) == len(self.records), f'ERROR: {len(self.labels)} and {len(self.records)}'
        print(f'Successfully loaded the Texas 100 dataset consisting of {len(self.records)} records',
            f'of {len(self.records[0])} attributes each.')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx], self.labels[idx]


def get_label_to_records(dataset):
    """Returns a dictionary of label -> list of record indexes."""
    label_to_records = defaultdict(list)
    record_to_label = dict()
    for r_idx, (_, label) in enumerate(iter(dataset)):
        label_to_records[label].append(r_idx)
        record_to_label[r_idx] = label
    return label_to_records, record_to_label


def get_dataset_name(dataset):
    # Note: the order matters here because CIFAR100 is a subclass of CIFAR10.
    if isinstance(dataset, CIFAR100):
        return 'cifar100'
    elif isinstance(dataset, CIFAR10):
        return 'cifar10'
    elif isinstance(dataset, Purchase100):
        return 'purchase100'
    else:
        raise ValueError(f'ERROR: Invalid dataset type {type(dataset)}')


def load_dataset(dataset, transform, dataset_size, seed, small=False, path='data/'):
    if transform == 'normalize':
        transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif transform == 'resize_normalize':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError(f'ERROR: Unsupported transform {transform}')

    if dataset == 'cifar10':
        train = CIFAR10(root=path, download=True, transform=transform)
        # Note that we use the test split for validation (early stopping), in
        # line with Carlini et al. (IEEE S&P, 2021).
        val = CIFAR10(root=path, train=False, transform=transform)
    elif dataset == 'cifar100':
        train = CIFAR100(root=path, download=True, transform=transform)
        val = CIFAR100(root=path, train=False, transform=transform)
    elif dataset == 'purchase100':
        train = Purchase100(root=path, small=small)
        val = None
    elif dataset == 'texas100':
        train = Texas100(root=path)
        val = None
    elif dataset == 'tiny-imagenet-200':
        train = ImageFolder(os.path.join(path, dataset, 'train'), 
                transform=transform)
        val = None
    elif 'celeba' in dataset:
        def is_smiling(labels):
            return labels[31].item()
        train = CelebA(root=path, download=False, split='all', 
                transform=transform, target_transform=is_smiling)
        val = None
        # Attribute list.
        attr = pd.read_csv(os.path.join(path, 'celeba', 'list_attr_celeba.txt'),
                delim_whitespace=True, header=1)
        idxs_young = []
        idxs_old = []
        for i in range(len(attr)):
            if attr.iloc[i].Young == 1:
                idxs_young.append(i)
            else:
                idxs_old.append(i)
        print("Young count: ", len(idxs_young), "Old count: ", len(idxs_old))
        if dataset == 'celeba-old':
            train = Subset(train, idxs_old)
        elif dataset == 'celeba':
            pass
        else:
            raise ValueError(f'The CelebA version {dataset} is not supported.')
        print(f'Size of {dataset}: {len(train)}')
    else:
        raise ValueError(f'The dataset {dataset} is not supported.')
    
    if dataset_size > 0:
        assert dataset_size <= len(train), \
            f'ERROR: `dataset_size` exceeds the size of the dataset.'
        # Shuffling the dataset and extracting a random subset of `dataset_size`
        # records.
        np.random.seed(seed)
        shuffled_idxs = np.random.permutation(len(train))
        #dataset_idxs = shuffled_idxs[:dataset_size]
        train = Subset(train, shuffled_idxs[:dataset_size])
    #dataset_idxs = np.arange(len(train))
    # The `train` split will be used to draw members (training records) and
    # non-members (unseen, test records), hence the 'train_and_test' name.
    return {'train_and_test': train, 'val': val}
