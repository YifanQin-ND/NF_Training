import sys

import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

from config import Config

sys.path.append('../')


def fetch_dataloader(types, train=True):
    if types == 'mnist':
        train_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        dev_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_set = MNIST(Config.DATA_PATH, train=True, download=True, transform=train_transformer)
        dev_set = MNIST(Config.DATA_PATH, train=False, download=True, transform=dev_transformer)

    elif types == 'cifar10':
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261))
        ])
        dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261))
        ])
        train_set = CIFAR10(Config.DATA_PATH, train=True, download=True, transform=train_transformer)
        dev_set = CIFAR10(Config.DATA_PATH, train=False, download=True, transform=dev_transformer)
        
    elif types == 'cifar100':
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        train_set = CIFAR100(Config.DATA_PATH, train=True, download=True, transform=train_transformer)
        dev_set = CIFAR100(Config.DATA_PATH, train=False, download=True, transform=dev_transformer)
        
    elif types == 'tiny':
        train_transformer = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = ImageFolder(f'{Config.DATA_PATH}/tiny-imagenet-200/train', transform=train_transformer)
        dev_set = ImageFolder(f'{Config.DATA_PATH}/tiny-imagenet-200/val', transform=dev_transformer)

    else:
        raise ValueError(f'Invalid dataset name: {types}')

    train_loader = data.DataLoader(train_set, Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    dev_loader = data.DataLoader(dev_set, Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    if train:
        return train_loader
    else:
        return dev_loader
