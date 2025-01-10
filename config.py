from dataclasses import dataclass

import torch


@dataclass
class Config:
    DATA_PATH: str = '/home/dataset'
    BATCH_SIZE: int = 128
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DEVICE: str = 'mps'
    LR: float = 1e-3
    WEIGHT_BIT: int = 8
    DEVICE_BIT: int = 2
    EPOCH: int = 200
    MC_times: int = 1000


# s-rate for different devices
s_factor = {
    'RRAM1': 1,
    'FeFET2': 2,
    'RRAM4': 4,
    'FeFET6': 6,
}

# beta for vgg8
vgg8_beta = {
    'base': {'mnist': 0, 'cifar10': 0},
    'correct': {'mnist': 0, 'cifar10': 0},
    'irs': {'mnist': 1e-2, 'cifar10': 1e-1},
    'ovf': {'mnist': 1e-3, 'cifar10': 1e-3},
}

res18_beta = {
    'base': {'mnist': 0, 'cifar10': 0, 'cifar100': 0, 'tiny': 0},
    'correct': {'mnist': 0, 'cifar10': 0, 'cifar100': 0, 'tiny': 0},
    'irs': {'mnist': 1e-1, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tiny': 1e-2},
    'ovf': {'mnist': 1e-4, 'cifar10': 1e-1, 'cifar100': 1e-1, 'tiny': 1e-2},
}



