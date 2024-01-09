import torch
from dataclasses import dataclass


@dataclass
class Config:
    DATA_PATH: str = 'load_dataset/datasets'
    BATCH_SIZE: int = 128
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR: float = 1e-3
    WEIGHT_BIT: int = 8
    DEVICE_BIT: int = 2
    EPOCH: int = 200
    MC_times: int = 200
