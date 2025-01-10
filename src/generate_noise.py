import sys
import torch
import numpy as np
from config import Config

sys.path.append('../')


# noise for weight

def generate_noise(weight, variation, s_rate=1):
    if variation == 0:
        return torch.zeros_like(weight).to(Config.DEVICE)
    else:
        variation = variation / np.sqrt((s_rate**2 * 0.4 + 0.6))
        var_list = [1, s_rate, s_rate, 1]
        scale = weight.abs().max().item()
        mask = ((0.25 < (weight.abs() / scale)) * ((weight.abs() / scale) < 0.75)).float()
        var1 = 0
        var2 = 0
        for i in range(Config.WEIGHT_BIT // Config.DEVICE_BIT):
            k = 2 ** (2 * i * Config.DEVICE_BIT)
            var1 += k
        for i in range(Config.WEIGHT_BIT // Config.DEVICE_BIT):
            j = 2 ** (i * Config.DEVICE_BIT)
            var2 += j
        # Calculate the standard deviation of the noise based on variation and scale
        var = ((pow(var1, 1 / 2) * scale) / var2) * variation
        w_noise = torch.normal(mean=0., std=var, size=weight.size()).to(Config.DEVICE)
        return w_noise * mask * var_list[1] + w_noise * (1 - mask) * var_list[0]


