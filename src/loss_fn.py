from scipy.special import lmbda
import torch
import torch.nn as nn


def negative_feedback(output_backbone, label, beta, criterion, outs):
    nf_out = output_backbone - beta * sum((10**(-i)) * output for i, output in enumerate(outs))
    loss = criterion(nf_out / (len(outs)+1), label)
    return loss


