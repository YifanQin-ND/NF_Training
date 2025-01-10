from scipy.special import lmbda
import torch
import torch.nn as nn


def negative_feedback(output_backbone, label, beta, criterion, outs):
    nf_out = output_backbone - beta * sum((10**(-i)) * output for i, output in enumerate(outs))
    loss = criterion(nf_out / (len(outs)+1), label)
    return loss


def correct_net(output, label, criterion, model):
    beta = 5e-4
    lmd = 1
    loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            w = m.weight
            shape1, shape2 = w.size(-2), w.size(-1)
            w = w.view(-1, shape1, shape2)
            # w: [batch, shape1, shape2]
            # w.swapaxes(-1, -2): [batch, shape2, shape1]
            # => (W^T W) - lmd*I
            loss += (w.swapaxes(-1, -2).bmm(w) - lmd * torch.eye(shape2).to(w.device)).pow(2).sum()
    loss = criterion(output, label) + beta * loss
    return loss

