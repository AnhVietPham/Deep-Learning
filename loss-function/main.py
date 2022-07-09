import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# https://zhang-yang.medium.com/pytorch-loss-funtions-in-plain-python-b79c05f8b53f

def sigmoid(x): return 1 / (1 + np.exp(-x))


def binary_cross_entropy(input, y): return -(input.log() * y + (1 - y) * (1 - input).log()).mean()


if __name__ == '__main__':
    batch_size, n_classes = 10, 4
    x = torch.randn(batch_size, n_classes)
    print(f"Shape: {x.shape}")
    print(x)
    target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)
    print(target)
    y = torch.zeros(batch_size, n_classes)
    print(y)
    y[range(y.shape[0]), target] = 1
    print(y)
    pred = sigmoid(x)
    loss = binary_cross_entropy(pred, y)
    print(loss)
    pred1 = torch.sigmoid(x)
    loss1 = F.binary_cross_entropy(pred1, y)
    print(loss1)
