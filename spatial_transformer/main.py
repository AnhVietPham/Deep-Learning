import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def downLoadDatasets():
    train_loader = DataLoader(
        dataset=datasets.MNIST(root='.', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1327,), (0.3081,))
                               ])),
        batch_size=32, shuffle=True, num_workers=4)

    test_loader = DataLoader(
        dataset=datasets.MNIST(root='.', train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1327,), (0.3081,))
                               ])),
        batch_size=32, shuffle=True, num_workers=4)


if __name__ == '__main__':
    downLoadDatasets()
