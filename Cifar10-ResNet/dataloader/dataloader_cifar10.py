import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split


def create_ci_far10_data_loader():
    torch.manual_seed(0)
    val_size = 5000
    batch_size = 128
    dataset = CIFAR10(root='data/', download=True,
                      transform=ToTensor()
                      )

    test_dataset = CIFAR10(root='data/', train=False,
                           transform=ToTensor()
                           )
    # class_count = {}
    # for _, index in dataset:
    #     label = dataset.classes[index]
    #     if label not in class_count:
    #         class_count[label] = 0
    #     class_count[label] += 1
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, 10,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size * 2,
        num_workers=2
    )
    return train_loader, val_loader, test_loader
