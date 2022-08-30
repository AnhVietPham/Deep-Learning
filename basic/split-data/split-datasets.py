#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:48:35 2022

@author: anhvietpham
"""

import os
from glob import glob
from skimage import io
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

x = np.arange(1, 25).reshape(12, 2)
y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0])
print(x)
print(y)

"""
x_train, x_test, y_train, y_test = train_test_split(x, y)
print("====X_train======")
print(x_train)
print("====Y_train======")
print(y_train)
print("====X_test======")
print(x_test)
print("====Y_test======")
print(y_test)
"""

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=5, random_state=4)
print("====X_train======")
print(x_train)
print("====Y_train======")
print(y_train)
print("====X_test======")
print(x_test)
print("====Y_test======")
print(y_test)
"""

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, random_state=4, stratify=y)
print("====X_train======")
print(x_train)
print("====Y_train======")
print(y_train)
print("====X_test======")
print(x_test)
print("====Y_test======")
print(y_test)
"""

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, random_state=4, shuffle=False)
print("====X_train======")
print(x_train)
print("====Y_train======")
print(y_train)
print("====X_test======")
print(x_test)
print("====Y_test======")
print(y_test)
"""

"""
Datasets Kvasir-SEG
"""

path = "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/thesis/data/Kvasir-SEG/"
image_path = glob(os.path.join(path, "images", "*"))
mask_path = glob(os.path.join(path, "masks", "*"))
print("=======Images Path=======")
print(image_path)
print("======Masks Path========")
print(mask_path)

x_train, x_valid = train_test_split(image_path, test_size=0.2, random_state=42)
y_train, y_valid = train_test_split(mask_path, test_size=0.2, random_state=42)
print(x_train[0])
print(y_train[0])

x_image = io.imread(x_train[0])
y_mask = io.imread(y_train[0])
y_mask = rgb2gray(y_mask)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(x_image)
axes[0].set_title("Image")

axes[1].imshow(y_mask)
axes[1].set_title("Mask")


class FCN2(nn.Module):
    def __init__(self, n_class=1):
        super(FCN2, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.layer2 = layers[5]
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.layer3 = layers[6]
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.layer4 = layers[7]
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)
        return out


class KvasirDataLoader(object):

    def __init__(self, path_images, path_masks, transforms):
        self.path_images = path_images
        self.path_masks = path_masks
        self.transforms = transforms

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        image = io.imread(self.path_images[index])
        image = resize(image, (256, 256), anti_aliasing=True)
        image = np.transpose(image, (2, 0, 1))
        mask = io.imread(self.path_masks[index])
        mask = rgb2gray(mask)
        mask = resize(mask, (256, 256), anti_aliasing=True)
        return image, mask


_size = 256, 256
# resize = transforms.Resize(_size, interpolation=0)


# set your transforms
train_transforms = transforms.Compose([
    transforms.Resize(_size, interpolation=0),
    transforms.RandomRotation(180),  # allow any rotation
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(_size, padding=10),  # needed after rotation (with original size)
])

test_transforms = transforms.Compose([
    transforms.Resize(_size, interpolation=0),
])

dataset_train = KvasirDataLoader(x_train, y_train, transforms=train_transforms)
dataset_val = KvasirDataLoader(x_valid, y_valid, transforms=test_transforms)

test = dataset_train.__getitem__(0)

test0 = test[0]

test1 = np.transpose(test[0], (1, 2, 0))

BATCH_SIZE = 10

# Create dataloaders from datasets with the native pytorch functions
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN2(n_class=1)
    model = model.to(DEVICE)
    dataiter = iter(dataloader_train)
    images_dataloader, labels_dataloader = dataiter.next()
    x_image = images_dataloader[0]
    y_mask = labels_dataloader[0]
    imgs, masks = x_image.to(DEVICE), y_mask.to(DEVICE)
    imgs, masks = imgs.float(), masks.float()
    imgs = imgs.unsqueeze(0)
    prediction = model(imgs)

    imgs = imgs.squeeze(0)
    imgs = imgs.detach().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    masks = masks.detach().numpy()
    masks = np.transpose(masks, (1, 2, 0))
    prediction = prediction.squeeze(0)
    prediction = prediction.detach().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].imshow(imgs)
    axes[0].set_title("Image")

    axes[1].imshow(masks)
    axes[1].set_title("Mask")

    axes[2].imshow(prediction)
    axes[2].set_title("Prediction")



