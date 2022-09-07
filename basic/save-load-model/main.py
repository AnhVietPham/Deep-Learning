#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 08:34:30 2022

@author: anhvietpham
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
import numpy as np


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


if __name__ == "__main__":
    original_image = io.imread(
        "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/thesis/data/Kvasir-SEG/images/cju1cbokpuiw70988j4lq1fpi.jpg")
    mask_image = io.imread(
        "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/thesis/data/Kvasir-SEG/masks/cju1cbokpuiw70988j4lq1fpi.jpg")
    original_image = resize(original_image, (256, 256), anti_aliasing=True)
    original_image = np.transpose(original_image, (2, 0, 1))
    torch_original_image = torch.tensor(original_image)
    torch_original_image = torch_original_image.float()
    torch_original_image = torch_original_image.unsqueeze(0)
    mask_image = rgb2gray(mask_image)
    mask_image = resize(mask_image, (256, 256), anti_aliasing=True)

    model = FCN2()
    before_predict_mask = model(torch_original_image)

    original_image1 = np.transpose(original_image, (1, 2, 0))

    before_predict_mask = before_predict_mask.squeeze(0)
    before_predict_mask = before_predict_mask.detach().numpy()
    before_predict_mask = np.transpose(before_predict_mask, (1, 2, 0))

    print("Before Model's state dict")
    for param_tenser in model.state_dict():
        print(param_tenser, "\t", model.state_dict()[param_tenser])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/Users/anhvietpham/Downloads/ckpt_FCN2.pth", map_location=device)
    model.load_state_dict(checkpoint['net'])

    print("After Model's state dict")
    for param_tenser in model.state_dict():
        print(param_tenser, "\t", model.state_dict()[param_tenser])

    after_predict_mask = model(torch_original_image)
    after_predict_mask = after_predict_mask.squeeze(0)
    after_predict_mask = after_predict_mask.detach().numpy()
    after_predict_mask = np.transpose(after_predict_mask, (1, 2, 0))

    fig, axes = plt.subplots(1, 4, figsize=(8, 4))
    axes[0].imshow(original_image1)
    axes[0].set_title("Image")

    axes[1].imshow(mask_image)
    axes[1].set_title("Mask")

    axes[2].imshow(before_predict_mask)
    axes[2].set_title("Before Prediction")

    axes[3].imshow(after_predict_mask)
    axes[3].set_title("After Prediction")



