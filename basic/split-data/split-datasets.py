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
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray


x = np.arange(1, 25).reshape(12, 2)
y = np.array([0,1,1,0,1,0,1,0,1,1,1,0])
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


fig, axes = plt.subplots(1,2, figsize = (8,4))
axes[0].imshow(x_image)
axes[0].set_title("Image")

axes[1].imshow(y_mask)
axes[1].set_title("Mask")



