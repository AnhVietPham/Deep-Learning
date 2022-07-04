import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# https://www.cs.toronto.edu/~lczhang/360/lec/w04/convnet.html

fc = nn.Linear(100, 30)


def func1():
    x = torch.randn(100)
    print(x)
    y = fc(x)
    print(y)
    print(f"Shape: {y.shape}")
    print("=====Understand=======")
    conv = nn.Conv2d(in_channels=3,
                     out_channels=7,
                     kernel_size=5)
    x1 = torch.randn(32, 3, 128, 128)
    print(x1)
    y1 = conv(x1)
    print(y1)
    print(f"Shape: {y1.shape}")


def func2():
    image_path = "./imgs/dog_mochi.png"
    img = plt.imread(image_path)
    plt.imshow(img)


def func3(img):
    x = torch.from_numpy(img)
    print(x.shape)
    x = x.permute(2, 0, 1)
    print(x.shape)
    x = x.reshape([1, 3, 241, 145])
    print(x.shape)


def func4():
    conv = nn.Conv2d(in_channels=3,
                     out_channels=7,
                     kernel_size=5)

    image_path = "./imgs/dog_mochi.png"
    img = plt.imread(image_path)
    x = torch.from_numpy(img)
    x = x.permute(2, 0, 1)
    x = x.reshape([1, 3, 241, 145])
    y = conv(x)
    y = y.detach().numpy()
    print(y.shape)
    y = y[0]
    print(y.shape)
    # [0,1]
    y_max = np.max(y)
    print(f"Y Max: {y_max}")
    y_min = np.min(y)
    print(f"Y Max: {y_min}")
    img_after_conv = y - y_min / (y_max - y_min)
    print(img_after_conv.shape)
    print(img_after_conv)


def func5():
    fc = nn.Linear(100, 30)
    fc_params = list(fc.parameters())
    print(f"len(fc_params): {len(fc_params)}")
    print(f"Weights: {fc_params[0].shape}")
    print(f"Bias: {fc_params[1].shape}")

    conv = nn.Conv2d(in_channels=3,
                     out_channels=7,
                     kernel_size=5,
                     padding=2)
    conv_params = list(conv.parameters())
    print(f"len(conv_params): {len(conv_params)}")
    print(f"Filters: {conv_params[0].shape}")
    print(f"Biases: {conv_params[1].shape}")


class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.name = "large"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


if __name__ == '__main__':
    model = LargeNet()
    print(model)
    print("==========")
