import torch
from torch import nn

if __name__ == '__main__':
    m = nn.AdaptiveAvgPool2d((5, 7))
    input = torch.randn(1, 64, 8, 9)
    output = m(input)
    print(f'Input Shape: {input.shape}')
    print(f'Output Shape: {output.shape}')
    m1 = nn.AdaptiveAvgPool2d(9)
    input1 = torch.randn(3, 3, 10, 11)
    output1 = m1(input1)
    print(f'Input1 shape: {input1.shape}')
    print(f'Output1 shape: {output1.shape}')
