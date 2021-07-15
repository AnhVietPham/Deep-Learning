import torch
import torch.nn as nn


def test_reflectionPad(padding):
    m = nn.ReflectionPad2d(padding)
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4)
    out = m(input)
    return out


if __name__ == '__main__':
    print(test_reflectionPad(1))
    x = torch.arange(4, dtype=torch.float).reshape(1, 1, 2, 2)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
    pad = nn.ReflectionPad2d(padding=1)

    out = pad(x)
    out = conv(out)

    conv_pad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=1)
    conv_pad.load_state_dict(conv.state_dict())

    out_conv_pad = conv_pad(x)

    print((out - out_conv_pad).abs().max())
