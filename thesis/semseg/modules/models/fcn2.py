import torch
from torchvision import models
from torch import nn
import torchsummary

base_model = models.resnet18(pretrained=True)


def find_last_layer(layer):
    children = list(layer.children())
    if len(children) == 0:
        return layer
    else:
        return find_last_layer(children[-1])


class FCN2(nn.Module):
    def __init__(self, n_class=1):
        super(FCN2, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())
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
        x = x.type(torch.float32)
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
    layers = list(base_model.children())
    print(layers)
    layer1 = nn.Sequential(*layers[:5])
    print("++++++++++++++++++++++++++")
    print(list(layer1))
    print("++++++++++++Torch-Summary++++++++++++++")
    fcn_model = FCN2(6)
    print(torchsummary.summary(fcn_model, input_size=(3, 224, 224)))
