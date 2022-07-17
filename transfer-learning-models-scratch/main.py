from torchsummary import summary
import torchvision.models as models
from models.resnet.preactivation_resnet import ReLUBeforeAdditionBottleneck, PreActivationResNet
from models.resnet.resnet import *


def ReLUBeforeAdditionResNet201(num_classes):
    return PreActivationResNet(ReLUBeforeAdditionBottleneck, [12, 16, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    # x = torch.tensor(
    #     [[-2, 1, 2, 6, 4], [-3, 1, 7, 2, -2], [-4, 2, 3, -1, -3], [-7, 1, 2, 3, 11], [5, -7, 8, 12, -9]]).float()
    # print(x)
    # x = x.unsqueeze(0)
    # print(x)
    # y_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    # y_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
    # print(y_1(x))
    # print(y_2(x))
    resnet = ResNet50()
    summary(resnet, (3, 224, 224))
    print(resnet)
