import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(3 * 3, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.softmax(F.relu(self.fc1(x)), dim=1)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    # checkpoint = torch.load('../saving-loading-models/best_model.pth')
    # print(checkpoint)

    x = torch.randn(1, 1, 28, 28)
    labels = torch.rand(1, 2)
    model = TheModelClass()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    preds = model(x)
    loss = (preds - labels).sum()
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), '../saving-loading-models/best_model.pth')
