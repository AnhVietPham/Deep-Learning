import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms

dataset_dir = 'plant-pathology-2020-fgvc7/images'
root = 'plant-pathology-2020-fgvc7/'
train = pd.read_csv(os.path.join(root, 'train.csv'))
test = pd.read_csv(os.path.join(root, 'test.csv'))
submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
images = os.path.join(root, 'images')
col = ['healthy', 'multiple_diseases', 'rust', 'scab']

BATCH_SIZE = 10
epochs = 20
learbing_rate = 5e-5


def get_num_files(path):
    """
    Counts the number of files in a folder.
    """
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


def get_path(image):
    return os.path.join(root, 'images', str(image) + '.jpg')


class PlantPathologyDataset(torch.utils.data.Dataset):
    def __init__(self, df, labels=None, test=False, transforms=None):
        self.df = df
        self.test = test
        if self.test == False:
            self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path)

        if self.test == False:
            labels = torch.tensor(np.argmax(self.labels.iloc[idx, :]))

        if self.transforms:
            transformed = self.transforms(image)

        if self.test == False:
            return transformed, labels
        return transformed

    def __len__(self):
        return self.df.shape[0]


def fit(epochs, model, criteria, optimizer):
    for epoch in range(epochs + 1):
        training_loss = 0.0
        correct = 0.0
        total = 0.0
        model.train()
        for batch in train_loader:
            data = batch[0]
            target = batch[1]

            optimizer.zero_grad()
            output = model(data)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * data.shape[0]
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += data.shape[0]
        print(
            f'{epoch + 1}/{epochs}... Traning loss {training_loss / len(train_loader.dataset)} : Traning accuracy {correct / len(train_loader.dataset)}')


def test_function(model, loader):
    preds_for_output = np.zeros((1, 4))
    with torch.no_grad():
        for images in loader:
            images = images
            model.eval()
            predictions = model(images)
            preds_for_output = np.append(preds_for_output, predictions.cpu().detach().numpy(), axis=0)
        preds_for_output = np.delete(preds_for_output, 0, 0)
    return preds_for_output


if __name__ == '__main__':
    train_data = train.copy()
    train_data['image_path'] = train_data['image_id'].apply(get_path)
    train_labels = train.loc[:, 'healthy':'scab']
    test_data = test.copy()
    test_data['image_path'] = test_data['image_id'].apply(get_path)
    test_paths = test_data['image_path']

    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms_imagenet = transforms.Compose([
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     normalize_imagenet,
    # ])
    # plant_pathology_dataset = PlantPathologyDataset(
    #     df=train_data,
    #     labels=train_labels,
    #     transforms=transforms_imagenet)
    #
    # val_size = int(np.floor(0.2 * len(plant_pathology_dataset)))
    # train_size = len(plant_pathology_dataset) - val_size
    # train_dataset, val_dataset = random_split(plant_pathology_dataset, [train_size, val_size])
    #
    # test_dataset = PlantPathologyDataset(df=test_data, test=True, transforms=transforms_imagenet)
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset))
    # valid_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
    # Creating a dataframe with 75%
    # values of original dataframe
    train_data_75 = train_data.sample(frac=0.75)
    train_labels_75 = train_data_75.loc[:, 'healthy':'scab']
    # Creating dataframe with
    # rest of the 25% values
    valid_data_25 = train_data.drop(train_data_75.index)
    valid_labels_25 = valid_data_25.loc[:, 'healthy':'scab']

    train_dataset = PlantPathologyDataset(
        df=train_data_75,
        labels=train_labels_75,
        transforms=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize_imagenet
        ]))
    valid_dataset = PlantPathologyDataset(
        df=valid_data_25,
        labels=valid_labels_25,
        transforms=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize_imagenet
        ]))
    train_loader_normalization = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=10)
    valid_loader_normalization = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=10)
    images, labels = next(iter(train_loader_normalization))
    print(images.shape)
    print(labels.shape)
    # model_conv = models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 4),
    #     nn.Softmax(dim=1)
    # )
    # criterion = nn.CrossEntropyLoss()
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # fit(1, model_conv, criterion, optimizer_conv)
    # pre = test_function(model_conv, test_loader)
    # print(pre)
    # images = next(iter(test_loader))
    # print(images.shape)
    # images, labels = next(iter(train_loader))
    # print(images.shape)
    # print(labels)
