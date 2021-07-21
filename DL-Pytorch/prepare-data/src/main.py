import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Dataset: https://www.kaggle.com/c/plant-seedlings-classification

def find_classes(fulldir):
    classes = [d for d in os.listdir(fulldir) if os.path.isdir(os.path.join(fulldir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))
    train = []
    for index, label in enumerate(classes):
        path = fulldir + label + "/"
        for file in os.listdir(path):
            train.append(['{}/{}'.format(label, file), label, index])
    df = pd.DataFrame(train, columns=['files', 'category', 'category_id'])
    return classes, class_to_idx, num_to_class, df


class SeedingDataset(Dataset):
    def __init__(self, filenames, labels, root_dir, subset=False, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fullname = os.path.join(self.root_dir, self.filenames.iloc[idx])
        image = Image.open(fullname).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels.iloc[idx]


if __name__ == "__main__":
    classes, class_to_idx, num_to_class, df = find_classes(
        '/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DL-Pytorch/prepare-data/plant-seedlings-classification/train/')
    X, y = df.drop(['category_id', 'category'], axis=1), df['category_id']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    train_dataset = SeedingDataset(
        X_train.files,
        y_train,
        "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DL-Pytorch/prepare-data/plant-seedlings-classification/train/",
        transform=transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
        ]))
    valid_dataset = SeedingDataset(
        X_val.files,
        y_val,
        "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DL-Pytorch/prepare-data/plant-seedlings-classification/train/",
        transform=transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToTensor(),
        ]))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=30)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=30)

    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)
