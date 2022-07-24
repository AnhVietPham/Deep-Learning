import os
from glob import glob

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from thesis.semseg.modules.loader.kvasir_data_loader import KvasirDataLoader

# working with images

if __name__ == "__main__":
    _size = 256, 256
    resize = transforms.Resize(_size, interpolation=0)

    # set your transforms
    train_transforms = transforms.Compose([
        transforms.Resize(_size, interpolation=0),
        transforms.RandomRotation(180),  # allow any rotation
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(_size, padding=10),  # needed after rotation (with original size)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(_size, interpolation=0),
    ])

    train_path = "../new_data/Kvasir-SEG/train/"
    valid_path = "../new_data/Kvasir-SEG/valid/"

    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_mask_paths.sort()
    valid_mask_paths.sort()

    dataset_train = KvasirDataLoader(train_image_paths, train_mask_paths, transforms=train_transforms)
    dataset_val = KvasirDataLoader(valid_image_paths, valid_mask_paths, transforms=test_transforms)

    BATCH_SIZE = 20

    # Create dataloaders from datasets with the native pytorch functions
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # Display image and label.
    train_features, train_labels = next(iter(dataloader_train))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
