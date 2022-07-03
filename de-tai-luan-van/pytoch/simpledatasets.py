from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import simulation


class SimpleDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = SimpleDataset(100, transform=trans)
    val_set = SimpleDataset(50, transform=trans)

    image_datasets = {
        'train': train_set,
        'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    print(dataset_sizes)
