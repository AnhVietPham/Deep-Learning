import os
from glob import glob

import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from thesis.semseg.modules.models.fcn2 import FCN2
from thesis.semseg.modules.loader.kvasir_data_loader import KvasirDataLoader


# working with images

class IoUBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = - (intersection + smooth) / (union + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        IoU_BCE = BCE + IoU

        return IoU_BCE


def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):
    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W -> BATCH x H x W
    labels = labels.squeeze(1).byte()

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()


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
    # train_features, train_labels = next(iter(dataloader_train))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0].squeeze()
    # plt.imshow(label)
    # plt.show()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device for training, i.e. gpu or cpu
    # Begin training
    model = FCN2(n_class=1)
    model = model.to(DEVICE)  # load model to DEVICE
    epochs = 2
    patience = 10
    # Define optimiser and criterion for the training. You can try different ones to see which works best for your data and task
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

    # criterion = IoULoss()
    # model_name = 'UNet_IoULoss_baseline'

    # criterion = nn.BCEWithLogitsLoss()
    # model_name = 'UNet_BCELoss_baseline'

    criterion = IoUBCELoss()
    model_name = 'FCN2'

    train_losses = []
    val_losses = []
    best_iou = 0
    best_loss = np.Inf
    best_epoch = -1
    state = {}

    for epoch in range(epochs):
        running_loss = 0
        running_iou = 0
        # Train
        model.train()
        for i, (imgs, masks) in enumerate(dataloader_train):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            prediction = model(imgs)

            optimiser.zero_grad()
            loss = criterion(prediction, masks)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}".format(epoch, epochs, i, len(dataloader_train),
                                                                             running_loss / (i + 1)), end="")

            running_iou += iou_pytorch_eval(prediction, masks)
            print("\r Epoch: {} of {}, Iter.: {} of {}, IoU:  {:.6f}".format(epoch, epochs, i, len(dataloader_train),
                                                                             running_iou / (i + 1)), end="")

        # Validate
        model.eval()
        val_loss = 0
        val_iou = 0
        for i, (imgs, masks) in enumerate(dataloader_val):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            prediction = model(imgs)
            loss = criterion(prediction, masks)
            val_loss += loss.item()
            print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}, Val. Loss: {:.6f}".format(epoch, epochs,
                                                                                                len(dataloader_train),
                                                                                                len(dataloader_train),
                                                                                                running_loss / len(
                                                                                                    dataloader_train),
                                                                                                val_loss / (i + 1)),
                  end="")

            val_iou += iou_pytorch_eval(prediction, masks)
            print("\r Epoch: {} of {}, Iter.: {} of {}, IoU: {:.6f}, Val. IoU: {:.6f}".format(epoch, epochs,
                                                                                              len(dataloader_train),
                                                                                              len(dataloader_train),
                                                                                              running_iou / len(
                                                                                                  dataloader_train),
                                                                                              val_iou / (i + 1)),
                  end="")

        # compute overall epoch losses
        epoch_train_loss = running_loss / len(dataloader_train)
        train_losses.append(epoch_train_loss)
        epoch_val_loss = val_loss / len(dataloader_val)
        val_losses.append(epoch_val_loss)
        # compute overall epoch iou-s
        epoch_train_iou = running_iou / len(dataloader_train)
        epoch_val_iou = val_iou / len(dataloader_val)

        print("\r Epoch: {} of {}, Iter.: {} of {}, Train Loss: {:.6f}, IoU: {:.6f}".format(epoch, epochs,
                                                                                            len(dataloader_train),
                                                                                            len(dataloader_train),
                                                                                            epoch_train_loss,
                                                                                            epoch_train_iou))
        print("\r Epoch: {} of {}, Iter.: {} of {}, Valid Loss: {:.6f}, IoU: {:.6f}".format(epoch, epochs,
                                                                                            len(dataloader_train),
                                                                                            len(dataloader_train),
                                                                                            epoch_val_loss,
                                                                                            epoch_val_iou))

        # # plot
        # plt.figure(figsize=(18, 9))
        # plt.plot(np.arange(len(train_losses)), train_losses,
        #          label=f'Train, loss: {epoch_train_loss:.4f}, IoU: {epoch_train_iou:.4f}', linewidth=3)
        # plt.plot(np.arange(len(val_losses)), val_losses,
        #          label=f'Valid, loss: {epoch_val_loss:.4f}, IoU: {epoch_val_iou:.4f}', linewidth=3)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title(f'Epoch {epoch}')
        # plt.legend(loc='best')
        # plt.show()

        # save if best results or break is has not improved for {patience} number of epochs
        best_iou = max(best_iou, epoch_val_iou)
        best_loss = min(best_loss, epoch_val_loss)
        best_epoch = epoch if best_iou == epoch_val_iou else best_epoch

        # record losses
        state['train_losses'] = train_losses
        state['val_losses'] = val_losses

        if best_epoch == epoch:
            # print('Saving..')
            state['net'] = model.state_dict()
            state['iou'] = best_iou
            state['epoch'] = epoch

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'../checkpoints/ckpt_{model_name}.pth')

        elif best_epoch + patience < epoch:
            print(f"\nEarly stopping. Target criteria has not improved for {patience} epochs.\n")
            break
