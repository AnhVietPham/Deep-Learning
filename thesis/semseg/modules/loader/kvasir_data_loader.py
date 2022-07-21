import os
import imageio
import numpy as np
import torch


class KvasirDataLoader(object):

    def __init__(self, ids, path_images, path_masks, transforms):
        self.ids = ids
        self.path_images = path_images
        self.path_masks = path_masks
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path_img = os.path.join(self.path_images, self.ids[index])
        path_mask = os.path.join(self.path_masks, self.ids[index])
        img = imageio.imread(path_img) / 255
        mask = imageio.imread(path_mask)[:, :, 0] / 255
        mask = mask.round()
        img = torch.FloatTensor(np.transpose(img, [2, 0, 1]))
        mask = torch.FloatTensor(mask).unsqueeze(0)
        sample = torch.cat((img, mask), 0)
        sample = self.transforms(sample)
        img = sample[:img.shape[0], ...]
        mask = sample[img.shape[0]:, ...]

        return img, mask
