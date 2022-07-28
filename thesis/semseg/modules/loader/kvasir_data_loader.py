import numpy as np
import cv2


def parse_image(img_path, image_size):
    image_rgb = cv2.imread(img_path, 1)
    h, w, _ = image_rgb.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = image_rgb / 255.0
    return image_rgb


def parse_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
    mask = np.expand_dims(mask, -1)
    mask = mask / 255.0

    return mask


class KvasirDataLoader(object):

    def __init__(self, path_images, path_masks, transforms):
        self.path_images = path_images
        self.path_masks = path_masks
        self.transforms = transforms

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        image = parse_image(self.path_images[index], 256)
        mask = parse_mask(self.path_masks[index], 256)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(dtype=np.float32)
        mask = np.transpose(mask, (2, 0, 1))
        mask = mask.astype(dtype=np.float32)
        return image, mask
