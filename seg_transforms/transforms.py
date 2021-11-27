# Transforms.py
import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    size = (size, size) if isinstance(size, int) else size
    h, w = size
    min_size = min(img.size)
    ow, oh = img.size
    if ow < w or oh < h:
        padh = h - oh if oh < h else 0
        padw = w - ow if ow < w else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = (min_size, min_size) if isinstance(min_size, int) else min_size
        if max_size is None:
            max_size = min_size
        self.max_size = (max_size, max_size) if isinstance(max_size, int) else max_size

    def __call__(self, image, target):
        h = random.randint(self.min_size[0], self.max_size[1])
        w = random.randint(self.min_size[0], self.max_size[1])
        size = (h, w)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    root = "G:/Codes/RealTime-Segementation/datasets/VOC2012"
    img_path = f"{root}/JPEGImages/2007_000033.jpg"
    label_path = f"{root}/SegmentationObject/2007_000033.png"
    img = Image.open(img_path)
    label = Image.open(label_path).convert('RGB')
    plt.subplot(221), plt.title("Ori Image"), plt.imshow(np.asarray(img))
    plt.subplot(222), plt.title("Ori Label"), plt.imshow(np.asarray(label))
    transforms = Compose([
        RandomResize((256, 256)),
        PILToTensor()
    ])
    img, label = transforms(img, label)
    print(img.shape)
    print(label.shape)
    plt.subplot(223), plt.title("Ori Image"), plt.imshow(np.asarray(img[0]))
    plt.subplot(224), plt.title("Ori Label"), plt.imshow(np.asarray(label))
    plt.show()