import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# -----------------------------------------------------------------------------
# Original-style transforms (operate on dicts)
# -----------------------------------------------------------------------------
class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size): self.size = size
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': F.resize(image, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)
        }

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5): self.p = p
    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return data

class RandomVerticalFlip(object):
    def __init__(self, p=0.5): self.p = p
    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        return data

class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean, self.std = mean, std
    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

class ComposeDict(object):
    """Compose transforms that accept and return dicts."""
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

# -----------------------------------------------------------------------------
# Dataset with upscaling + tiling
# -----------------------------------------------------------------------------
class FullDataset(Dataset):
    """
    Reads 512×512 images and masks, upscales to 4× size, splits into 4×4 grid
    of tiles (each size×size), applies original transforms per tile.

    Args:
        image_root (str): directory of images
        gt_root    (str): directory of masks
        size       (int): tile size (e.g. 512)
        mode       (str): 'train' or 'test'
    """
    TILE = 512
    UPSCALE = 2048

    def __init__(self, image_root, gt_root, size, mode='train'):
        self.image_paths = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.mask_paths = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith('.png')
        ])
        assert len(self.image_paths) == len(self.mask_paths), "Image/mask count mismatch"
        self.size = size
        self.upscale = size * 4
        self.mode = mode

        # build per-tile transform pipeline
        transforms = []
        # use original Resize to ensure tiles are at `size` (though crop gives size)
        transforms.append(Resize((size, size)))
        if mode == 'train':
            transforms.append(RandomHorizontalFlip(p=0.5))
            transforms.append(RandomVerticalFlip(p=0.5))
        transforms.append(ToTensor())
        transforms.append(Normalize())
        self.transform = ComposeDict(transforms)

    def __len__(self):
        return len(self.image_paths) * 16

    def __getitem__(self, idx):
        img_idx, tile_idx = divmod(idx, 16)
        row, col = divmod(tile_idx, 4)
        y = row * self.size
        x = col * self.size

        # load and upscale original 512×512
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        msk = Image.open(self.mask_paths[img_idx]).convert('L')
        img = img.resize((self.upscale, self.upscale), Image.BICUBIC)
        msk = msk.resize((self.upscale, self.upscale), Image.NEAREST)

        # crop tile
        img_tile = img.crop((x, y, x+self.size, y+self.size))
        msk_tile = msk.crop((x, y, x+self.size, y+self.size))

        data = {'image': img_tile, 'label': msk_tile}
        data = self.transform(data)
        # binarize mask
        data['label'] = (data['label'] > 0.5).float()
        return data

# -----------------------------------------------------------------------------
# TestDataset with load_data interface
# -----------------------------------------------------------------------------
class TestDataset(Dataset):
    """Tiles images/masks in test mode, exposes load_data()."""
    def __init__(self, image_root, gt_root, size):
        # reuse FullDataset in 'test' mode
        self.dataset = FullDataset(image_root, gt_root, size, mode='test')
        self.index = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def load_data(self):
        data = self.dataset[self.index]
        img = data['image'].unsqueeze(0)
        gt = data['label'].squeeze(0).numpy()
        img_idx = self.index // 16
        name = os.path.basename(self.dataset.image_paths[img_idx])
        self.index += 1
        return img, gt, name
