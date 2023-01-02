import cv2
import glob
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import shutil
import random
import time
from pathlib import Path

# import albumentations as A

import segmentation_models_pytorch as smp

import torchvision.transforms.functional as TF
import torch.nn.functional as F

from skimage import io


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, stride=1, bias=False,
                      dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, stride=1, bias=False,
                      dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Dataset_loader(Dataset):

    def __init__(self, img_tiles, transforms=None):
        self.img_tiles = img_tiles
        self.transforms = transforms
        self.indices = list(range(len(self.img_tiles)))

        # with open(os.path.join(self.data_root + 'image_tiles/'+'0.txt')) as f:
        #     self.img_names = f.read().splitlines()
        #     self.indices = list(range(len(self.img_names)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.transforms is None:
            img = np.transpose(self.img_tiles[self.indices[item]], (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)/255
