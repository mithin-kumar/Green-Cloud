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

import albumentations as A

import segmentation_models_pytorch as smp

import torchvision.transforms.functional as TF
import torch.nn.functional as F

from skimage import io

from extrafiles import DoubleConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512],
                 rates=(1, 1, 1, 1)):
        super(UNet, self).__init__()
        self.down_part = nn.ModuleList()
        self.up_part = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Part
        for i, feature in enumerate(features):
            self.down_part.append(DoubleConv(
                in_channels, feature, dilation=rates[i]))
            in_channels = feature
        # Decoder Part
        for i, feature in enumerate(reversed(features)):
            self.up_part.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_part.append(DoubleConv(
                2*feature, feature, dilation=rates[i]))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.output = nn.Conv2d(
            features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        for down in self.down_part:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_part), 2):
            x = self.up_part[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_part[idx + 1](concat_skip)

        return self.output(x)
