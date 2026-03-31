import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

"""
The below code is taken from:
https://github.com/milesial/Pytorch-UNet
"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
class PositiveUNet(UNet):
    """
    UNet variant that forces the first output channel to be positive.

    Changes from the original:
    - Applies the chosen post-activation (Softplus by default or ReLU) only to
      the first output channel (channel 0).
    - Clamps the final outputs to be at least `min_value`.

    Parameters added:
    - softplus_beta: Softplus beta parameter (if using Softplus).
    - min_value: minimum allowed output value (float).
    - use_relu: if True, use ReLU instead of Softplus for the first channel.
    """
    def __init__(self, n_channels, n_classes, bilinear=False, softplus_beta=1.0, min_value=0.0, use_relu=False):
        super(PositiveUNet, self).__init__(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        if use_relu:
            self.post_activation = nn.ReLU()
        else:
            self.post_activation = nn.Softplus(beta=float(softplus_beta))
        self.min_value = float(min_value)

    def forward(self, x):
        logits = super(PositiveUNet, self).forward(x)

        # Apply activation only to the first channel
        first_chan = self.post_activation(logits[:, :1, ...])

        if logits.size(1) > 1:
            other = logits[:, 1:, ...]
            out = torch.cat([first_chan, other], dim=1)
        else:
            out = first_chan

        # Shift the activated first channel by min_value so its floor is `min_value`.
        # This avoids hard-clamping which can change gradients; adding an offset
        # preserves activation smoothness (Softplus) and keeps ReLU consistent.
        if self.min_value != 0.0:
            first = out[:, :1, ...] + self.min_value
            rest = out[:, 1:, ...] if out.size(1) > 1 else None
            out = first if rest is None else torch.cat([first, rest], dim=1)

        return out
