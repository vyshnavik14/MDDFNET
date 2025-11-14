import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityHead(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, x, upsample_size=None):
        dmap = self.net(x)
        if upsample_size is not None:
            dmap = F.interpolate(dmap, size=upsample_size, mode='bilinear', align_corners=False)
        return dmap
