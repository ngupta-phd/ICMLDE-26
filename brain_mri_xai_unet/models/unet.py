import torch
import torch.nn as nn

# Fully convolutional; inference resizes MRI slices to this spatial size.
IMAGE_SIZE = 224


class LightweightUNet(nn.Module):
    def __init__(self):
        super(LightweightUNet, self).__init__()
        # Encoder Block (inplace=False to avoid PyTorch autograd hook errors)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False)
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck Layer (Grad-CAM Target)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False)
        )
        
        # Decoder Block
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        b = self.bottleneck(self.pool(x1))
        u = self.up(b)
        out = self.dec1(torch.cat([u, x1], dim=1))
        return out