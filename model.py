import torch
import torch.nn as nn
from torchvision import models

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.base_layers = list(base_model.children())

        # Encoder
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # 64
        self.layer2 = self.base_layers[5]                   # 128
        self.layer3 = self.base_layers[6]                   # 256
        self.layer4 = self.base_layers[7]                   # 512

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ConvRelu(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvRelu(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvRelu(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvRelu(64 + 64, 64)

        # Final output layer
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder path
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1(d1)

        x = self.final_upsample(d1)
        out = self.final(x)

        return out
