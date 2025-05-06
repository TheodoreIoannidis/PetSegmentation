import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionSegment(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(InceptionSegment, self).__init__()
        self.weights_init()
        self.inception1 = InceptionBlock(in_channels, 64)
        self.inception2 = InceptionBlock(256, 128)
        self.inception3 = InceptionBlock(512, 256)

        self.conv1x1 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x
