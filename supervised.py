import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

#=======================================
#========= UNet Architecture ===========
#=======================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final(dec1)

#=======================================
#======= Inception Architecture ========
#=======================================
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

class Inception(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Inception, self).__init__()
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
        height, width = x.shape[2], x.shape[3]
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        return x

#=======================================
#======= Swin Transformer ==============
#=======================================
class Segformer(nn.Module):
    def __init__(self, model_name='nvidia/segformer-b0-finetuned-ade-512-512', num_classes=2):
        super(Segformer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.normalizer = T.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)

    def forward(self, x):
        x = self.normalizer(x)
        logits = self.model(pixel_values=x).logits  # Shape: [B, C, H', W']
        logits = F.interpolate(logits, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return logits