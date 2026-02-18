import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = conv_block(channels, channels)

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + x)

class Net(nn.Module):
    """
    ResNet9-style architecture adapted for single-channel input.
    Uses AdaptiveAvgPool2d so the model accepts variable img_size.
    """
    def __init__(self, num_classes: int, img_size: int = None, in_channels: int = 1):
        super().__init__()
        # feature extractor
        self.conv1 = conv_block(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv2 = conv_block(64, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.conv4 = conv_block(256, 512)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x + self.res1(x)  

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x + self.res2(x)  

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x + self.res3(x)  

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = x + self.res4(x)  

        x = self.classifier(x)
        return x