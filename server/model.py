import os

import torch
import torch.nn as nn
import torchvision.models as models

# DUPLICATED FROM train/utils.py

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                channels // reduction
            ),
            nn.ReLU(),
            nn.Linear(
                channels // reduction,
                channels
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x)
        w = w.view(b, c)
        w = self.fc(w)
        w = w.view(b, c, 1, 1)

        return x * w


class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(half)

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)
        return torch.cat(
            [self.IN(first),
             self.BN(second)],
            dim=1)

# DUPLICATED FROM train/model.py

class SEIBNBlock(nn.Module):
    def __init__(self, block, se):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.conv3 = block.conv3
        self.bn3 = block.bn3
        self.downsample = block.downsample
        self.se = se

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class EmbeddingNet(nn.Module):
    def __init__(self, num_classes=576):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for layer in [base.layer1, base.layer2]:
            for block in layer:
                out_channels = block.bn1.num_features
                block.bn1 = IBN(out_channels)

        for layer in [base.layer1, base.layer2, base.layer3, base.layer4]:
            for i, block in enumerate(layer):
                out_channels = block.bn3.num_features
                se = SEBlock(out_channels)
                layer[i] = SEIBNBlock(block, se)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

        self.embed_bn = nn.BatchNorm1d(2048)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        embed = self.embed_bn(x)
        logits = self.classifier(embed)

        return embed, logits
