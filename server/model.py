import torch
import torch.nn as nn
import torchvision.models as models

# DUPLICATED FROM train/utils.py + train/model.py (inference only, so no pretrained download)

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


class EmbeddingNet(nn.Module):
    """ResNet50-IBN-a + BNNeck. In eval mode forward returns (embedding, logits)."""

    def __init__(self, num_classes=576):
        super().__init__()

        base = models.resnet50(weights=None)

        # IBN-a: first BN of every block in layers 1-3
        for layer in [base.layer1, base.layer2, base.layer3]:
            for block in layer:
                block.bn1 = IBN(block.bn1.num_features)

        # last stride 1
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

        self.embed_bn = nn.BatchNorm1d(2048)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        embed = self.embed_bn(feat)
        logits = self.classifier(embed)

        return embed, logits
