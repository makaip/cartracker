"""
attempt at implementing: https://arxiv.org/pdf/2110.07933v2
original code at: https://github.com/adhirajghosh/RPTM_reid


triplet mining: https://omoindrot.github.io/triplet-loss#triplet-mining

240x240 provided highest rank 1 perf.

- Instance-Batch Normalization (IBN)
    - in the feature embedding & extraction stage
    - backbone is ResNet50-IBN-a with the pretrained weights from https://github.com/XingangPan/IBN-Net

relational matrix:
- quantify relationship between image pairs with Grid-Based Motion Statistics (GMS)
- for each image, extract 10000 ORB features (w/ orientation) and find nearest neighbors by brute force (hamming dist)

RPTM scheme (big idea of the paper):
- when selecting anchor positive pairs, take the image whose number of GMS matches with the anchor is closest to threshold tau
- use batch hard triplet mining for negative selection

Loss: E = \\lambda_ent * E_ent + \\lambda_tri * E_tri; E_ent = entropy loss; E_tri = triplet loss

hyperparams:
> Stochastic Gradient Descent(SGD) is used as the optimiser for the model
> The initial learning rate is initialised at 0.005 and is set to decay by a factor of 0.1 every 20 epochs.
> The model is trained for 80 epochs with a batch size of 24

"""
import torch
import torch.nn as nn
import torchvision.models as models

from utils import IBN

# pretrained ResNet50-IBN-a weights (ImageNet), cached under $TORCH_HOME/hub/checkpoints
IBN_A_URL = 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth'


def resnet50_ibn_a(pretrained=True):
    base = models.resnet50(weights=None)

    # IBN-a: the first BN of every block in layers 1-3 is half IN, half BN (layer4 stays plain BN)
    # the module/attribute names match IBN-Net so its checkpoint loads as is
    for layer in [base.layer1, base.layer2, base.layer3]:
        for block in layer:
            block.bn1 = IBN(block.bn1.num_features)

    if pretrained:
        state = torch.hub.load_state_dict_from_url(IBN_A_URL, map_location='cpu')
        base.load_state_dict(state)     # strict, so a mismatch fails loudly instead of training from scratch

    return base


class EmbeddingNet(nn.Module):
    def __init__(self, num_classes=576, pretrained=True):
        super().__init__()

        base = resnet50_ibn_a(pretrained)

        # last stride 1: keep a 2x larger final feature map (same weights, more spatial detail)
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, # output: (B, 2048, 1, 1)
        )

        # BNNeck: triplet loss sees the feature before BN, CE loss and inference see it after BN
        self.embed_bn = nn.BatchNorm1d(2048)
        self.embed_bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        x = self.backbone(x)                # output: (B, 2048, 1, 1)
        feat = x.flatten(1)                 # output: (B, 2048) -> triplet loss
        embed = self.embed_bn(feat)         # output: (B, 2048) -> inference embedding
        logits = self.classifier(embed)     # output: (B, num_classes) -> CE loss

        if self.training:
            return feat, embed, logits

        return embed, logits
