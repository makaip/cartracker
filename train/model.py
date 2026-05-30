"""
attempt at implementing: https://arxiv.org/pdf/2110.07933v2
original code at: https://github.com/adhirajghosh/RPTM_reid


triplet mining: https://omoindrot.github.io/triplet-loss#triplet-mining

240x240 provided highest rank 1 perf.

- Instance-Batch Normalization (IBN)
    - in the feature embedding & extraction stage
    - TODO: find a ResNet-IBN implementation and add SE on top

relational matrix:
- quantify relationship between image pairs with Grid-Based Motion Statistics (GMS)
- for each image, extract 10000 ORB features (w/ orientation) and find nearest neighbors by brute force (hamming dist)

RPTM scheme (big idea of the paper):
- when selecting anchor positive pairs, take the image whose number of GMS matches with the anchor is closest to threshold tau
- use batch hard triplet mining for negative selection

Loss: E = \lambda_ent * E_ent + \lambda_tri * E_tri; E_ent = entropy loss; E_tri = triplet loss

hyperparams:
> Stochastic Gradient Descent(SGD) is used as the optimiser for the model
> The initial learning rate is initialised at 0.005 and is set to decay by a factor of 0.1 every 20 epochs.
> The model is trained for 80 epochs with a batch size of 24

"""
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import VeRi
from utils import SEBlock, IBN

# training script for vehicle identification embedding model using VeRi dataset and triplet loss

# resnet input: 3x224x224
# output: 512-dim embedding
# final layer needed to get 128-dim embedding for triplet loss

class EmbeddingNet(nn.Module):
    def __init__(self, num_classes=576):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # add IBN to to resnet
        for layer in [base.layer1, base.layer2]:        # target first two layers
            for block in layer:
                out_channels = block.bn1.num_features   # get num of channels in layer
                block.bn1 = IBN(out_channels)           # replace first BN layer with IBN

        # add SE to resnet
        for layer in [base.layer1, base.layer2, base.layer3, base.layer4]:  # target all layers
            for block in layer:
                out_channels = block.bn3.num_features   # get num of channels in layer
                se = SEBlock(out_channels)              # make SE block
                block.se = se                           # and set it in the layer

                def create_forward(b, s):
                    def block_forward(x):                   # wrap each residual block's forward function to add SE before residual
                        id = x                              # identity connection   (residual: y = F(x) + x)

                        # copy the normal architecture
                        out = b.conv1(x); out = b.bn1(out); out = b.relu(out)
                        out = b.conv2(out); out = b.bn2(out); out = b.relu(out)
                        out = b.conv3(out); out = b.bn3(out)

                        # apply SE block
                        out = s(out)

                        if b.downsample is not None:
                            id = b.downsample(x)
                        
                        out += id
                        out = b.relu(out)

                        return out
                    return block_forward

                block.forward = create_forward(block, se)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, # output: (B, 2048, 1, 1)
        )

        # note that for inference, we generate embeddings, but for training we do a classifier, so we skip allis
        # self.encoder = nn.Sequential(*list(base.children())[:-1])   # remove final classification layer
        # self.fc1 = nn.Linear(2048, 512)                             # resnet50 outputs 2048-dim features
        # self.fc2 = nn.Linear(512, 128)                              # replace it with a different one

        self.embed_bn = nn.BatchNorm1d(2048)            # batch norm for embedding
        self.classifier = nn.Linear(2048, num_classes)  # classifier head for CE loss
    
    def forward(self, x):
        x = self.backbone(x)                # output: (B, 2048, 1, 1)
        x = x.flatten(1)                    # output: (B, 2048)
        embed = self.embed_bn(x)            # output: (B, 2048) -> triplet loss
        logits = self.classifier(embed)     # output: (B, num_classes) -> CE loss

        return embed, logits

        # x = self.encoder(x).flatten(1).clone()  # use flatten(1) over squeeze() squeeze() with batch_size=1 collapses batch dim too
        # # .clone() breaks the inplace version chain
        # x = torch.nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        # return nn.functional.normalize(x, p=2, dim=1)  # L2 norm. cosine similarity
