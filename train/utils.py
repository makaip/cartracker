import torch
import torch.nn as nn
import numpy as np

"""
Squeeze-Excitation (SE) layer: https://arxiv.org/pdf/1709.01507
- integrated into residual blocks of conv net
- performs global average pooling on feature maps, then squeeze, then excitation steps
- out -> set of weights for channel-wise multiplication (importance map)
"""

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # squeeze: (B, C, H, W) -> (B, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(
                channels,               # 1 x 1 x C 
                channels // reduction   # 1 x 1 x C/r
            ),
            nn.ReLU(),                  # 1 x 1 x C/r
            nn.Linear(
                channels // reduction,  # 1 x 1 x C/r
                channels                # 1 x 1 x C
            ),
            nn.Sigmoid()                # 1 x 1 x C
        )

    def forward(self, x):
        # b: batch; c: channels; _: height; _: width
        b, c, _, _ = x.shape            # unpack batch & channel dims
        w = self.pool(x)                # pool to B x C x 1 x 1
        w = w.view(b, c)                # reshape to B x C
        w = self.fc(w)                  # pass through FC layers to get B x C weights
        w = w.view(b, c, 1, 1)          # reshape back to B x C x 1 x 1
        
        return x * w                    # channel-wise mult

"""
https://arxiv.org/abs/1807.09441

split channels, replace the first BN layer in certain residual blocks with a half IN, half BN operation, then concatenate
"""

class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2    # split
        self.IN = nn.InstanceNorm2d(half, affine=True)  # instancenorm
        self.BN = nn.BatchNorm2d(half)  # batchnorm

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)
        return torch.cat( # concatenate
            [self.IN(first),
             self.BN(second)],
            dim=1)

"""
Relation Preserving Triplet Mining (RPTM) triplet mining, as described in https://arxiv.org/pdf/2110.07933v2
1. find all pos & neg samples for each anchor
2. for pos samples, find the sample whose number of GMS matches with the anchor is closest to tau (mean of non-zero matches)
3. for neg samples, find the sample with the smallest embedding distance to the anchor (batch hard triplet mining)
4. add & return the triplets
"""

def rtpm(
        embeds: torch.Tensor,
        labels: torch.Tensor,
        indicies: torch.Tensor,
        relational_matrix: torch.Tensor
) -> list:
    
    triplets = []

    embeds = embeds.detach().cpu()
    labels = labels.cpu().numpy()
    indicies = indicies.cpu().numpy()

    for i in range(len(indicies)):
        # anchor
        a_idx = indicies[i]
        a_label = labels[i]

        same = [j for j in range(len(labels))           # find all indices of images with the same label as the anchor
                if labels[j] == a_label and j != i]     # exclude the anchor itself
        
        diff = [j for j in range(len(labels))           # find all indices of images with a different label than the anchor
                if labels[j] != a_label]                # exclude the anchor itself
        
        if not same or not diff:                        # if no positive or negative samples, skip
            continue

        # RPTM pos - closest match count to tau
        sd_idx = indicies[same]
        counts = relational_matrix[a_idx, sd_idx]
        nonzero = counts[counts > 0]

        tau = nonzero.mean() if len(nonzero) > 0 else 0
        best_local = np.argmin(np.abs(counts - tau))
        pos_idx = same[best_local]

        diff_embeds = embeds[diff]
        dists = torch.norm(diff_embeds - embeds[i], dim=1)  # compute distances to anchor
        neg_idx = diff[torch.argmin(dists).item()]          # find index of

        triplets.append((i, pos_idx, neg_idx))
    
    return triplets
