"""
Squeeze-Excitation (SE) layer: https://arxiv.org/pdf/1709.01507
- integrated into residual blocks of conv net
- performs global average pooling on feature maps, then squeeze, then excitation steps
- out -> set of weights for channel-wise multiplication (importance map)
"""

from torch import nn

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
        b, c, _, _ = x.shape()          # unpack batch & channel dims
        w = self.pool(x)                # pool to B x C x 1 x 1
        w = w.view(b, c)                # reshape to B x C
        w = self.fc(w)                  # pass through FC layers to get B x C weights
        w = w.view(b, c, 1, 1)          # reshape back to B x C x 1 x 1
        
        return x * w                    # channel-wise mult