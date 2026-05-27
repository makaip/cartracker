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
    def __init__(self):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # add IBN to to resnet
        for layer in [base.layer1, base.layer2]:      # target first two layers
            for block in layer:
                out_channels = block.bn1.num_features   # get num of channels in layer
                block.bn1 = IBN(out_channels)           # replace first BN layer with IBN

        # add SE to resnet
        for layer in [base.layer1, base.layer2, base.layer3, base.layer4]:  # target all layers
            for block in layer:
                out_channels = block.bn3.num_features   # get num of channels in layer
                se = SEBlock(out_channels)              # make SE block
                block.se = se                           # and set it in the layer

                def block_forward(x):                   # wrap each residual block's forward function to add SE before residual
                    id = x                              # identity connection   (residual: y = F(x) + x)

                    # copy the normal architecture
                    out = block.conv1(x); out = block.bn1(out); out = block.relu(out)
                    out = block.conv2(x); out = block.bn2(out); out = block.relu(out)
                    out = block.conv3(x); out = block.bn3(out)

                    # apply SE block
                    out = se(out)

                    if block.downsample is not None:
                        id = block.downsample(x)
                    
                    out += id
                    out = block.relu(out)

                    return out

                block.forward = block_forward

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, # output: (B, 2048, 1, 1)
        )

        self.encoder = nn.Sequential(*list(base.children())[:-1])   # remove final classification layer
        self.fc1 = nn.Linear(2048, 512)                             # resnet50 outputs 2048-dim features
        self.fc2 = nn.Linear(512, 128)                              # replace it with a different one
    
    def forward(self, x):
        x = self.encoder(x).flatten(1).clone()  # use flatten(1) over squeeze() squeeze() with batch_size=1 collapses batch dim too
        # .clone() breaks the inplace version chain
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 norm. cosine similarity

if __name__ == "__main__":
    print(torch.cuda.is_available())

    # https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    model = EmbeddingNet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = VeRi(data_dir='datasets/VeRi/image_train', file='datasets/VeRi/train_label.xml', transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=256, 
                                             sampler=sampler, 
                                             num_workers=8, 
                                             pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss = 0.0

        for anchors, positives, negatives in dataloader:
            anchors = anchors.cuda(local_rank, non_blocking=True)
            positives = positives.cuda(local_rank, non_blocking=True)
            negatives = negatives.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            batch_loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss/len(dataloader):.4f}")
    
    dist.destroy_process_group()

    model.eval()

    with torch.no_grad():
        for i in range(5):
            anchor, positive, negative = dataset[i]

            anchor_emb = model(anchor.unsqueeze(0).to("cuda"))
            positive_emb = model(positive.unsqueeze(0).to("cuda"))
            negative_emb = model(negative.unsqueeze(0).to("cuda"))

            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)

            print(f"Sample {i}: Pos Dist={pos_dist.item():.4f}, Neg Dist={neg_dist.item():.4f}")
    
    if local_rank == 0:
        torch.save(model.module.state_dict(), "/mnt/beegfs/home/jpindell2022/ouri_project/mltests/caridentify/results/veri_embedding_model.pth")