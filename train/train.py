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

import random
from PIL import Image
import re
import copy
import os
import numpy as np
import cv2
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import xml.etree.ElementTree as ET

# training script for vehicle identification embedding model using VeRi dataset and triplet loss

# resnet input: 3x224x224
# output: 512-dim embedding
# final layer needed to get 128-dim embedding for triplet loss

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])   # remove final classification layer
        self.fc1 = nn.Linear(2048, 512)                             # resnet50 outputs 2048-dim features
        self.fc2 = nn.Linear(512, 128)                              # replace it with a different one
    
    def forward(self, x):
        x = self.encoder(x).flatten(1).clone()  # use flatten(1) over squeeze() squeeze() with batch_size=1 collapses batch dim too
        # .clone() breaks the inplace version chain
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 norm. cosine similarity

class VeRi(torch.utils.data.Dataset):
    def __init__(self, data_dir, file, transform=None):
        self.root = data_dir
        self.transform = transform or transforms.ToTensor()
        self.imgs, self.vid2imgs = self._parse_labels(file)     # vid = Vehicle ID

        self.unique_vids = sorted(list(self.vid2imgs.keys()))
        self.vid2idx = {vid: idx for idx, vid in enumerate(self.unique_vids)}

    def _parse_labels(self, label_file):
        with open(label_file, 'r', encoding='gb2312', errors='ignore') as f:
            content = f.read()
        
        # python built-in parser doesn't natively support gb2312 multi-byte encoding 
        content = re.sub(r"<\?xml.*?\?>", "", content)
        root = ET.fromstring(content)
        
        imgs = []
        vid2imgs = {}

        for item in root.findall('.//Item'):                # pair each VID with its images
            img_name = item.get('imageName')
            vid = item.get('vehicleID')
            imgs.append((img_name, vid))
            vid2imgs.setdefault(vid, []).append(img_name)

        return imgs, vid2imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        anchor_name, anchor_vid = self.imgs[idx]
        
        # positive sample
        pos_candidates = [n for n in self.vid2imgs[anchor_vid] if n != anchor_name]
        pos_name = random.choice(pos_candidates) if pos_candidates else anchor_name
        
        # negative sample
        neg_vid = random.choice([vid for vid in self.vid2imgs.keys() if vid != anchor_vid])
        neg_name = random.choice(self.vid2imgs[neg_vid])

        anchor = Image.open(os.path.join(self.root, anchor_name)).convert('RGB')
        positive = Image.open(os.path.join(self.root, pos_name)).convert('RGB')
        negative = Image.open(os.path.join(self.root, neg_name)).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


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