import os
import random
from PIL import Image
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

import xml.etree.ElementTree as ET

# training script for vehicle identification embedding model using VeRi dataset and triplet loss

# resnet input: 3x224x224
# output: 512-dim embedding
# final layer needed to get 128-dim embedding for triplet loss

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # remove final classification layer
        self.fc1 = nn.Linear(2048, 512)                             # resnet50 outputs 2048-dim features
        self.fc2 = nn.Linear(512, 128)                              # replace it with a different one
    
    def forward(self, x):
        x = self.encoder(x).squeeze()
        x = self.fc1(x)
        x = self.fc2(x)

        return nn.functional.normalize(x, p=2, dim=1)  # L2 norm. cosine similarity


class VeRi(torch.utils.data.Dataset):
    def __init__(self, data_dir, file, transform=None):
        self.root = data_dir
        self.transform = transform or transforms.ToTensor()
        self.imgs, self.vid2imgs = self._parse_labels(file)

    def _parse_labels(self, label_file):
        with open(label_file, 'r', encoding='gb2312', errors='ignore') as f:
            content = f.read()
        
        # python built-in parser doesn't natively support gb2312 multi-byte encoding 
        content = re.sub(r"<\?xml.*?\?>", "", content)
        root = ET.fromstring(content)
        
        imgs = []
        vid2imgs = {}

        for item in root.findall('.//Item'):
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

    model = EmbeddingNet()
    model.to("cuda")

    loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = VeRi(data_dir='datasets/VeRi/image_train', file='datasets/VeRi/train_label.xml', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        loss = 0.0

        for anchors, positives, negatives in dataloader:
            anchors, positives, negatives = anchors.to("cuda"), positives.to("cuda"), negatives.to("cuda")

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            batch_loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss/len(dataloader):.4f}")
    
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
    