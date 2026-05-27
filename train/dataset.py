import re
import os
import random

import xml.etree.ElementTree as ET
from PIL import Image

import torch
import torchvision.transforms as transforms


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
