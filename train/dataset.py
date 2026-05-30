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
        # move the anchor/pos/neg selection logic outside of dataset
        img_name, vid = self.imgs[idx]
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.vid2idx[vid]
        return img, label, idx
