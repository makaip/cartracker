import re
import os
import random

import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms

from utils import rptm_positives


class VeRi(torch.utils.data.Dataset):
    """
    without rel_mat: returns (img, label, idx)
    with rel_mat (path to rel_mat.npy): returns (anchor, RPTM positive, label, idx)
    """

    def __init__(self, data_dir, file, transform=None, rel_mat=None):
        self.root = data_dir
        self.transform = transform or transforms.ToTensor()
        self.imgs, self.vid2imgs = self._parse_labels(file)     # vid = Vehicle ID

        self.unique_vids = sorted(list(self.vid2imgs.keys()))
        self.vid2idx = {vid: idx for idx, vid in enumerate(self.unique_vids)}

        # dataset indices of every image of each vehicle ID
        self.vid2indices = {}
        for idx, (_, vid) in enumerate(self.imgs):
            self.vid2indices.setdefault(vid, []).append(idx)

        self.pos = None
        if rel_mat is not None:
            matrix = np.load(rel_mat, mmap_mode='r')    # m x m int32 is several GB, so don't load it whole
            if matrix.shape[0] != len(self.imgs):
                raise ValueError(f"{rel_mat} is {matrix.shape} but the dataset has {len(self.imgs)} images")

            self.pos = rptm_positives(matrix, list(self.vid2indices.values()))

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

    def _load(self, idx):
        img_name, _ = self.imgs[idx]
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, idx):
        _, vid = self.imgs[idx]
        label = self.vid2idx[vid]

        if self.pos is None:
            return self._load(idx), label, idx

        pos_idx = int(self.pos[idx])
        if pos_idx < 0:
            # anchor has no GMS matches at all: fall back to a random image of the same vehicle
            others = [i for i in self.vid2indices[vid] if i != idx]
            pos_idx = random.choice(others) if others else idx

        return self._load(idx), self._load(pos_idx), label, idx


class VeRiEval(torch.utils.data.Dataset):
    """query / gallery images listed in name_query.txt / name_test.txt, returns (img, vehicle ID, camera ID)"""

    def __init__(self, data_dir, file, transform=None):
        self.root = data_dir
        self.transform = transform or transforms.ToTensor()

        with open(file, 'r') as f:
            self.img_names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # parse filename like "0002_c002_00030600_0.jpg"
        parts = img_name.split('_')
        vid = int(parts[0])             # "0002" -> 2
        cid = int(parts[1][1:])         # "c002" -> 2 (skip 'c')

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        img = self.transform(img)

        return img, vid, cid
