"""
create relational matrix (runs once before training) for VeRi dataset using GMS to quantify relationship between image pairs
- for each image, extract 10000 ORB features (w/ orientation)
- find nearest neighbors by brute force (hamming dist)

- ORB: Oriented FAST and Rotated BRIEF                  (fast feature detector and descriptor)
- FAST: Features from Accelerated Segment Test          (corner detection method)
- BRIEF: Binary Robust Independent Elementary Features  (feature descriptor that uses binary strings)
- GMS: Grid-Based Motion Statistics                     (feature matching via. grid-based motion statistics)

https://www.geeksforgeeks.org/python/feature-matching-using-orb-algorithm-in-python-opencv/

> The GMS feature matching parameters are: 10,000
> ORB features whose orientation parameter is set to true and
> nearest neighbours are identified with the brute-force hamming distance
> (page 6)

"""

from collections import defaultdict
import os

import cv2
import numpy as np
from tqdm import tqdm

from dataset import VeRi


dataset = VeRi(data_dir='./datasets/VeRi/image_train', file='./datasets/VeRi/train_label.xml')

m = len(dataset)    # get number of images in dataset
matrix = np.zeros(  # correlational matrix
    (m, m),         # m x m so each image pair has a value
    dtype=np.int32  # number of GMS matches is an integer
)

# might need to install opencv-contrib-python for GMS
orb = cv2.ORB_create(nfeatures=10000)                   # create ORB feature extractor
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)               # make brute force matcher via. hamming distance

# group dataset indices by vehicle ID
vid2indices = defaultdict(list)                         # map vehicle ID to list of indices in dataset
for idx, (img_name, vid) in enumerate(dataset.imgs):    # enumerate over dataset images and their vehicle IDs
    vid2indices[vid].append(idx)                        # append index to list of indices for that vehicle ID

# only match within same ID pairs
for vid, indices in tqdm(vid2indices.items()):
    images = [] # the list of images for this vehicle ID to be used for GMS matching

    for idx in indices:
        img_name, _ = dataset.imgs[idx]
        img = cv2.imread(os.path.join(dataset.root, img_name))
        img = cv2.resize(img, (224, 224))   # resize to 224x224 so that it will fit into resnet
        images.append(img)

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            # kp: keypoints, des: descriptors
            kp1, des1 = orb.detectAndCompute(images[i], None)
            kp2, des2 = orb.detectAndCompute(images[j], None)

            if des1 is None or des2 is None:
                continue # pass if junk

            raw_matches = matcher.match(des1, des2) # use BFMatcher to find nearest neighbors

            # docs for xfeatures2d.matchGMS method are only availble in java for some reason
            # https://docs.opencv.org/4.x/javadoc/org/opencv/xfeatures2d/Xfeatures2d.html

            gms_matches = cv2.xfeatures2d.matchGMS( # use GMS to filter out bad matches
                (224, 224), (224, 224),             # pass sizes for both images
                kp1, kp2, raw_matches,              # pass keypoints and raw matches
                withRotation=True,                  # specified in paper
                withScale=False
            )

            count = len(gms_matches)                # get number of GMS-filtered matches
            matrix[indices[i], indices[j]] = count  # assign to upper triangle of matrix
            matrix[indices[j], indices[i]] = count  # assign to lower tri of mat bc symmetric

np.save('./train/rel_mat.npy', matrix)
print(f"saved relational matrix of shape {matrix.shape}")