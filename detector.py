from matplotlib import transforms
import numpy as np
import cv2
from ultralytics import YOLO
import logging

from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

print("Cuda is available:", torch.cuda.is_available())

# from train/train.py
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


detector = YOLO('yolov8s.pt')
detector.to('cuda')  # force GPU

classifier = EmbeddingNet()
classifier.load_state_dict(torch.load('veri_emb_rn50.pt', map_location='cuda'))
classifier.to('cuda')
classifier.eval()


def detect_cars(frame_array: np.ndarray): #idk what the return type is
    # YOLOv8 classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    results = detector(frame_array, classes=[2, 3, 5, 7], verbose=False)

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].int().tolist()
        car_image = frame_array[y1:y2, x1:x2]
        results[0].embedding = classify_car(car_image)
    
    return results

def classify_car(frame_array: np.ndarray) -> np.ndarray:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(frame_array).unsqueeze(0)  # add batch dimension (# of tensors)
    input_tensor = input_tensor.cuda()  # move to GPU

    with torch.no_grad():
        embedding = classifier(input_tensor)
    
    return embedding
