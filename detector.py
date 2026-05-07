import numpy as np
import cv2
from ultralytics import YOLO
import logging

import torch
import torch.nn as nn
import torchvision.models as models

print(torch.cuda.is_available())


model = YOLO('yolov8s.pt')
model.to('cuda')  # force GPU

def detect_cars(frame_array: np.ndarray) -> np.ndarray:
    if model is None:
        return frame_array

    # YOLOv8 classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    results = model(frame_array, classes=[2, 3, 5, 7], verbose=False)
    annotated_frame = results[0].plot()
    return annotated_frame


