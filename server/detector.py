import os
import queue
import sqlite3
from pathlib import Path
import cv2
import multiprocessing as mp
import asyncio

import numpy as np
from ultralytics import YOLO

import torch
import torch.nn as nn
import torch.nn.functional as F
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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def precalc_targets(device: torch.device, 
                    classifier: EmbeddingNet
                    ) -> dict[str, torch.Tensor]:

    base_dir = Path(__file__).resolve()
    db_path = base_dir / "vehicles.db"
    uploads_dir = base_dir / "uploads"
    targets = {}
    
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT uuid FROM vehicles").fetchall()
    conn.close()

    for (uuid,) in rows:
        # (vehicle directory)
        vdir = uploads_dir / uuid

        embs = []
        for ext in (".jpg", ".jpeg", ".png"):
            for img_path in vdir.glob(f"*.{ext}"):
                img = cv2.imread(str(img_path))
                img_tensor = transform(np.array(img)).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = classifier(img_tensor)
                embs.append(emb)
        
        mean_emb = torch.mean(torch.cat(embs, dim=0), dim=0)
        targets[uuid] = mean_emb
    
    return targets

async def classify_car(frame_array: np.ndarray,
                       device: torch.device, 
                       classifier: EmbeddingNet
                       ) -> np.ndarray:
    input_tensor = transform(frame_array).unsqueeze(0)  # add batch dimension (# of tensors)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        embedding = classifier(input_tensor)
    
    return embedding

async def match_embedding(embedding: torch.Tensor, 
                          target_matrix: torch.Tensor
                          ) -> list[str]:
    if target_matrix is None:
        return None
    
    # cosine similarity: sim(a, b) = (a . b) / (||a|| * ||b||)
    # since embeddings are L2 normalized, ||a|| = ||b|| = 1, so we can just do dot product
    return torch.mm(embedding, target_matrix.t()).squeeze(0)  # shape: (num_targets,)

async def gpu_worker(
        gpu_id: int,                    # gpu index for worker
        frame_queue: mp.Queue,       # receive frames from main thread
        result_queue: queue.Queue,      # send results back to main thread
        stop_event: asyncio.Event,    # stop worker thread
        update_event: asyncio.Event   # new vehicle added (update embeddings)
    ) -> None:
    
    # setup worker - init
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve()

    detector = YOLO('yolov8s.pt')
    detector.to(device)  # force GPU

    classifier = EmbeddingNet()
    classifier.load_state_dict(torch.load('veri_emb_rn50.pt', map_location=device))
    classifier.to(device)
    classifier.eval()

    targets = precalc_targets(device, classifier)
    target_uuids = list(targets.keys())
    target_matrix = torch.cat(list(targets.values()), dim=0) if targets else None

    # event loop - update
    while not stop_event:
        if update_event.is_set():
            print(f"[GPU {gpu_id}] Updating target embeddings...")
            targets = precalc_targets(device, classifier)
            target_uuids = list(targets.keys())
            target_matrix = torch.cat(list(targets.values()), dim=0) if targets else None
            update_event.clear()
        
        try:
            if (item := frame_queue.get(timeout=1.0)) is None:
                continue

            camera_uuid, frame_array = item
        
        except queue.Empty:
            continue
        if target_matrix is None:
            continue
        
        # YOLOv8 classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        yolo_results = detector(frame_array, classes=[2, 3, 5, 7], verbose=False)
        bounding_boxes = [box for box in yolo_results[0].boxes if box.id is None]
        # ^ extract bounding boxes and assign an ID to each box

        for result in yolo_results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].int().tolist()
            car_image = frame_array[y1:y2, x1:x2]
            yolo_results[0].embedding = await classify_car(car_image)
        
        """
        simmilarity matrix format
        format:
            target_uuid: {
                vehicle_id: <similarity_score>,
                ...
            }
            ...
        """

        sim_matrix = []

        for vehicle in target_matrix:
            similarity = await match_embedding(yolo_results[0].embedding, vehicle, target_matrix)
            if similarity is not None and similarity.max() > 0.75:  # threshold for match
                matched_uuid = target_uuids[similarity.argmax()]
                sim_matrix.append((matched_uuid, similarity.max().item()))
        
        # put results in queue for main thread
        try:
            result_queue.put_nowait({
                "camera_uuid": camera_uuid,
                "bounding_boxes": bounding_boxes,
                "similarity_matrix": sim_matrix
            })
        except queue.Full:
            pass
