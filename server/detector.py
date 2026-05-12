import os
import queue
import sqlite3
from pathlib import Path
import cv2
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent
import yaml
import time
import numpy as np


SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
YOLO_CONFIG_DIR = PROJECT_DIR / '.cache' / 'ultralytics'
TORCH_HOME = PROJECT_DIR / '.cache' / 'torch'
XDG_CACHE_HOME = PROJECT_DIR / '.cache'

os.environ.setdefault('YOLO_CONFIG_DIR', str(YOLO_CONFIG_DIR))
os.environ.setdefault('TORCH_HOME', str(TORCH_HOME))
os.environ.setdefault('XDG_CACHE_HOME', str(XDG_CACHE_HOME))

from ultralytics import YOLO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

with open(SERVER_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']

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

    base_dir = Path(__file__).resolve().parent
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
        for ext in ("jpg", "jpeg", "png"):
            for img_path in vdir.glob(f"*.{ext}"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_tensor = transform(np.array(img)).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = classifier(img_tensor)
                embs.append(emb)
        
        if embs:
            mean_emb = torch.mean(torch.cat(embs, dim=0), dim=0)
            targets[uuid] = mean_emb
    
    return targets


def build_target_matrix(targets: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not targets:
        return None

    return torch.stack(list(targets.values()), dim=0)

def classify_car(frame_array: np.ndarray,
                       device: torch.device, 
                       classifier: EmbeddingNet
                       ) -> np.ndarray:
    input_tensor = transform(frame_array).unsqueeze(0)  # add batch dimension (# of tensors)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        embedding = classifier(input_tensor)
    
    return embedding

def match_embedding(embedding: torch.Tensor, 
                          target_matrix: torch.Tensor
                          ) -> torch.Tensor:
    if target_matrix is None:
        return None
    
    # cosine similarity: sim(a, b) = (a . b) / (||a|| * ||b||)
    # since embeddings are L2 normalized, ||a|| = ||b|| = 1, so we can just do dot product
    return torch.mm(embedding, target_matrix.t()).squeeze(0)  # shape: (num_targets,)

def gpu_worker(
        gpu_id: int,                # gpu index for worker
        frame_queue: mp.Queue,      # receive frames from main thread
        result_queue: queue.Queue,  # send results back to main thread
        stop_event: MpEvent,        # stop worker thread
        update_event: MpEvent       # new vehicle added (update embeddings)
    ) -> None:
    
    # setup worker - init
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(SERVER_DIR / 'yolov8s.pt'))
    detector.to(device)  # force GPU

    classifier = EmbeddingNet()
    classifier.load_state_dict(torch.load(SERVER_DIR / 'veri_emb_rn50.pt', map_location=device))
    classifier.to(device)
    classifier.eval()

    targets = precalc_targets(device, classifier)
    target_uuids = list(targets.keys())
    target_matrix = build_target_matrix(targets)

    # event loop - update
    while not stop_event.is_set():
        if update_event.is_set():
            print(f"[GPU {gpu_id}] Updating target embeddings...")
            targets = precalc_targets(device, classifier)
            target_uuids = list(targets.keys())
            target_matrix = build_target_matrix(targets)
            update_event.clear()
        
        try:
            if (item := frame_queue.get(timeout=1.0)) is None:
                continue

            camera_uuid, frame_array = item
        
        except queue.Empty:
            continue
        if target_matrix is None:
            target_uuids = []
        
        # YOLOv8 classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        yolo_results = detector.predict(frame_array, device=device, classes=[2, 3, 5, 7], verbose=False)

        frame_detections = []

        for i, result in enumerate(yolo_results[0].boxes):
            x1, y1, x2, y2 = result.xyxy[0].int().tolist()
            
            # handle out of bounds bounding boxes securely
            if y2 <= y1 or x2 <= x1:
                continue
                
            car_image = frame_array[y1:y2, x1:x2]
            embedding = classify_car(car_image, device, classifier)

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

            similarity = match_embedding(embedding, target_matrix)
            if similarity is not None and len(target_uuids) > 0:
                matched_idx = similarity.argmax().item()
                matched_uuid = target_uuids[matched_idx]
                sim_matrix.append((matched_uuid, similarity.max().item()))

            frame_detections.append({
                "vehicle_id": i,
                "box": [x1, y1, x2, y2],
                "matches": sim_matrix
            })

        try:
            result_queue.put_nowait({
                "camera_uuid": camera_uuid,
                "timestamp": time.time(),
                "detections": frame_detections
            })
        except queue.Full:
            pass
