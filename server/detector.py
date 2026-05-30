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

from model import EmbeddingNet

def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(crop_bgr, (224, 224))
    rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    rgb_crop = rgb_crop.transpose(2, 0, 1)
    # To float [0, 1]
    return torch.from_numpy(rgb_crop).float().div(255.0)

def batch_classify_cars(crops: list[np.ndarray],
                        device: torch.device, 
                        classifier: EmbeddingNet
                        ) -> torch.Tensor:
    if not crops:
        return torch.empty((0, 2048), device=device)

    tensors = [_preprocess_crop(crop) for crop in crops]
    batch_tensor = torch.stack(tensors).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch_tensor = batch_tensor.sub(mean).div(std)

    with torch.no_grad():
        embeddings, _ = classifier(batch_tensor)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings

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
                
                img_tensor = _preprocess_crop(img).unsqueeze(0).to(device)
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                img_tensor = img_tensor.sub(mean).div(std)

                with torch.no_grad():
                    emb, _ = classifier(img_tensor)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embs.append(emb)
        
        if embs:
            mean_emb = torch.mean(torch.cat(embs, dim=0), dim=0)
            mean_emb = torch.nn.functional.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            targets[uuid] = mean_emb
    
    return targets


def build_target_matrix(targets: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not targets:
        return None

    return torch.stack(list(targets.values()), dim=0)

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
    classifier.load_state_dict(torch.load(SERVER_DIR / 'veri_rtpm_rn50.pt', map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()

    targets = precalc_targets(device, classifier)
    target_uuids = list(targets.keys())
    target_matrix = build_target_matrix(targets)

    # profiling setup
    profiler = None
    profile_path = None
    if config.get('profile', False):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        profile_path = SERVER_DIR / f'cProfile_gpu_{gpu_id}.txt'
        print(f"[GPU {gpu_id}] Live profiling enabled. Writing stats every 50 frames to {profile_path}")

    frame_count = 0
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
        valid_crops = []
        valid_boxes = []
        vehicle_ids = []

        for i, result in enumerate(yolo_results[0].boxes):
            x1, y1, x2, y2 = result.xyxy[0].int().tolist()
            
            # handle out of bounds bounding boxes securely
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_array.shape[1], x2), min(frame_array.shape[0], y2)

            if y2 <= y1 or x2 <= x1:
                continue
                
            car_image = frame_array[y1:y2, x1:x2]
            valid_crops.append(car_image)
            valid_boxes.append([x1, y1, x2, y2])
            vehicle_ids.append(i)

        if valid_crops:
            embeddings = batch_classify_cars(valid_crops, device, classifier)

            for idx, embedding in enumerate(embeddings):
                sim_matrix = []
                similarity = match_embedding(embedding.unsqueeze(0), target_matrix)
                
                if similarity is not None and len(target_uuids) > 0:
                    matched_idx = similarity.argmax().item()
                    matched_uuid = target_uuids[matched_idx]
                    sim_matrix.append((matched_uuid, similarity.max().item()))

                frame_detections.append({
                    "vehicle_id": vehicle_ids[idx],
                    "box": valid_boxes[idx],
                    "matches": sim_matrix
                })

        try:
            result_queue.put_nowait({
                "camera_uuid": camera_uuid,
                "detections": frame_detections
            })
        except queue.Full:
            pass
        except:
            pass

        if profiler:
            frame_count += 1
            if frame_count % 50 == 0:
                import pstats
                profiler.disable()
                try:
                    with open(profile_path, 'w') as stream:
                        stats = pstats.Stats(profiler, stream=stream).sort_stats('tottime')
                        stats.print_stats(50)
                except Exception as e:
                    pass
                profiler.enable()
