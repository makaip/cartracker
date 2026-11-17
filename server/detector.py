import os
import queue
import sqlite3
from pathlib import Path
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
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

INPUT_SIZE = 240    # must match IMG_SIZE in train/train.py
BATCH_SIZE = int(config.get('batch_size', 16))  # frames per YOLO call
CROP_BATCH_SIZE = 128                           # crops per embedding call
WEIGHTS = 'veri_rptm_r50ibn.pt'

def _embed(classifier: EmbeddingNet, batch: torch.Tensor) -> torch.Tensor:
    # same as train/eval.py: average the embedding of the image and its horizontal flip, then L2 normalize
    emb = classifier(batch)[0] + classifier(torch.flip(batch, dims=[3]))[0]
    return torch.nn.functional.normalize(emb, p=2, dim=1)

def _preprocess_crop(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    crop_rgb = crop_bgr[:, :, ::-1].copy()  # BGR to RGB
    # HWC -> CHW, convert to float [0, 1], push to GPU
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).to(device, dtype=torch.float32).div(255.0)
    return transforms.functional.resize(tensor, [INPUT_SIZE, INPUT_SIZE], antialias=True)

def batch_classify_cars(crops: list[np.ndarray],
                        device: torch.device, 
                        classifier: EmbeddingNet
                        ) -> torch.Tensor:
    if not crops:
        return torch.empty((0, 2048), device=device)

    tensors = [_preprocess_crop(crop, device) for crop in crops]
    batch_tensor = torch.stack(tensors)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch_tensor = batch_tensor.sub(mean).div(std)

    with torch.no_grad():
        embeddings = _embed(classifier, batch_tensor)
    
    return embeddings

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}

def precalc_targets(device: torch.device,
                    classifier: EmbeddingNet
                    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """embedding per vehicle, and the score a match must reach to be recorded
    as a sighting (only for vehicles with tracking on)"""

    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "vehicles.db"
    uploads_dir = base_dir / "uploads"
    targets = {}
    thresholds = {}

    conn = sqlite3.connect(str(db_path), timeout=30)
    rows = conn.execute("SELECT uuid, tracking, threshold FROM vehicles").fetchall()
    conn.close()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for uuid, tracking, threshold in rows:
        # (vehicle directory)
        vdir = uploads_dir / uuid
        if not vdir.is_dir():
            continue

        embs = []
        for img_path in sorted(vdir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_tensor = _preprocess_crop(img, device).unsqueeze(0)
            img_tensor = img_tensor.sub(mean).div(std)

            with torch.no_grad():
                emb = _embed(classifier, img_tensor)
            embs.append(emb)

        if embs:
            mean_emb = torch.mean(torch.cat(embs, dim=0), dim=0)
            mean_emb = torch.nn.functional.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            targets[uuid] = mean_emb
            if tracking:
                thresholds[uuid] = float(threshold)

    return targets, thresholds


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

def _load_frame(item: dict):
    """copy a queued frame out of shared memory and free the block; None if it is gone"""
    try:
        shm = shared_memory.SharedMemory(name=item['shm_name'])  # reconstruct array from shared memory pointer
    except Exception:
        return None

    try:
        shm_array = np.ndarray(item['shape'], dtype=item['dtype'], buffer=shm.buf)
        frame_array = shm_array.copy()  # copy immediately to release block
    except Exception:
        frame_array = None
    finally:
        # prevent mem leak
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    if frame_array is None:
        return None
    return item['camera_uuid'], item.get('frame_id', 0), frame_array


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

    # alternatively export this to TensorRT. will prob do later bc startup is slow
    # detector.export(format='engine', device=gpu_id)
    
    detector.to(device)  # force GPU

    classifier = EmbeddingNet()
    classifier.load_state_dict(torch.load(SERVER_DIR / WEIGHTS, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()
    classifier = torch.compile(classifier)

    targets, thresholds = precalc_targets(device, classifier)
    target_uuids = list(targets.keys())
    target_matrix = build_target_matrix(targets)
    # per target, the score that counts as a sighting; inf when tracking is off
    record_thresholds = [thresholds.get(u, float('inf')) for u in target_uuids]

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
    last_profile_dump = 0
    # event loop - update
    while not stop_event.is_set():
        if update_event.is_set():
            print(f"[GPU {gpu_id}] Updating target embeddings...")
            # clear first: a change that lands while reloading must trigger another reload
            update_event.clear()
            targets, thresholds = precalc_targets(device, classifier)
            target_uuids = list(targets.keys())
            target_matrix = build_target_matrix(targets)
            record_thresholds = [thresholds.get(u, float('inf')) for u in target_uuids]
        
        # take one frame (blocking), then whatever else is already waiting, up to BATCH_SIZE.
        # one YOLO call on a batch is several times faster than one call per frame.
        try:
            item = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception:
            continue

        batch = []  # (camera_uuid, frame_id, frame_array)
        while True:
            if item is not None:
                frame = _load_frame(item)
                if frame is not None:
                    batch.append(frame)
            if len(batch) >= BATCH_SIZE:
                break
            try:
                item = frame_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

        if not batch:
            continue

        if target_matrix is None:
            target_uuids = []

        # YOLOv8 classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        yolo_results = detector.predict(
            [frame_array for _, _, frame_array in batch],
            device=device, classes=[2, 3, 5, 7], verbose=False
        )

        batch_detections = [[] for _ in batch]
        # per frame: target index -> (score, crop index) of its best match worth recording
        batch_hits = [{} for _ in batch]
        valid_crops = []
        crop_owner = []  # (index into batch, vehicle_id, box) per crop

        for b, (_, _, frame_array) in enumerate(batch):
            for i, result in enumerate(yolo_results[b].boxes):
                x1, y1, x2, y2 = result.xyxy[0].int().tolist()

                # handle out of bounds bounding boxes securely
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_array.shape[1], x2), min(frame_array.shape[0], y2)

                if y2 <= y1 or x2 <= x1:
                    continue

                valid_crops.append(frame_array[y1:y2, x1:x2])
                crop_owner.append((b, i, [x1, y1, x2, y2]))

        if valid_crops:
            best_scores = None
            best_idxs = None

            if target_matrix is not None and len(target_uuids) > 0:
                embeddings = torch.cat([
                    batch_classify_cars(valid_crops[c:c + CROP_BATCH_SIZE], device, classifier)
                    for c in range(0, len(valid_crops), CROP_BATCH_SIZE)
                ], dim=0)
                # cosine similarity of every crop against every target, best target per crop
                scores, idxs = torch.mm(embeddings, target_matrix.t()).max(dim=1)
                best_scores = scores.tolist()
                best_idxs = idxs.tolist()

            for idx, (b, vehicle_id, box) in enumerate(crop_owner):
                sim_matrix = []
                if best_scores is not None:
                    target, score = best_idxs[idx], best_scores[idx]
                    sim_matrix.append((target_uuids[target], score))

                    if score >= record_thresholds[target]:
                        hits = batch_hits[b]
                        if target not in hits or score > hits[target][0]:
                            hits[target] = (score, idx)

                batch_detections[b].append({
                    "vehicle_id": vehicle_id,
                    "box": box,
                    "matches": sim_matrix
                })

        for b, (camera_uuid, frame_id, _) in enumerate(batch):
            payload = {
                "camera_uuid": camera_uuid,
                "detections": batch_detections[b],
                "frame_id": frame_id
            }
            if batch_hits[b]:
                # just the vehicle's bounding box; main.py stores these and strips them before broadcasting
                payload["sightings"] = [
                    {
                        "vehicle_uuid": target_uuids[target],
                        "score": score,
                        "jpeg": cv2.imencode('.jpg', valid_crops[idx], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                    }
                    for target, (score, idx) in batch_hits[b].items()
                ]

            try:
                result_queue.put_nowait(payload)
            except queue.Full:
                pass
            except:
                pass

        if profiler:
            frame_count += len(batch)
            if frame_count - last_profile_dump >= 50:
                last_profile_dump = frame_count
                import pstats
                profiler.disable()
                try:
                    with open(profile_path, 'w') as stream:
                        stats = pstats.Stats(profiler, stream=stream).sort_stats('tottime')
                        stats.print_stats(50)
                except Exception as e:
                    pass
                profiler.enable()