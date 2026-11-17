import json
import os
import uuid as uuid_lib
import shutil
import asyncio
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import multiprocessing as mp
from multiprocessing import shared_memory
import queue

import retrieve
from retrieve import process_camera_stream, generate_frames
import database
import sightings

import cProfile
import pstats
import io

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
YOLO_CONFIG_DIR = PROJECT_DIR / '.cache' / 'ultralytics'

os.environ.setdefault('YOLO_CONFIG_DIR', str(YOLO_CONFIG_DIR))

MP_CONTEXT = mp.get_context('spawn')

# command to forward port for WebSocket connection to HPC cluster
# ssh -L 8765:compute-node-name:8765 user@cluster.edu

with open(SERVER_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']

FRAME_QUEUE_MAXSIZE = config['frame_queue_maxsize']
RESULT_QUEUE_MAXSIZE = config['result_queue_maxsize']
WS_HOST = config['ws_host']
WS_PORT = config['ws_port']
NUM_GPUS = config['num_gpus']

frame_queue = MP_CONTEXT.Queue(maxsize=FRAME_QUEUE_MAXSIZE)     # in: dict containing shm pointers
result_queue = MP_CONTEXT.Queue(maxsize=RESULT_QUEUE_MAXSIZE)   # out: match payload dict

# ----

# set by run_hpc.py when the cameras are split across several nodes
SHARD_INDEX = int(os.environ.get('SHARD_INDEX', 0))
SHARD_COUNT = max(1, int(os.environ.get('SHARD_COUNT', 1)))

CONNECTED_CLIENTS: set[WebSocket] = set()
CAMERA_STATUS: dict[str, bool] = {}  # camera_uuid -> is_online
CAMERA_NAMES: dict[str, str] = {}    # camera_uuid -> name, stored with each sighting as its location

async def result_broadcaster():
    while True:
        try:
            # always drain: results also release the camera's slot for its next frame
            payload = result_queue.get_nowait()
            retrieve.mark_result(payload['camera_uuid'])

            # sightings are recorded whether or not a client is watching
            hits = payload.pop('sightings', None)
            if hits:
                camera_uuid = payload['camera_uuid']
                await asyncio.to_thread(sightings.record, camera_uuid, CAMERA_NAMES.get(camera_uuid), hits)

            if CONNECTED_CLIENTS:
                retrieve.queue_detection(payload['camera_uuid'], payload['frame_id'], payload)
        except queue.Empty:
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Broadcaster error: {e}")
            await asyncio.sleep(1)

async def vehicle_db_watcher(update_events):
    """reload target embeddings when vehicles.db changes on disk.

    With several nodes the client adds/deletes vehicles through one server only; the
    others share the same database file and pick the change up here.
    """
    def mtime():
        try:
            return database.DB_PATH.stat().st_mtime_ns
        except OSError:
            return None

    last = mtime()
    while True:
        await asyncio.sleep(2.0)
        current = mtime()
        if current != last:
            last = current
            for e in update_events:
                e.set()

def run_worker(*args):
    from detector import gpu_worker
    gpu_worker(*args)

async def my_broadcast(payload_str):
    for client in list(CONNECTED_CLIENTS):
        try:
            await client.send_text(payload_str)
        except Exception:
            pass

retrieve.BROADCAST_CALLBACK = my_broadcast

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    broadcaster_task = asyncio.create_task(result_broadcaster())
    
    stop_event = MP_CONTEXT.Event()
    
    workers = []
    import torch
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = min(NUM_GPUS, num_gpus_available)
    update_events = [MP_CONTEXT.Event() for _ in range(num_workers)]
    print(f"Starting {num_workers} GPU workers...")
    for i in range(num_workers):
        p = MP_CONTEXT.Process(target=run_worker, args=(i, frame_queue, result_queue, stop_event, update_events[i]))
        p.start()
        workers.append(p)
        
    app.state.update_events = update_events
    watcher_task = asyncio.create_task(vehicle_db_watcher(update_events))

    camera_tasks = []
    cameras = load_cameras()
    camera_uuids = list(cameras.keys()) if isinstance(cameras, dict) else cameras
    if isinstance(cameras, dict):
        CAMERA_NAMES.update(cameras)
    print(f"Starting background streams for {len(camera_uuids)} cameras (node {SHARD_INDEX + 1} of {SHARD_COUNT})...")
    for cam_uuid in camera_uuids:
        CAMERA_STATUS[cam_uuid] = True  # Initialize as online
        task = asyncio.create_task(process_camera_stream(cam_uuid, frame_queue, CAMERA_STATUS))
        camera_tasks.append(task)

    yield

    # shutdown
    for task in camera_tasks:
        task.cancel()
    broadcaster_task.cancel()
    watcher_task.cancel()
    stop_event.set()
    
    for _ in workers:
        try:
            frame_queue.put_nowait(None)
        except queue.Full:
            pass
            
    for p in workers:
        p.join()
    
    # clean up any remaining shared memory blocks left in the queue on shutdown
    while not frame_queue.empty():
        try:
            item = frame_queue.get_nowait()
            if item is not None and isinstance(item, dict) and 'shm_name' in item:
                try:
                    shm = shared_memory.SharedMemory(name=item['shm_name'])
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        except queue.Empty:
            break

app = FastAPI(lifespan=lifespan)

# enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = SERVER_DIR / 'uploads'

database.init_db()

def load_cameras() -> dict:
    """cameras handled by this server; with several nodes, each takes every Nth camera"""
    try:
        with open(SERVER_DIR / 'tc_pbc.json', 'r') as f:
            cameras = json.load(f)
    except FileNotFoundError:
        return {}

    if SHARD_COUNT > 1:
        return {uuid: cameras[uuid] for uuid in list(cameras)[SHARD_INDEX::SHARD_COUNT]}
    return cameras

@app.get('/cameras')
async def get_cameras():
    return load_cameras()

@app.get('/camera_status')
async def get_camera_status():
    """Return which cameras are online/offline"""
    return CAMERA_STATUS

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}

def vehicle_images(vehicle_uuid: str) -> list[str]:
    upload_path = UPLOAD_FOLDER / vehicle_uuid
    if not upload_path.is_dir():
        return []
    return sorted(p.name for p in upload_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

def save_images(vehicle_uuid: str, pictures: list[UploadFile]) -> int:
    """store uploaded reference images; returns how many were saved"""
    upload_path = UPLOAD_FOLDER / vehicle_uuid
    upload_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in pictures:
        suffix = Path(f.filename or '').suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        # own name: uploads from different folders may share a file name
        with open(upload_path / f'{uuid_lib.uuid4().hex[:12]}{suffix}', 'wb') as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved += 1
    return saved

def require_vehicle(vehicle_uuid: str) -> dict:
    vehicle = database.get_vehicle(vehicle_uuid)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Unknown vehicle")
    return vehicle

def reload_targets():
    """make the GPU workers rebuild their target embeddings; other nodes notice vehicles.db changing"""
    database.touch_vehicles()
    for e in app.state.update_events:
        e.set()

@app.get('/vehicles')
def list_vehicles():
    stats = database.sighting_stats()
    vehicles = database.get_vehicles()
    for v in vehicles:
        v['tracking'] = bool(v['tracking'])
        v['images'] = vehicle_images(v['uuid'])
        v['sightings'] = stats.get(v['uuid'], {}).get('count', 0)
        v['last_seen'] = stats.get(v['uuid'], {}).get('last_seen')
    return vehicles

class VehicleUpdate(BaseModel):
    name: str | None = None
    tracking: bool | None = None
    threshold: float | None = None

@app.patch('/vehicles/{vehicle_uuid}')
def update_vehicle(vehicle_uuid: str, update: VehicleUpdate):
    require_vehicle(vehicle_uuid)
    fields = update.model_dump(exclude_none=True)
    if 'threshold' in fields and not 0 <= fields['threshold'] <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    if 'tracking' in fields:
        fields['tracking'] = int(fields['tracking'])

    database.update_vehicle(vehicle_uuid, **fields)
    if 'tracking' in fields or 'threshold' in fields:
        reload_targets()
    return {"status": "updated", "uuid": vehicle_uuid}

@app.get('/vehicles/{vehicle_uuid}/images/{filename}')
def get_vehicle_image(vehicle_uuid: str, filename: str):
    require_vehicle(vehicle_uuid)
    if filename not in vehicle_images(vehicle_uuid):
        raise HTTPException(status_code=404, detail="Unknown image")
    return FileResponse(UPLOAD_FOLDER / vehicle_uuid / filename)

@app.post('/vehicles/{vehicle_uuid}/images')
def add_vehicle_images(vehicle_uuid: str, pictures: list[UploadFile] = File(...)):
    require_vehicle(vehicle_uuid)
    if save_images(vehicle_uuid, pictures) == 0:
        raise HTTPException(status_code=400, detail="No supported images uploaded (jpg, png, webp)")
    reload_targets()
    return {"images": vehicle_images(vehicle_uuid)}

@app.delete('/vehicles/{vehicle_uuid}/images/{filename}')
def delete_vehicle_image(vehicle_uuid: str, filename: str):
    require_vehicle(vehicle_uuid)
    if filename not in vehicle_images(vehicle_uuid):
        raise HTTPException(status_code=404, detail="Unknown image")
    (UPLOAD_FOLDER / vehicle_uuid / filename).unlink()
    reload_targets()
    return {"images": vehicle_images(vehicle_uuid)}

@app.get('/sightings')
def list_sightings(vehicle: str | None = None, limit: int = 100, offset: int = 0, min_score: float = 0.0):
    total, items = database.get_sightings(vehicle, max(1, min(limit, 500)), max(0, offset), min_score)
    return {"total": total, "items": items}

@app.get('/sightings/{sighting_id}/image')
def get_sighting_image(sighting_id: int):
    sighting = database.get_sighting(sighting_id)
    if sighting is None or not sightings.image_path(sighting['image']).is_file():
        raise HTTPException(status_code=404, detail="Unknown sighting")
    return FileResponse(sightings.image_path(sighting['image']))

@app.delete('/sightings/{sighting_id}')
def delete_sighting(sighting_id: int):
    image = database.delete_sighting(sighting_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Unknown sighting")
    sightings.remove_images([image])
    return {"status": "deleted", "id": sighting_id}

@app.delete('/sightings')
def clear_sightings(vehicle: str):
    images = database.delete_sightings(vehicle)
    sightings.remove_images(images)
    return {"status": "deleted", "count": len(images)}

@app.get('/')
async def index() -> str:
    return HTMLResponse("<h1>Car Tracker Server</h1>")

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    CONNECTED_CLIENTS.add(websocket)
    print(f"WebSocket client connected. Total clients: {len(CONNECTED_CLIENTS)}")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message: {data}")

    except Exception as e:
        print(f"WebSocket client disconnected: {e}")
    finally:
        if websocket in CONNECTED_CLIENTS:
            CONNECTED_CLIENTS.remove(websocket)
        print(f"WebSocket client removed. Total clients: {len(CONNECTED_CLIENTS)}")

@app.get('/video_feed/{uuid}')
async def video_feed(uuid: str):
    if not uuid:
        raise HTTPException(status_code=400, detail="No UUID provided")
    
    stop_event = asyncio.Event()
    return StreamingResponse(generate_frames(uuid, stop_event), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/add_vehicle')
def add_vehicle(pictures: list[UploadFile] = File(...), name: str = Form(None)):
    vehicle_uuid = str(uuid_lib.uuid4())

    if save_images(vehicle_uuid, pictures) == 0:
        shutil.rmtree(UPLOAD_FOLDER / vehicle_uuid, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No supported images uploaded (jpg, png, webp)")

    database.add_vehicle(vehicle_uuid, name)
    for e in app.state.update_events: e.set() # update embeddings in GPU workers

    return {"uuid": vehicle_uuid, "name": name}

@app.post('/delete_vehicle')
def delete_vehicle(uuid: str = Form(...)):
    vehicle_uuid = uuid
    require_vehicle(vehicle_uuid)

    database.delete_vehicle(vehicle_uuid)
    sightings.remove_images(database.delete_sightings(vehicle_uuid))

    upload_path = UPLOAD_FOLDER / vehicle_uuid
    if upload_path.exists():
        shutil.rmtree(upload_path)

    for e in app.state.update_events: e.set() # update embeddings in GPU workers
    return {"status": "deleted", "uuid": vehicle_uuid}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", WS_PORT))
    reload_enabled = os.environ.get("UVICORN_RELOAD", "0") == "1"
    uvicorn.run(app, host=WS_HOST, port=port, reload=reload_enabled)