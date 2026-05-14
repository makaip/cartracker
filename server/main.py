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
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

import multiprocessing as mp
import queue

from retrieve import generate_frames, process_camera_stream
import database

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

frame_queue = MP_CONTEXT.Queue(maxsize=FRAME_QUEUE_MAXSIZE)     # in: (camera_uuid, frame_array)
result_queue = MP_CONTEXT.Queue(maxsize=RESULT_QUEUE_MAXSIZE)   # out: match payload dict

# ----

CONNECTED_CLIENTS: set[WebSocket] = set()
CAMERA_STATUS: dict[str, bool] = {}  # camera_uuid -> is_online

async def result_broadcaster():
    while True:
        try:
            if not CONNECTED_CLIENTS:
                await asyncio.sleep(0.05)
                continue

            payload_str = json.dumps(result_queue.get_nowait())
            for client in list(CONNECTED_CLIENTS):
                try:
                    await client.send_text(payload_str)
                except Exception as e:
                    print(f"Error sending to client: {e}")
        except queue.Empty:
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Broadcaster error: {e}")
            await asyncio.sleep(1)

def run_worker(*args):
    from detector import gpu_worker
    gpu_worker(*args)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    broadcaster_task = asyncio.create_task(result_broadcaster())
    
    stop_event = MP_CONTEXT.Event()
    update_event = MP_CONTEXT.Event()
    
    workers = []
    import torch
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = min(NUM_GPUS, num_gpus_available)
    print(f"Starting {num_workers} GPU workers...")
    for i in range(num_workers):
        p = MP_CONTEXT.Process(target=run_worker, args=(i, frame_queue, result_queue, stop_event, update_event))
        p.start()
        workers.append(p)
        
    app.state.update_event = update_event

    camera_tasks = []
    cameras = load_cameras()
    camera_uuids = list(cameras.keys()) if isinstance(cameras, dict) else cameras
    print(f"Starting background streams for {len(camera_uuids)} cameras...")
    for cam_uuid in camera_uuids:
        CAMERA_STATUS[cam_uuid] = True  # Initialize as online
        task = asyncio.create_task(process_camera_stream(cam_uuid, frame_queue, CAMERA_STATUS))
        camera_tasks.append(task)

    yield

    # shutdown
    for task in camera_tasks:
        task.cancel()
    broadcaster_task.cancel()
    stop_event.set()
    for _ in workers:
        frame_queue.put(None)
    for p in workers:
        p.join()

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
    try:
        with open(SERVER_DIR / 'tc_pbc.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.get('/cameras')
async def get_cameras():
    return load_cameras()

@app.get('/camera_status')
async def get_camera_status():
    """Return which cameras are online/offline"""
    return CAMERA_STATUS

@app.get('/vehicles')
async def list_vehicles():
    return database.get_vehicles()

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
async def add_vehicle(pictures: list[UploadFile] = File(...), name: str = Form(None)):
    if not pictures or all(f.filename == '' for f in pictures):
        raise HTTPException(status_code=400, detail="No pictures uploaded")
    
    vehicle_uuid = str(uuid_lib.uuid4())
    upload_path = UPLOAD_FOLDER / vehicle_uuid
    upload_path.mkdir(parents=True, exist_ok=True)
    
    for f in pictures:
        if f.filename:
            filename = os.path.basename(f.filename)
            with open(upload_path / filename, 'wb') as buffer:
                shutil.copyfileobj(f.file, buffer)
            
    database.add_vehicle(vehicle_uuid, name)
    app.state.update_event.set() # update embeddings in GPU workers

    return {"uuid": vehicle_uuid, "name": name}
    
@app.post('/delete_vehicle')
async def delete_vehicle(uuid: str = Form(...)):
    vehicle_uuid = uuid
    if not vehicle_uuid:
        raise HTTPException(status_code=400, detail="No UUID provided")
        
    database.delete_vehicle(vehicle_uuid)
    
    upload_path = UPLOAD_FOLDER / vehicle_uuid
    if upload_path.exists():
        shutil.rmtree(upload_path)
    
    app.state.update_event.set() # update embeddings in GPU workers
    return {"status": "deleted", "uuid": vehicle_uuid}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", WS_PORT))
    reload_enabled = os.environ.get("UVICORN_RELOAD", "0") == "1"
    uvicorn.run(app, host=WS_HOST, port=port, reload=reload_enabled)