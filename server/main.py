import json
import os
import uuid as uuid_lib
import shutil
import asyncio
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

import multiprocessing as mp
import queue

from retrieve import generate_frames
import database

# command to forward port for WebSocket connection to HPC cluster
# ssh -L 8765:compute-node-name:8765 user@cluster.edu

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']

FRAME_QUEUE_MAXSIZE = config['frame_queue_maxsize']
RESULT_QUEUE_MAXSIZE = config['result_queue_maxsize']
WS_HOST = config['ws_host']
WS_PORT = config['ws_port']
NUM_GPUS = config['num_gpus']

frame_queue = mp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)     # in: (camera_uuid, frame_array)
result_queue = mp.Queue(maxsize=RESULT_QUEUE_MAXSIZE)   # out: match payload dict

# ----

CONNECTED_CLIENTS: set[WebSocket] = set()

async def result_broadcaster():
    while True:
        try:
            if CONNECTED_CLIENTS:
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
    import asyncio
    from detector import gpu_worker
    asyncio.run(gpu_worker(*args))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    broadcaster_task = asyncio.create_task(result_broadcaster())
    
    stop_event = mp.Event()
    update_event = mp.Event()
    
    workers = []
    import torch
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = min(NUM_GPUS, num_gpus_available)
    print(f"Starting {num_workers} GPU workers...")
    for i in range(num_workers):
        p = mp.Process(target=run_worker, args=(i, frame_queue, result_queue, stop_event, update_event))
        p.start()
        workers.append(p)
        
    app.state.update_event = update_event

    yield

    # shutdown
    broadcaster_task.cancel()
    stop_event.set()
    for _ in workers:
        frame_queue.put(None)
    for p in workers:
        p.join()

app = FastAPI(lifespan=lifespan)
UPLOAD_FOLDER = 'uploads/'

database.init_db()

def load_cameras() -> dict:
    try:
        with open('traffic_cameras.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

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
    return StreamingResponse(generate_frames(uuid, frame_queue, stop_event), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/add_vehicle')
async def add_vehicle(pictures: list[UploadFile] = File(...)):
    if not pictures or all(f.filename == '' for f in pictures):
        raise HTTPException(status_code=400, detail="No pictures uploaded")
    
    vehicle_uuid = str(uuid_lib.uuid4())
    upload_path = os.path.join(UPLOAD_FOLDER, vehicle_uuid)
    os.makedirs(upload_path, exist_ok=True)
    
    for f in pictures:
        if f.filename:
            filename = os.path.basename(f.filename)
            with open(os.path.join(upload_path, filename), 'wb') as buffer:
                shutil.copyfileobj(f.file, buffer)
            
    database.add_vehicle(vehicle_uuid)
    return {"uuid": vehicle_uuid}
    
@app.post('/delete_vehicle')
async def delete_vehicle(uuid: str = Form(...)):
    vehicle_uuid = uuid
    if not vehicle_uuid:
        raise HTTPException(status_code=400, detail="No UUID provided")
        
    database.delete_vehicle(vehicle_uuid)
    
    upload_path = os.path.join(UPLOAD_FOLDER, vehicle_uuid)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
    return {"status": "deleted", "uuid": vehicle_uuid}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", WS_PORT))
    uvicorn.run(app, host=WS_HOST, port=port, reload=True)