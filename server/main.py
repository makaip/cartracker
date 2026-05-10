import json
import os
import uuid as uuid_lib
import shutil

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

import multiprocessing as mp
import queue

from retrieve import generate_frames
import database

import yaml

# command to forward port for WebSocket connection to HPC cluster
# ssh -L 8765:compute-node-name:8765 user@cluster.edu

FRAME_QUEUE_MAXSIZE = 64
RESULT_QUEUE_MAXSIZE = 256

frame_queue = mp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)     # in: (camera_uuid, frame_array)
result_queue = mp.Queue(maxsize=RESULT_QUEUE_MAXSIZE)   # out: match payload dict

# ----

WS_HOST         = "0.0.0.0"
WS_PORT         = 8765

app = FastAPI()
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
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message: {data}")

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get('/video_feed/{uuid}')
async def video_feed(uuid: str):
    if not uuid:
        raise HTTPException(status_code=400, detail="No UUID provided")
    
    return StreamingResponse(generate_frames(uuid), media_type='multipart/x-mixed-replace; boundary=frame')

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
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host='0.0.0.0', port=port, reload=True)