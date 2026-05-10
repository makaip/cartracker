import asyncio
import json
import time
from io import BytesIO
import numpy as np
import av
import cv2
import multiprocessing as mp
import yaml

from detector import gpu_worker

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']
    
FRAME_SKIP = config['frame_skip']

async def generate_frames(
        camera_uuid: str,
        frame_queue: mp.Queue,
        stop_event: asyncio.Event
    ):
    """async entry point for generating frames from a camera stream"""

    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"
    
    while not stop_event.is_set():
        try:
            async for frame_bytes in frame_generator(url, FRAME_SKIP, camera_uuid, frame_queue):
                yield frame_bytes
                if stop_event.is_set():
                    break
        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            break
    
    print(f"Stopping frame generation for camera {camera_uuid}")

async def frame_generator(url: str,                 # URL of the camera's HLS stream
                          k_skip: int,              # process every k frames for detection
                          camera_uuid: str,         # unique identifier for the camera
                          frame_queue: mp.Queue):   # queue to send frames to GPU worker
    """generate continuous JPEG frames from the cameras HLS stream"""

    container = av.open(url)
    last_frame_time = None
    last_real_time = None
    frame_count = 0

    for frame in container.decode(video=0):
        if last_frame_time is not None and frame.time is not None:
            frame_delay = float(frame.time - last_frame_time)
            real_elapsed = time.time() - last_real_time
            
            sleep_time = frame_delay - real_elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        if frame.time is not None:
            last_frame_time = frame.time
            last_real_time = time.time()

        frame_array = frame.to_ndarray(format='bgr24')
        
        frame_count += 1
        if frame_count % k_skip == 0:
            try:
                frame_queue.put_nowait((camera_uuid, frame_array))
            except Exception as e:
                pass # queue is full

        _, img_encoded = cv2.imencode('.jpg', frame_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = img_encoded.tobytes()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')