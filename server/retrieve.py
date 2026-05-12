import asyncio
import json
import time
from io import BytesIO
import numpy as np
import av
import cv2
import multiprocessing as mp
import yaml
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent

with open(SERVER_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']
    
FRAME_SKIP = config['frame_skip']


def _next_frame(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

async def process_camera_stream(camera_uuid: str, 
                                frame_queue: mp.Queue,
                                camera_status: dict) -> None:
    """continuously push stream frames to the GPU worker queue."""
    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"

    while True:
        try:
            container = await asyncio.to_thread(av.open, url)
            camera_status[camera_uuid] = True  # Mark as online
            frame_count = 0
            iterator = container.decode(video=0)
            
            while True:
                frame = await asyncio.to_thread(_next_frame, iterator)
                if frame is None:
                    break
                    
                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    frame_array = frame.to_ndarray(format='bgr24')
                    try:
                        frame_queue.put_nowait((camera_uuid, frame_array))
                    except Exception:
                        pass # queue is full

                await asyncio.sleep(0.001)
        except Exception as e:
            error_str = str(e)
            if '404' in error_str or 'Not Found' in error_str:
                camera_status[camera_uuid] = False  # mark offline
                print(f"Background stream error for {camera_uuid}: {e}")
            else:
                print(f"Background stream error for {camera_uuid}: {e}")
            await asyncio.sleep(5.0)

async def generate_frames(
        camera_uuid: str,
        stop_event: asyncio.Event
    ):
    """async entry point for generating frames from a camera stream for HTTP video viewing"""

    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"
    
    while not stop_event.is_set():
        try:
            async for frame_bytes in frame_generator(url):
                yield frame_bytes
                if stop_event.is_set():
                    break
        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            break
            
    print(f"Stopping HTTP frame generation for camera {camera_uuid}")

async def frame_generator(url: str):
    """generate continuous JPEG frames from the cameras HLS stream"""

    container = await asyncio.to_thread(av.open, url)
    last_frame_time = None
    last_real_time = None
    iterator = container.decode(video=0)

    while True:
        frame = await asyncio.to_thread(_next_frame, iterator)
        if frame is None:
            break
            
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
        _, img_encoded = cv2.imencode('.jpg', frame_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')