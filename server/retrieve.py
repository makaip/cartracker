import asyncio
import json
import time
from io import BytesIO
import numpy as np
import av
import cv2
import multiprocessing as mp

from detector import gpu_worker

async def generate_frames(
        camera_uuid: str,
        frame_queue: mp.Queue,
        stop_event: asyncio.Event
    ):
    """async entry point for generating frames from a camera stream"""

    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"
    
    while not stop_event.is_set():
        try:
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_generator(url) + b'\r\n')

        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            break
    
    print(f"Stopping frame generation for camera {camera_uuid}")

async def frame_generator(url: str):
    """generate continuous JPEG frames from the cameras HLS stream"""

    container = av.open(url)
    last_frame_time = None
    last_real_time = None

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

        # revisit - might not be optimal
        img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        frame_bytes = buf.getvalue()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')