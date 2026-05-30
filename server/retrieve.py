import asyncio
import json
import time
import queue
from io import BytesIO
import numpy as np
import av
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import yaml
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent

with open(SERVER_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']

FRAME_SKIP = config['frame_skip']


LATEST_FRAMES = {}


def _stream_loop(url: str, camera_uuid: str, frame_queue: mp.Queue, camera_status: dict, frame_skip: int):
    try:
        container = av.open(url)
        camera_status[camera_uuid] = True
        frame_count = 0
        for frame in container.decode(video=0):
            frame_count += 1
            frame_array = frame.to_ndarray(format='bgr24')

            # Encode for frontend view
            _, img_encoded = cv2.imencode(
                '.jpg',
                frame_array,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            jpeg_bytes = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + \
                img_encoded.tobytes() + b'\r\n'
            LATEST_FRAMES[camera_uuid] = jpeg_bytes

            if frame_count % frame_skip == 0:
                shm = None
                try:
                    # allocate shared memory block based on array size
                    shm = shared_memory.SharedMemory(create=True, size=frame_array.nbytes)
                    shm_array = np.ndarray(frame_array.shape, dtype=frame_array.dtype, buffer=shm.buf)
                    shm_array[:] = frame_array[:]  # direct memory copy

                    frame_queue.put_nowait({
                        'camera_uuid': camera_uuid,
                        'shm_name': shm.name,
                        'shape': frame_array.shape,
                        'dtype': frame_array.dtype.str
                    })

                    # close local handle
                    # data persists until unlinked

                    shm.close()

                except queue.Full: # if queue full clean up memory block so no leak
                    if shm is not None:
                        shm.close()
                        shm.unlink()

                except Exception as e:
                    if shm is not None:
                        shm.close()

                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            pass

    except Exception as e:
        return e
    return None


async def process_camera_stream(camera_uuid: str,
                                frame_queue: mp.Queue,
                                camera_status: dict) -> None:
    """continuously push stream frames to the GPU worker queue."""
    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"

    while True:
        e = await asyncio.to_thread(_stream_loop, url, camera_uuid, frame_queue, camera_status, FRAME_SKIP)
        camera_status[camera_uuid] = False
        if e is not None:
            error_str = str(e)
            if '404' in error_str or 'Not Found' in error_str:
                print(f"Background stream error for {camera_uuid}: {e}")
            else:
                print(f"Background stream error for {camera_uuid}: {e}")
        await asyncio.sleep(5.0)


async def generate_frames(
    camera_uuid: str,
    stop_event: asyncio.Event
):
    """async entry point for generating frames from a camera stream for HTTP video viewing"""

    last_frame = None
    while not stop_event.is_set():
        try:
            frame = LATEST_FRAMES.get(camera_uuid)
            if frame is not None and frame != last_frame:
                last_frame = frame
                yield frame

            # ~50 Hz polling rate is fine for viewing
            await asyncio.sleep(0.02)

        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            break

    print(f"Stopping HTTP frame generation for camera {camera_uuid}")