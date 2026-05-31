from collections import deque
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

DETECTION_QUEUES = {}
BROADCAST_CALLBACK = None

def queue_detection(camera_uuid: str, frame_id: int, payload: dict):
    if camera_uuid not in DETECTION_QUEUES:
        DETECTION_QUEUES[camera_uuid] = []
    
    det_list = DETECTION_QUEUES[camera_uuid]
    payload_str = json.dumps(payload)
    det_list.append((frame_id, payload_str))
    det_list.sort(key=lambda x: x[0])


async def smooth_playback(
    camera_uuid: str,
        playback_dq: deque
) -> None:

    start_real_time = None
    start_frame_time = None

    while True:
        if not playback_dq:
            await asyncio.sleep(0.01)
            continue

        frame_time, frame_id, jpeg_bytes = playback_dq.popleft()

        if start_real_time is None:
            start_real_time = time.time()
            start_frame_time = frame_time

        target_real_time = start_real_time + (frame_time - start_frame_time)
        sleep_amount = target_real_time - time.time()

        if sleep_amount > 0:
            await asyncio.sleep(min(sleep_amount, 1.0))
        elif sleep_amount < -5.0:
            start_real_time = time.time()
            start_frame_time = frame_time

        LATEST_FRAMES[camera_uuid] = jpeg_bytes
        
        if BROADCAST_CALLBACK and camera_uuid in DETECTION_QUEUES:
            det_list = DETECTION_QUEUES[camera_uuid]
            
            payload_to_send = None
            while det_list and det_list[0][0] <= frame_id:
                if det_list[0][0] == frame_id:
                    payload_to_send = det_list[0][1]
                det_list.pop(0)
                
            if payload_to_send is not None:
                await BROADCAST_CALLBACK(payload_to_send)


def _stream_loop(
    url: str,
    camera_uuid: str,
    frame_queue: mp.Queue,
    camera_status: dict,
    frame_skip: int,
    playback_dq: deque
) -> None:

    try:
        container = av.open(url)
        camera_status[camera_uuid] = True
        frame_count = 0

        fps_fraction = container.streams.video[0].average_rate
        fps = float(fps_fraction) if fps_fraction else 30.0
        if fps <= 0:
            fps = 30.0
        time_per_frame = 1.0 / fps

        for frame in container.decode(video=0):
            frame_count += 1
            frame_array = frame.to_ndarray(format='bgr24')

            _, img_encoded = cv2.imencode(  # for frontend streaming
                '.jpg',
                frame_array,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            jpeg_bytes = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + \
                img_encoded.tobytes() + b'\r\n'

            if len(playback_dq) < 300:
                playback_dq.append((frame_count * time_per_frame, frame_count, jpeg_bytes))

            if frame_count % frame_skip == 0:
                shm = None
                try:
                    # allocate shared memory block based on array size
                    shm = shared_memory.SharedMemory(
                        create=True, size=frame_array.nbytes)
                    shm_array = np.ndarray(
                        frame_array.shape, dtype=frame_array.dtype, buffer=shm.buf)
                    shm_array[:] = frame_array[:]  # direct memory copy

                    frame_queue.put_nowait({
                        'camera_uuid': camera_uuid,
                        'shm_name': shm.name,
                        'shape': frame_array.shape,
                        'dtype': frame_array.dtype.str,
                        'frame_id': frame_count
                    })

                    # close local handle
                    # data persists until unlinked

                    shm.close()

                except queue.Full:  # if queue full clean up memory block so no leak
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


async def process_camera_stream(
    camera_uuid: str,
    frame_queue: mp.Queue,
    camera_status: dict
) -> None:
    """continuously push stream frames to the GPU worker queue."""

    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"

    playback_dq = deque()
    playback_task = asyncio.create_task(
        smooth_playback(camera_uuid, playback_dq)
    )

    while True:
        e = await asyncio.to_thread(
            _stream_loop,
            url,
            camera_uuid,
            frame_queue,
            camera_status,
            FRAME_SKIP,
            playback_dq
        )

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

            hz = 50  # 50 Hz polling rate fine for viewing
            await asyncio.sleep(1.0 / hz)

        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            break

    print(f"Stopping HTTP frame generation for camera {camera_uuid}")
