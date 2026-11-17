from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
# seconds the displayed video runs behind the decoder, so a frame's detections
# have time to come back from the GPUs before that frame is shown
PLAYBACK_DELAY = float(config.get('playback_delay', 3.0))
# a camera whose frame got no result within this many seconds may submit again
IN_FLIGHT_TIMEOUT = 5.0

# camera guid -> .m3u8 playlist URL (written by scrape/pbc_uuids.py)
with open(SERVER_DIR / 'tc_pbc_urls.json', 'r') as f:
    CAMERA_URLS = json.load(f)

# Each camera's _stream_loop blocks one thread for as long as the stream is up.
# asyncio.to_thread uses the default executor, capped at min(32, cpus + 4) threads,
# so only the first 32 cameras ever started. Give every camera its own thread.
STREAM_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, len(CAMERA_URLS)),
    thread_name_prefix='camera-stream'
)


LATEST_FRAMES = {}

DETECTION_QUEUES = {}
BROADCAST_CALLBACK = None

# camera_uuid -> time.monotonic() when its frame was handed to the GPU workers.
# A camera submits its next frame only after the previous result came back, so the
# GPUs are shared evenly across cameras and never fed faster than they can process.
# Without this, 200 cameras flooded the queue and every result arrived too late to show.
IN_FLIGHT = {}

def mark_result(camera_uuid: str) -> None:
    IN_FLIGHT.pop(camera_uuid, None)

def _is_in_flight(camera_uuid: str) -> bool:
    sent = IN_FLIGHT.get(camera_uuid)
    return sent is not None and (time.monotonic() - sent) < IN_FLIGHT_TIMEOUT

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

        target_real_time = start_real_time + (frame_time - start_frame_time) + PLAYBACK_DELAY
        sleep_amount = target_real_time - time.time()

        if sleep_amount > 0:
            await asyncio.sleep(min(sleep_amount, PLAYBACK_DELAY + 1.0))
        elif sleep_amount < -5.0:
            start_real_time = time.time()
            start_frame_time = frame_time

        LATEST_FRAMES[camera_uuid] = jpeg_bytes
        
        if BROADCAST_CALLBACK and camera_uuid in DETECTION_QUEUES:
            det_list = DETECTION_QUEUES[camera_uuid]
            
            payload_to_send = None
            # send the newest result at or before the frame being shown; a result that
            # arrives after its frame was shown is still sent, just slightly late
            while det_list and det_list[0][0] <= frame_id:
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
) -> tuple[Exception | None, int]:

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

            if frame_count % frame_skip == 0 and not _is_in_flight(camera_uuid):
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
                    IN_FLIGHT[camera_uuid] = time.monotonic()

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
        return e, 0
    
    return None, frame_count


async def process_camera_stream(
    camera_uuid: str,
    frame_queue: mp.Queue,
    camera_status: dict
) -> None:
    url = CAMERA_URLS[camera_uuid]
    playback_dq = deque()
    playback_task = asyncio.create_task(smooth_playback(camera_uuid, playback_dq))

    while True:
        # attempt 1 (try normally)
        result = await asyncio.get_running_loop().run_in_executor(
            STREAM_EXECUTOR, _stream_loop, url, camera_uuid, frame_queue, camera_status, FRAME_SKIP, playback_dq
        )
        
        e, frame_count = result if isinstance(result, tuple) else (result, 0)

        if e is None and frame_count > 0:
            camera_status[camera_uuid] = False
            await asyncio.sleep(2.0)
            continue

        # attempt 1 failed or was black (0 frames)
        camera_status[camera_uuid] = False
        await asyncio.sleep(5.0)

        # attempt 2 (refresh)
        print(f"[{camera_uuid}] Retrying stream...")
        result = await asyncio.get_running_loop().run_in_executor(
            STREAM_EXECUTOR, _stream_loop, url, camera_uuid, frame_queue, camera_status, FRAME_SKIP, playback_dq
        )
        e, frame_count = result if isinstance(result, tuple) else (result, 0)

        if e is None and frame_count > 0:
            camera_status[camera_uuid] = False
            await asyncio.sleep(2.0)
            continue

        print(f"[{camera_uuid}] Stream totally offline. Giving up for 60 seconds.")
        camera_status[camera_uuid] = "FAILED"
        
        await asyncio.sleep(60.0)  # wait 60 seconds before retrying

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
