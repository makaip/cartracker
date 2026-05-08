import json
import time
from io import BytesIO
import av
from PIL import Image
import cv2
from detector import detect_cars

def generate_frames(camera_uuid: str) -> bytes:
    """generate continuous JPEG frames from the cameras HLS stream"""

    url = f"https://pbcvideostreams1.pbc.gov/memfs/{camera_uuid}.m3u8"
    print(f"Opening video stream for {camera_uuid}: {url}")
    
    try:
        container = av.open(url)
        last_frame_time = None
        last_real_time = None

        for frame in container.decode(video=0):
            if last_frame_time is not None and frame.time is not None:
                frame_delay = float(frame.time - last_frame_time)
                real_elapsed = time.time() - last_real_time
                
                sleep_time = frame_delay - real_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            if frame.time is not None:
                last_frame_time = frame.time
                last_real_time = time.time()

            frame_array = frame.to_ndarray(format='bgr24')

            results = detect_cars(frame_array)
            annotated_array = results[0].plot()

            img = Image.fromarray(cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB))
            
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=85)
            frame_bytes = buf.getvalue()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
