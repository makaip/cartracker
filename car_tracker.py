import asyncio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from ultralytics import YOLO
from deep_sort_realtime import DeepSort
from sklearn.metrics.pairwise import cosine_similarity

from retrieve import (
    load_camera_data,
    select_random_camera,
    stream_camera_video,
    update_camera_frame
)
from multi_camera import MultiCameraViewer
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarTracker:
    def __init__(self, num_cameras=4, data_dir="data", confidence_threshold=0.5):
        """Initialize the car tracking system.
        
        Args:
            num_cameras (int): Number of camera feeds to monitor
            data_dir (str): Directory containing car reference images
            confidence_threshold (float): Minimum confidence for car detection
        """

        self.num_cameras = num_cameras
        self.data_dir = Path(data_dir)
        self.confidence_threshold = confidence_threshold
        
        self.yolo_model = None
        self.tracker = None
        
        self.cameras_data = load_camera_data()
        self.camera_objects = []
        self.reference_cars = {}
        self.car_features = {}
        self.last_data_check = 0
        self.check_interval = 5
        
        self.browser = None
        self.playwright = None
        self.fig = None
        
    async def initialize(self):
        """Initialize all components of the car tracking system."""
        logger.info("Initializing car tracking system...")
        
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
            
        # DeepSORT tracker
        try:
            self.tracker = DeepSort(max_age=50, n_init=3)
            logger.info("DeepSORT tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            return False
        
        # reference car data
        self.load_reference_cars()
        
        if not await self.initialize_cameras():
            logger.error("Failed to initialize camera feeds")
            return False
            
        logger.info(f"Car tracking system initialized with {len(self.camera_objects)} cameras")
        return True
        
    def load_reference_cars(self):
        """Load reference car images from the data directory."""
        logger.info("Loading reference car images...")
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return
            
        car_count = 0
        for uuid_dir in self.data_dir.iterdir():
            if uuid_dir.is_dir():
                car_uuid = uuid_dir.name
                car_images = []
                
                for img_file in uuid_dir.glob("*.jpg") or uuid_dir.glob("*.png") or uuid_dir.glob("*.jpeg"):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            car_images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_file}: {e}")
                        
                if car_images:
                    self.reference_cars[car_uuid] = car_images
                    self.extract_car_features(car_uuid, car_images)
                    car_count += 1
                    
        logger.info(f"Loaded {car_count} reference cars with {sum(len(imgs) for imgs in self.reference_cars.values())} total images")
        
    def extract_car_features(self, car_uuid: str, images: List[np.ndarray]):
        """Extract features from reference car images using YOLO and create feature vectors."""
        if not self.yolo_model:
            return
            
        features = []
        for img in images:
            try:
                results = self.yolo_model(img)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # process car detections--class 2 in COCO dataset
                            if int(box.cls) == 2 and float(box.conf) > self.confidence_threshold:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                car_crop = img[y1:y2, x1:x2]
                                
                                # simple feature vector, color histogram + HOG
                                feature = self.create_feature_vector(car_crop)
                                if feature is not None:
                                    features.append(feature)
                                    
            except Exception as e:
                logger.warning(f"Failed to extract features from image for car {car_uuid}: {e}")
                
        if features:
            # Average the features for this car
            self.car_features[car_uuid] = np.mean(features, axis=0)
            logger.info(f"Extracted features for car {car_uuid} from {len(features)} detections")
        
    def create_feature_vector(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create a feature vector from a car image crop."""
        try:
            resized = cv2.resize(image, (64, 64))
            
            hist_b = cv2.calcHist([resized], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([resized], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([resized], [2], None, [16], [0, 256])
            
            hist_b = hist_b.flatten() / np.sum(hist_b)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_r = hist_r.flatten() / np.sum(hist_r)
            
            # Combine features
            feature_vector = np.concatenate([hist_b, hist_g, hist_r])
            
            return feature_vector
            
        except Exception as e:
            logger.warning(f"Failed to create feature vector: {e}")
            return None
            
    async def initialize_cameras(self):
        """Initialize the camera feeds using the multi-camera viewer."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            
            selected_cameras = {}
            camera_ids = list(self.cameras_data.keys())
            random.shuffle(camera_ids)
            
            for i in range(min(self.num_cameras, len(camera_ids))):
                camera_id = camera_ids[i]
                camera_name = self.cameras_data[camera_id]
                selected_cameras[camera_id] = camera_name
                
            plt.ion()
            
            if self.num_cameras <= 3:
                rows, cols = 1, self.num_cameras
            elif self.num_cameras <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3
                
            # create fig with gridspec
            self.fig = plt.figure(figsize=(cols*5, rows*4))
            gs = GridSpec(rows, cols, figure=self.fig)
            
            # init camera streams
            self.camera_objects = []
            index = 0
            
            for camera_id, camera_name in selected_cameras.items():
                row = index // cols
                col = index % cols
                
                ax = self.fig.add_subplot(gs[row, col])
                logger.info(f"Initializing camera {index+1}/{self.num_cameras}: {camera_name}")
                
                fig, ax, im_display, page, browser, image = await stream_camera_video(
                    camera_id, 
                    camera_name, 
                    figure=self.fig, 
                    ax=ax, 
                    browser=self.browser,
                    close_browser=False
                )
                
                if im_display is not None:
                    self.camera_objects.append({
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'page': page,
                        'im_display': im_display,
                        'ax': ax,
                        'tracker_id': index  # unique tracker ID for camera
                    })
                    
                index += 1
                
            self.fig.tight_layout()
            return len(self.camera_objects) > 0
            
        except Exception as e:
            logger.error(f"Failed to initialize cameras: {e}")
            return False
    
    def check_for_new_cars(self):
        """Check for new car directories and load them."""
        current_time = time.time()
        if current_time - self.last_data_check < self.check_interval:
            return
            
        self.last_data_check = current_time
        
        if not self.data_dir.exists():
            return
            
        existing_cars = set(self.reference_cars.keys())
        current_cars = set()
        
        for uuid_dir in self.data_dir.iterdir():
            if uuid_dir.is_dir():
                current_cars.add(uuid_dir.name)
                
        new_cars = current_cars - existing_cars
        
        if new_cars:
            logger.info(f"Found {len(new_cars)} new cars: {new_cars}")
            for car_uuid in new_cars:
                uuid_dir = self.data_dir / car_uuid
                car_images = []
                
                for img_file in uuid_dir.glob("*.jpg") or uuid_dir.glob("*.png") or uuid_dir.glob("*.jpeg"):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            car_images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load new image {img_file}: {e}")
                        
                if car_images:
                    self.reference_cars[car_uuid] = car_images
                    self.extract_car_features(car_uuid, car_images)
    
    def detect_and_track_cars(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """Detect and track cars in a frame."""
        if not self.yolo_model:
            return []
            
        detected_cars = []
        
        try:
            results = self.yolo_model(frame)
            
            detections = []
            confidences = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # process car detections: class 2 in COCO dataset
                        if int(box.cls) == 2 and float(box.conf) > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf)
                            
                            # DeepSORT format: [x, y, w, h]
                            detection = [x1, y1, x2 - x1, y2 - y1]
                            detections.append(detection)
                            confidences.append(conf)
            
            # Update tracker
            if detections:
                tracks = self.tracker.update_tracks(detections, frame=frame)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Extract car crop for matching
                    car_crop = frame[y1:y2, x1:x2]
                    feature_vector = self.create_feature_vector(car_crop)
                    
                    # Try to match with reference cars
                    matched_car = None
                    if feature_vector is not None:
                        matched_car = self.match_car(feature_vector)
                    
                    detected_cars.append({
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'camera_id': camera_id,
                        'matched_car': matched_car,
                        'timestamp': datetime.now()
                    })
                    
        except Exception as e:
            logger.warning(f"Error in car detection for camera {camera_id}: {e}")
            
        return detected_cars
    
    def match_car(self, feature_vector: np.ndarray, similarity_threshold: float = 0.7) -> Optional[str]:
        """Match a detected car with reference cars."""
        if not self.car_features:
            return None
            
        best_match = None
        best_similarity = 0
        
        for car_uuid, ref_features in self.car_features.items():
            try:
                similarity = cosine_similarity([feature_vector], [ref_features])[0][0]
                
                if similarity > best_similarity and similarity > similarity_threshold:
                    best_similarity = similarity
                    best_match = car_uuid
                    
            except Exception as e:
                logger.warning(f"Error calculating similarity for car {car_uuid}: {e}")
                
        if best_match:
            logger.info(f"Matched car {best_match} with similarity {best_similarity:.3f}")
            
        return best_match
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            matched_car = detection['matched_car']
            
            # Choose color based on match status
            if matched_car:
                color = (0, 255, 0)  # Green for matched cars
                label = f"Car {track_id}: {matched_car[:8]}"
            else:
                color = (255, 0, 0)  # Red for unmatched cars
                label = f"Car {track_id}: Unknown"
                
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                       
        return annotated_frame
    
    async def process_camera_frame(self, camera: Dict) -> bool:
        """Process a single camera frame for car detection and tracking."""
        try:
            # Update the camera frame
            success, frame_data = await update_camera_frame(camera['page'], camera['im_display'])
            
            if not success or frame_data is None:
                return False
                
            # Convert PIL image to OpenCV format
            if hasattr(frame_data, 'mode'):
                # It's a PIL image
                frame = cv2.cvtColor(np.array(frame_data), cv2.COLOR_RGB2BGR)
            else:
                # It's already a numpy array
                frame = frame_data
                
            # Detect and track cars
            detections = self.detect_and_track_cars(frame, camera['tracker_id'])
            
            # Draw detections on frame
            if detections:
                annotated_frame = self.draw_detections(frame, detections)
                
                # Convert back to RGB for matplotlib
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update the display
                camera['im_display'].set_array(annotated_frame_rgb)
                camera['ax'].set_title(f"{camera['camera_name']} - {len(detections)} cars detected")
                
                # Log matches
                matched_cars = [d['matched_car'] for d in detections if d['matched_car']]
                if matched_cars:
                    logger.info(f"Camera {camera['camera_name']}: Found matches for {matched_cars}")
            else:
                camera['ax'].set_title(f"{camera['camera_name']} - No cars detected")
                
            return True
            
        except Exception as e:
            logger.warning(f"Error processing frame for camera {camera['camera_name']}: {e}")
            return False
    
    async def run(self):
        """Run the car tracking system."""
        if not await self.initialize():
            logger.error("Failed to initialize car tracking system")
            return
            
        logger.info(f"Starting car tracking on {len(self.camera_objects)} cameras")
        logger.info(f"Monitoring {len(self.reference_cars)} reference cars")
        
        try:
            while True:
                # Check for new cars periodically
                self.check_for_new_cars()
                
                # Process each camera
                for camera in self.camera_objects:
                    await self.process_camera_frame(camera)
                
                # Update display
                plt.pause(0.5)
                
        except (KeyboardInterrupt, SystemExit):
            logger.info("Car tracking stopped by user")
        except Exception as e:
            logger.error(f"An error occurred during tracking: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up car tracking resources...")
        
        # Close all pages
        for camera in self.camera_objects:
            if 'page' in camera and camera['page'] and not camera['page'].is_closed():
                await camera['page'].close()
                
        # Close browser and playwright
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            
        # Close matplotlib figure
        if self.fig:
            plt.close(self.fig)
        plt.ioff()
        
        logger.info("All resources released")

async def main():
    """Main function to run the car tracking system."""
    # Parse command line arguments
    num_cameras = 4  # Default
    data_dir = "data"  # Default
    confidence = 0.5  # Default
    
    if len(sys.argv) > 1:
        try:
            num_cameras = int(sys.argv[1])
        except ValueError:
            logger.warning("Invalid number of cameras. Using default (4).")
    
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
        
    if len(sys.argv) > 3:
        try:
            confidence = float(sys.argv[3])
        except ValueError:
            logger.warning("Invalid confidence threshold. Using default (0.5).")
    
    logger.info(f"Starting car tracker with {num_cameras} cameras, data_dir='{data_dir}', confidence={confidence}")
    
    tracker = CarTracker(num_cameras=num_cameras, data_dir=data_dir, confidence_threshold=confidence)
    await tracker.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Script failed: {e}")
