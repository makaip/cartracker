# Car Tracker

An advanced car tracking system that uses multiple traffic cameras to detect and identify specific vehicles using YOLOv8 and DeepSORT/ReID models.

## Features

- **Multi-Camera Monitoring**: Monitor up to 9 traffic cameras simultaneously
- **Real-time Car Detection**: Uses YOLOv8 for fast and accurate car detection
- **Car Identification**: Matches detected cars against a database of reference images
- **Automatic Updates**: Periodically scans for new reference cars
- **Visual Interface**: Real-time display with bounding boxes and match information
- **Configurable**: Easy-to-modify configuration for different scenarios

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd cartracker
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model** (automatic on first run):
   The system will automatically download the YOLOv8 nano model on first use.

## Quick Start

### 1. Set up sample car data (for testing):
```bash
python setup_cars.py sample data 3
```

This creates 3 sample cars with placeholder images in the `data/` directory.

### 2. Run the car tracker:
```bash
python car_tracker.py
```

Or with custom parameters:
```bash
python car_tracker.py 4 data 0.5
```
- `4`: Number of cameras to monitor
- `data`: Directory containing reference car images  
- `0.5`: Confidence threshold for car detection

## Directory Structure

```
cartracker/
├── car_tracker.py          # Main tracking script
├── multi_camera.py         # Multi-camera viewer
├── retrieve.py             # Camera streaming utilities
├── setup_cars.py           # Setup utility for car data
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── traffic_cameras.json    # Camera data
└── data/                   # Reference car images
    ├── {uuid-1}/
    │   ├── front.jpg
    │   ├── rear.jpg
    │   ├── side_left.jpg
    │   └── side_right.jpg
    ├── {uuid-2}/
    │   ├── angle1.jpg
    │   └── angle2.jpg
    └── manifest.json       # Optional metadata
```

## Usage

### Basic Usage

Run with default settings (4 cameras, data/ directory):
```bash
python car_tracker.py
```

### Advanced Usage

```bash
python car_tracker.py [num_cameras] [data_directory] [confidence_threshold]
```

**Parameters:**
- `num_cameras`: Number of camera feeds (1-9, default: 4)
- `data_directory`: Path to reference car images (default: "data")
- `confidence_threshold`: YOLO detection confidence (0.0-1.0, default: 0.5)

**Examples:**
```bash
# Monitor 6 cameras with high confidence threshold
python car_tracker.py 6 data 0.7

# Use custom data directory
python car_tracker.py 4 my_cars 0.5

# Single camera monitoring
python car_tracker.py 1
```

## Adding Reference Cars

### Method 1: Manual Setup

1. Create a directory with a unique name (UUID recommended):
   ```
   data/12345678-1234-1234-1234-123456789abc/
   ```

2. Add car images from different angles:
   ```
   data/12345678-1234-1234-1234-123456789abc/
   ├── front.jpg
   ├── rear.jpg
   ├── side_left.jpg
   └── side_right.jpg
   ```

3. The system will automatically detect new cars within 30 seconds

### Method 2: Using Setup Script

```bash
# Set up directory structure
python setup_cars.py setup

# Create sample cars for testing
python setup_cars.py sample data 5

# Validate existing data
python setup_cars.py validate data
```

## Configuration

Edit `config.json` to customize the system:

```json
{
  "tracking": {
    "confidence_threshold": 0.5,    # YOLO detection confidence
    "similarity_threshold": 0.7,     # Car matching threshold
    "yolo_model": "yolov8n.pt",     # YOLO model variant
    "check_interval": 30             # Seconds between data scans
  },
  "display": {
    "update_interval": 0.5,         # Display refresh rate
    "show_unmatched_cars": true,    # Show unknown cars
    "show_tracking_ids": true       # Show tracking IDs
  }
}
```

## How It Works

### 1. Car Detection
- Uses YOLOv8 to detect cars in each camera frame
- Filters detections by confidence threshold
- Extracts car regions for further processing

### 2. Car Tracking
- Uses DeepSORT algorithm to track cars across frames
- Assigns unique tracking IDs to each car
- Maintains tracking through temporary occlusions

### 3. Car Identification
- Extracts feature vectors from detected cars (color histograms)
- Compares features with reference car database
- Uses cosine similarity for matching
- Only reports matches above similarity threshold

### 4. Reference Car Management
- Automatically scans data directory for new cars
- Extracts features from all reference images
- Averages features across multiple angles for robust matching

## Performance Tips

### For Better Speed:
- Use fewer cameras (1-4 recommended)
- Use YOLOv8n (nano) model instead of larger variants
- Reduce image resolution in camera streams
- Increase update_interval in config

### For Better Accuracy:
- Use multiple high-quality reference images per car
- Include images from different angles and lighting conditions
- Use higher confidence_threshold to reduce false positives
- Use YOLOv8s or YOLOv8m models for better detection

## Troubleshooting

### Common Issues

**"No camera data found"**
- Ensure `traffic_cameras.json` exists and contains valid camera data
- Check internet connection for camera access

**"Failed to load YOLO model"**
- Ensure sufficient disk space for model download
- Check internet connection for initial download

**"No cars detected"**
- Lower the confidence_threshold
- Check camera image quality
- Verify cameras are showing traffic

**"No car matches found"**
- Lower the similarity_threshold
- Add more reference images with better quality
- Ensure reference images contain clear car views

### Performance Issues

**High CPU/GPU usage:**
- Reduce number of cameras
- Use YOLOv8n instead of larger models
- Increase update intervals

**Memory issues:**
- Reduce number of reference cars
- Use smaller image sizes
- Monitor system resources

## Dependencies

- **Core**: Python 3.8+, OpenCV, NumPy
- **ML Models**: ultralytics (YOLOv8), deep-sort-realtime
- **Web**: playwright (for camera streaming)
- **Visualization**: matplotlib
- **Utilities**: PIL, scikit-learn

## Limitations

- Requires internet connection for camera access
- Performance depends on system hardware
- Car matching is based on visual features (may not work for very similar cars)
- Limited to traffic cameras supported by the existing system

## Future Enhancements

- Support for local video files
- Advanced ReID models for better car matching
- License plate recognition integration
- Database storage for tracking history
- Web-based interface
- Mobile app integration

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
