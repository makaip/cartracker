# Car Tracker / Traffic Camera Viewer

This project allows you to view live traffic camera feeds from a selection of cameras.

## Files

- `retrieve.py` - Module for retrieving and displaying a single camera feed
- `multi_camera.py` - Module for displaying multiple camera feeds simultaneously
- `uuids.py` - Script to update the list of available traffic cameras
- `traffic_cameras.json` - JSON file containing camera IDs and names

## Requirements

- Python 3.7+
- Playwright (Chrome)
- matplotlib
- PIL (Pillow)
- requests
- beautifulsoup4 (for updating camera list)

## Installation

```bash
pip install playwright pillow matplotlib requests beautifulsoup4
playwright install chrome
```

## Usage

### Single Camera Mode

To view a single random traffic camera:

```bash
python retrieve.py
```

### Multi-Camera Mode

To view multiple traffic camera feeds simultaneously:

```bash
python multi_camera.py [number_of_cameras]
```

Where `number_of_cameras` is an optional parameter specifying how many cameras to display (default is 4, maximum is 9).

Examples:
```bash
# Show 4 cameras (default)
python multi_camera.py

# Show 6 cameras
python multi_camera.py 6
```

### Updating Camera List

To update the list of available traffic cameras:

```bash
python uuids.py
```

## How It Works

1. Camera data is loaded from `traffic_cameras.json`
2. For each camera, a headless Chrome browser is launched using Playwright
3. The video feed is captured frame-by-frame and displayed using matplotlib
4. In multi-camera mode, the display is organized in a grid layout

## Controls

- Press `Ctrl+C` in the terminal to stop the video stream(s)
