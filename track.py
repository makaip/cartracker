#!/usr/bin/env python3
"""
Simple car tracking entry point.
This script provides a simplified interface to the car tracking system.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Import the main car tracker
from car_tracker import CarTracker

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_usage():
    """Print usage information."""
    print("Car Tracker - Simple Interface")
    print("\nUsage:")
    print("  python track.py                    # Use defaults (4 cameras, data/ dir)")
    print("  python track.py [cameras]          # Specify number of cameras")
    print("  python track.py [cameras] [dir]    # Specify cameras and data directory")
    print("\nExamples:")
    print("  python track.py                    # 4 cameras, data/ directory")
    print("  python track.py 2                  # 2 cameras, data/ directory")
    print("  python track.py 6 my_cars         # 6 cameras, my_cars/ directory")
    print("\nBefore running:")
    print("  1. Set up car data: python setup_cars.py sample")
    print("  2. Install dependencies: pip install -r requirements.txt")

async def run_tracker(num_cameras=4, data_dir="data"):
    """Run the car tracker with specified parameters."""
    # Validate data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        print(f"Create it with: python setup_cars.py sample {data_dir}")
        return False
    
    # Check if there are any car directories
    car_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not car_dirs:
        print(f"Warning: No car directories found in '{data_dir}'")
        print(f"Add sample cars with: python setup_cars.py sample {data_dir}")
        print("Continuing anyway - you can add cars while the tracker is running...")
    
    # Create and run the tracker
    tracker = CarTracker(num_cameras=num_cameras, data_dir=data_dir)
    await tracker.run()
    return True

def main():
    """Main function with command line parsing."""
    setup_logging()
    
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        return
    
    # Parse command line arguments
    num_cameras = 4
    data_dir = "data"
    
    if len(sys.argv) > 1:
        try:
            num_cameras = int(sys.argv[1])
            if num_cameras < 1 or num_cameras > 9:
                print("Error: Number of cameras must be between 1 and 9")
                return
        except ValueError:
            print(f"Error: Invalid number of cameras '{sys.argv[1]}'")
            print_usage()
            return
    
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    
    # Display configuration
    print(f"Starting Car Tracker:")
    print(f"  Cameras: {num_cameras}")
    print(f"  Data Directory: {data_dir}")
    print(f"  Press Ctrl+C to stop")
    print()
    
    # Run the tracker
    try:
        success = asyncio.run(run_tracker(num_cameras, data_dir))
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTracker stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()