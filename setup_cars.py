#!/usr/bin/env python3
"""
Utility script to help set up the car tracking system.
This script can:
1. Create sample car directories with placeholder images
2. Download sample car images from the internet
3. Help organize existing car images
"""

import os
import sys
import uuid
import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_car_data(data_dir="data", num_cars=5):
    """Create sample car directories with placeholder images."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print(f"Creating {num_cars} sample car directories in {data_dir}/")
    
    # Sample car types and colors for variety
    car_types = ["sedan", "suv", "truck", "hatchback", "coupe"]
    colors = ["red", "blue", "white", "black", "silver", "gray", "green"]
    
    created_cars = []
    
    for i in range(num_cars):
        # Generate a UUID for the car
        car_uuid = str(uuid.uuid4())
        car_dir = data_path / car_uuid
        car_dir.mkdir(exist_ok=True)
        
        # Create some placeholder images
        car_type = random.choice(car_types)
        car_color = random.choice(colors)
        
        for angle in ["front", "rear", "side_left", "side_right"]:
            # Create a simple placeholder image
            img = Image.new('RGB', (300, 200), color=car_color)
            draw = ImageDraw.Draw(img)
            
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw car info on the image
            text = f"{car_color.title()} {car_type.title()}\n{angle.replace('_', ' ').title()}\nUUID: {car_uuid[:8]}"
            draw.text((10, 10), text, fill="white", font=font)
            
            # Add a simple car shape
            if angle == "front" or angle == "rear":
                # Draw a simple front/rear view
                draw.rectangle([50, 80, 250, 160], outline="black", width=3)
                draw.ellipse([60, 170, 90, 190], fill="black")  # Wheel
                draw.ellipse([210, 170, 240, 190], fill="black")  # Wheel
            else:
                # Draw a simple side view
                draw.rectangle([30, 100, 270, 150], outline="black", width=3)
                draw.ellipse([40, 150, 80, 180], fill="black")  # Wheel
                draw.ellipse([220, 150, 260, 180], fill="black")  # Wheel
            
            # Save the image
            img_path = car_dir / f"{angle}.jpg"
            img.save(img_path)
        
        created_cars.append({
            "uuid": car_uuid,
            "type": car_type,
            "color": car_color,
            "path": str(car_dir)
        })
        
        print(f"Created car {i+1}/{num_cars}: {car_color} {car_type} ({car_uuid[:8]})")
    
    # Save a manifest file
    manifest_path = data_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(created_cars, f, indent=2)
    
    print(f"\nCreated {len(created_cars)} sample cars in {data_dir}/")
    print(f"Manifest saved to {manifest_path}")
    return created_cars

def setup_data_directory(data_dir="data"):
    """Set up the data directory structure."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create a README file
    readme_path = data_path / "README.md"
    readme_content = """# Car Tracking Data Directory

This directory contains reference car images organized by UUID.

## Structure
```
data/
├── {uuid-1}/
│   ├── front.jpg
│   ├── rear.jpg
│   ├── side_left.jpg
│   └── side_right.jpg
├── {uuid-2}/
│   ├── angle1.jpg
│   ├── angle2.jpg
│   └── ...
└── manifest.json (optional, created by setup script)
```

## Adding New Cars

1. Create a new directory with a UUID name (or any unique identifier)
2. Add car images from different angles to that directory
3. Supported formats: .jpg, .jpeg, .png
4. The car tracker will automatically detect new directories

## Image Guidelines

- Include multiple angles of the same car for better recognition
- Ensure good lighting and clear visibility of the car
- Higher resolution images generally work better
- Try to minimize background clutter

## Usage

The car tracking system will:
1. Automatically scan this directory for car folders
2. Extract features from all images in each folder
3. Use these features to identify matching cars in live camera feeds
4. Check for new car additions every 30 seconds
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Data directory set up at {data_path}")
    print(f"README created at {readme_path}")

def validate_data_directory(data_dir="data"):
    """Validate the data directory and report on its contents."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist!")
        return False
    
    car_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not car_dirs:
        print(f"No car directories found in {data_dir}")
        return False
    
    print(f"Found {len(car_dirs)} car directories in {data_dir}:")
    
    total_images = 0
    for car_dir in car_dirs:
        images = list(car_dir.glob("*.jpg")) + list(car_dir.glob("*.jpeg")) + list(car_dir.glob("*.png"))
        total_images += len(images)
        print(f"  {car_dir.name}: {len(images)} images")
        
        if len(images) == 0:
            print(f"    WARNING: No images found in {car_dir.name}")
    
    print(f"\nTotal: {len(car_dirs)} cars, {total_images} images")
    return True

def main():
    """Main function with command line interface."""
    if len(sys.argv) < 2:
        print("Car Tracker Setup Utility")
        print("\nUsage:")
        print("  python setup_cars.py setup [data_dir]          - Set up data directory structure")
        print("  python setup_cars.py sample [data_dir] [num]   - Create sample car data")
        print("  python setup_cars.py validate [data_dir]       - Validate existing data")
        print("\nExamples:")
        print("  python setup_cars.py setup")
        print("  python setup_cars.py sample data 3")
        print("  python setup_cars.py validate")
        return
    
    command = sys.argv[1].lower()
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    if command == "setup":
        setup_data_directory(data_dir)
        
    elif command == "sample":
        num_cars = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        setup_data_directory(data_dir)
        create_sample_car_data(data_dir, num_cars)
        
    elif command == "validate":
        validate_data_directory(data_dir)
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: setup, sample, validate")

if __name__ == "__main__":
    main()
