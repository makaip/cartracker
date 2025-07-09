import asyncio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import random
from retrieve import (
    load_camera_data,
    select_random_camera,
    stream_camera_video,
    update_camera_frame
)
from playwright.async_api import async_playwright

class MultiCameraViewer:
    def __init__(self, num_cameras=4):
        """Initialize a multi-camera viewer.
        
        Args:
            num_cameras (int): Number of camera feeds to display (1-9)
        """
        self.num_cameras = min(9, max(1, num_cameras))  # Limit to 1-9 cameras
        self.cameras_data = load_camera_data()
        if not self.cameras_data:
            print("No camera data found. Exiting.")
            sys.exit(1)
            
        self.cameras = []
        self.browser = None
        self.playwright = None
        
    async def initialize(self):
        """Initialize the browser and create the figure for multiple cameras."""
        # Start playwright and browser
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        # Select random cameras
        selected_cameras = {}
        camera_ids = list(self.cameras_data.keys())
        random.shuffle(camera_ids)
        
        for i in range(min(self.num_cameras, len(camera_ids))):
            camera_id = camera_ids[i]
            camera_name = self.cameras_data[camera_id]
            selected_cameras[camera_id] = camera_name
            
        # Create figure and subplots based on number of cameras
        plt.ion()  # Turn on interactive mode
        
        # Determine grid dimensions
        if self.num_cameras <= 3:
            rows, cols = 1, self.num_cameras
        elif self.num_cameras <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(cols*5, rows*4))
        gs = GridSpec(rows, cols, figure=self.fig)
        
        # Initialize camera streams
        self.camera_objects = []
        index = 0
        
        for camera_id, camera_name in selected_cameras.items():
            row = index // cols
            col = index % cols
            
            ax = self.fig.add_subplot(gs[row, col])
            print(f"Initializing camera {index+1}/{self.num_cameras}: {camera_name}")
            
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
                    'ax': ax
                })
                
            index += 1
            
        self.fig.tight_layout()
        return len(self.camera_objects) > 0
        
    async def run(self):
        """Run the multi-camera viewer."""
        if not await self.initialize():
            print("Failed to initialize camera feeds.")
            return
            
        print(f"\nStreaming {len(self.camera_objects)} camera feeds. Press Ctrl+C to stop.")
        
        try:
            while True:
                for camera in self.camera_objects:
                    success, _ = await update_camera_frame(camera['page'], camera['im_display'])
                    if not success:
                        print(f"Could not update frame for {camera['camera_name']}")
                
                plt.pause(0.5)  # Update display
                
        except (KeyboardInterrupt, SystemExit):
            print("\nStream stopped by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        
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
        plt.close(self.fig)
        plt.ioff()
        print("All resources released.")

async def main():
    """Main function to run the multi-camera viewer."""
    # Get number of cameras from command line or use default
    num_cameras = 4  # Default to 4 cameras
    if len(sys.argv) > 1:
        try:
            num_cameras = int(sys.argv[1])
        except ValueError:
            print("Invalid number of cameras. Using default (4).")
    
    viewer = MultiCameraViewer(num_cameras)
    await viewer.run()

if __name__ == "__main__":
    # On Windows, the default event loop policy may cause issues with Playwright.
    # Using a different policy can help.
    if asyncio.get_event_loop().is_running():
         print("Asyncio loop is already running.")
    
    # To run playwright in a script, it's best to use asyncio.run()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nScript interrupted by user.")
