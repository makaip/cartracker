import json
import random
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import uuid
from playwright.async_api import async_playwright
import asyncio
import time
import base64

async def stream_camera_video(camera_uuid, camera_name, figure=None, ax=None, update_interval=0.5, browser=None, close_browser=True):
    """Streams video from a camera by capturing frames continuously.
    
    Args:
        camera_uuid (str): The UUID of the camera to stream
        camera_name (str): The name of the camera
        figure (Figure, optional): A matplotlib figure to use for display. If None, a new one will be created.
        ax (Axes, optional): A matplotlib axes to use for display. If None, a new one will be created.
        update_interval (float, optional): Time between frame updates in seconds.
        browser (Browser, optional): An existing browser instance to use. If None, a new one will be created.
        close_browser (bool, optional): Whether to close the browser when done. Default is True.
        
    Returns:
        tuple: (figure, axes, im_display) objects used for displaying the video
    """
    browser_created = False
    page = None
    fig = figure
    
    try:
        if browser is None:
            browser_created = True
            p = await async_playwright().start()
            browser = await p.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            viewport={'width': 1280, 'height': 720},  # Reduced resolution for multiple cameras
            has_touch=True,
        )
        page = await context.new_page()
        url = f"https://discover.pbc.gov/SiteAssets/helpers/Traffic-Cams/trafficByID.htm?source={camera_uuid}"
        print(f"Navigating to: {url}")
        await page.goto(url, wait_until='networkidle')
        
        video_selector = "video.jw-video.jw-reset"
        await page.wait_for_selector(video_selector, timeout=20000)
        
        # Wait a moment for the video to start playing.
        await page.wait_for_timeout(1000)
        
        js_code = """
        async (videoSelector) => {
            const video = document.querySelector(videoSelector);
            if (!video) return null;

            if (video.readyState < 2) { // HAVE_CURRENT_DATA
                await new Promise(r => setTimeout(r, 100));
            }

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png');
        }
        """
        
        if fig is None and ax is None:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(8, 6))  # Reduced size for multiple cameras
        
        ax.set_title(f"Traffic Camera: {camera_name}\nCamera ID: {camera_uuid}")
        ax.axis('off')
        fig.tight_layout()
        
        im_display = None
        
        print(f"\nStarting video stream for {camera_name}... Press Ctrl+C in the console to stop.")
        
        # Process one frame and return the display objects
        image_data_url = await page.evaluate(js_code, video_selector)
        if image_data_url:
            image_data = base64.b64decode(image_data_url.split(',')[1])
            image = Image.open(BytesIO(image_data))
            
            if im_display is None:
                im_display = ax.imshow(image)
            else:
                im_display.set_data(image)
                
            plt.pause(0.1)  # Just to update the display
            
            return fig, ax, im_display, page, browser, image
        else:
            print("Could not capture initial frame.")
            return fig, ax, None, page, browser, None

    except Exception as e:
        print(f"An error occurred during streaming setup: {e}")
        if page and not page.is_closed():
            await page.close()
        if browser and browser_created and close_browser:
            await browser.close()
        return fig, ax, None, None, None, None

async def update_camera_frame(page, im_display, js_code="", video_selector="video.jw-video.jw-reset"):
    """Updates a single frame from the camera."""
    if js_code == "":
        js_code = """
        async (videoSelector) => {
            const video = document.querySelector(videoSelector);
            if (!video) return null;

            if (video.readyState < 2) { // HAVE_CURRENT_DATA
                await new Promise(r => setTimeout(r, 100));
            }

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png');
        }
        """
    
    try:
        image_data_url = await page.evaluate(js_code, video_selector)
        if image_data_url:
            image_data = base64.b64decode(image_data_url.split(',')[1])
            image = Image.open(BytesIO(image_data))
            
            if im_display is not None:
                im_display.set_data(image)
                return True, image
        return False, None
    except Exception as e:
        print(f"Error updating frame: {e}")
        return False, None

def load_camera_data(filename='traffic_cameras.json'):
    """Loads camera data from a JSON file.
    
    Args:
        filename (str): Path to the JSON file containing camera data
        
    Returns:
        dict: Dictionary mapping camera UUIDs to camera names
    """
    try:
        with open(filename, 'r') as f:
            cameras_data = json.load(f)
        return cameras_data
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return {}

def select_random_camera(cameras_data):
    """Selects a random camera from the camera data.
    
    Args:
        cameras_data (dict): Dictionary mapping camera UUIDs to camera names
        
    Returns:
        tuple: (camera_id, camera_name) of the selected camera
    """
    if not cameras_data:
        return None, None
        
    random_camera_id = random.choice(list(cameras_data.keys()))
    random_camera_name = cameras_data[random_camera_id]
    
    return random_camera_id, random_camera_name

async def stream_single_camera():
    """Stream video from a single randomly selected camera."""
    cameras_data = load_camera_data()
    if not cameras_data:
        return
        
    camera_id, camera_name = select_random_camera(cameras_data)
    
    print(f"Selected Camera: {camera_name}")
    print(f"Camera ID (UUID): {camera_id}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig, ax, im_display, page, browser, _ = await stream_camera_video(camera_id, camera_name, fig, ax)
    
    try:
        while True:
            success, _ = await update_camera_frame(page, im_display)
            if not success:
                print("Could not capture frame.")
            plt.pause(0.5)
    except (KeyboardInterrupt, SystemExit):
        print("\nStream stopped by user.")
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
    finally:
        plt.close(fig)
        plt.ioff()
        if page and not page.is_closed():
            await page.close()
        if browser:
            await browser.close()
        print("Browser closed and resources released.")

async def main():
    """Main function to select a camera and start the video stream."""
    await stream_single_camera()

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

