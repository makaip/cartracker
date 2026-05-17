import io
import time
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ID = "091-086_0-sb-ipv"
URL = (f"https://www.tpktraffic.com/CreateImage.ashx?resolution=1280x720&ip_address={ID}&dns={ID}")

plt.ion()  # interactive mode

fig, ax = plt.subplots()
image_display = None

while True:
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()

        image = mpimg.imread(io.BytesIO(response.content), format='jpg')

        if image_display is None:
            image_display = ax.imshow(image)
            ax.axis("off")
        else:
            image_display.set_data(image)

        fig.canvas.draw()
        fig.canvas.flush_events()

    except Exception as e:
        print(f"Error updating image: {e}")

    time.sleep(1)