import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent

url = "https://discover.pbc.gov/iss/Pages/LIVE-Traffic-Cams.aspx#"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

# div by webpartid
target_div = soup.find('div', attrs={'id': 'WebPartWPQ5'})
if not target_div:
    raise Exception("Target div not found.")

links = target_div.find_all('a')

# dict to store UUID -> location name; extract UUID from href
camera_data = {}
uuid_regex = re.compile(r'source=([a-f0-9\-]+)')

for link in links:
    href = link.get('href')
    text = link.get_text(strip=True)

    if href and text:
        match = uuid_regex.search(href)
        if match:
            uuid = match.group(1)
            camera_data[uuid] = text

# to json
with open(SERVER_DIR / 'traffic_cameras.json', 'w') as f:
    json.dump(camera_data, f, indent=4)