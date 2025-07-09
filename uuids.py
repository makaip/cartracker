import requests
from bs4 import BeautifulSoup
import json
import re

# Target page URL
url = "https://discover.pbc.gov/iss/Pages/LIVE-Traffic-Cams.aspx#"

# Send GET request
response = requests.get(url)
response.raise_for_status()  # Raise if there’s an error

# Parse HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Find the div by webpartid
target_div = soup.find('div', attrs={'webpartid': '4163b065-726c-4dba-9683-87907cce1070'})
if not target_div:
    raise Exception("Target div not found.")

# Extract all anchor tags within the div
links = target_div.find_all('a')

# Dictionary to store UUID -> Location name
camera_data = {}

# Regex to extract UUID from href
uuid_regex = re.compile(r'source=([a-f0-9\-]+)')

for link in links:
    href = link.get('href')
    text = link.get_text(strip=True)

    if href and text:
        match = uuid_regex.search(href)
        if match:
            uuid = match.group(1)
            camera_data[uuid] = text

# Save to JSON
with open('traffic_cameras.json', 'w') as f:
    json.dump(camera_data, f, indent=4)

print(f"Extracted {len(camera_data)} cameras and saved to 'traffic_cameras.json'")
