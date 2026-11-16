import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

SCRAPE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRAPE_DIR.parent

# camera list behind the map at https://pbctrafficstreams.pbc.gov/portal/
# each <stream> has guid, title, lat, lng, and href (the .m3u8 playlist)
FEED_URL = "https://pbctrafficstreams.pbc.gov/portal/feeds/loader.php"

JSON_OUTPUTS = [SCRAPE_DIR / 'tc_pbc.json', ROOT_DIR / 'server' / 'tc_pbc.json']
# guid -> playlist URL; the path (vf1/vf2) varies per camera, so it can't be built from the guid
URL_OUTPUTS = [ROOT_DIR / 'server' / 'tc_pbc_urls.json']
CSV_OUTPUTS = [SCRAPE_DIR / 'pbc_coords.csv', ROOT_DIR / 'client' / 'public' / 'pbc_coords.csv']

response = requests.get(FEED_URL, timeout=30)
response.raise_for_status()

root = ET.fromstring(response.content)

# guid -> location name, plus coordinate rows for the map
camera_data = {}
camera_urls = {}
coord_rows = []

for stream in root.iter('stream'):
    guid = stream.get('guid')
    title = (stream.get('title') or '').strip()
    lat = stream.get('lat')
    lng = stream.get('lng')
    href = stream.get('href')

    if not guid or not title or not href or stream.get('state') != 'active':
        continue

    camera_data[guid] = title
    camera_urls[guid] = href
    if lat and lng:
        coord_rows.append((guid, title, lat, lng))

if not camera_data:
    raise Exception("No cameras found in feed.")

for path in JSON_OUTPUTS:
    with open(path, 'w') as f:
        json.dump(camera_data, f, indent=4)

for path in URL_OUTPUTS:
    with open(path, 'w') as f:
        json.dump(camera_urls, f, indent=4)

for path in CSV_OUTPUTS:
    with open(path, 'w', newline='') as f:
        f.write("UUID, Name, Latitude, Longitude\n")
        for guid, title, lat, lng in coord_rows:
            f.write(f"{guid}, {title}, {lat}, {lng}\n")

print(f"Wrote {len(camera_data)} cameras ({len(coord_rows)} with coordinates).")
