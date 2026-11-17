import time
import uuid as uuid_lib
from pathlib import Path

import yaml

import database

SERVER_DIR = Path(__file__).resolve().parent
SIGHTINGS_DIR = SERVER_DIR / 'sightings'

with open(SERVER_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)['server']

# a vehicle that keeps matching on the same camera is one sighting; a new one
# starts only after it has gone unseen there for this many seconds
COOLDOWN = float(config.get('sighting_cooldown', 60))

# (vehicle_uuid, camera_uuid) -> {'id', 'image', 'score', 'last'} of the ongoing sighting
_ongoing: dict[tuple[str, str], dict] = {}


def image_path(image: str) -> Path:
    return SIGHTINGS_DIR / image


def record(camera_uuid: str, camera_name: str | None, hits: list[dict]) -> None:
    """store matches a GPU worker reported for one frame: [{vehicle_uuid, score, jpeg}]"""
    SIGHTINGS_DIR.mkdir(exist_ok=True)
    now = time.time()

    for hit in hits:
        key = (hit['vehicle_uuid'], camera_uuid)
        ongoing = _ongoing.get(key)

        if ongoing and now - ongoing['last'] < COOLDOWN:
            ongoing['last'] = now
            if hit['score'] <= ongoing['score']:
                continue
            # same pass, better look at the vehicle: keep the better picture
            if database.update_sighting_score(ongoing['id'], hit['score']):
                image_path(ongoing['image']).write_bytes(hit['jpeg'])
                ongoing['score'] = hit['score']
                continue
            # the sighting was deleted meanwhile; fall through and start a new one

        image = f'{uuid_lib.uuid4().hex}.jpg'
        image_path(image).write_bytes(hit['jpeg'])
        sighting_id = database.add_sighting(
            hit['vehicle_uuid'], camera_uuid, camera_name, hit['score'], now, image
        )
        _ongoing[key] = {'id': sighting_id, 'image': image, 'score': hit['score'], 'last': now}


def remove_images(images: list[str]) -> None:
    for image in images:
        image_path(image).unlink(missing_ok=True)
