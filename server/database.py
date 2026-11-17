import os
import sqlite3
from contextlib import closing
from pathlib import Path

SERVER_DIR = Path(__file__).parent
DB_PATH = SERVER_DIR / 'vehicles.db'
# Sightings get their own file: every write to vehicles.db makes all GPU workers
# reload their target embeddings (see vehicle_db_watcher in main.py).
SIGHTINGS_DB_PATH = SERVER_DIR / 'sightings.db'

DEFAULT_THRESHOLD = 0.5


def _connect(path: Path) -> sqlite3.Connection:
    # several nodes share these files; wait for another writer instead of failing
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_db_connection():
    return _connect(DB_PATH)


def init_db():
    with closing(_connect(DB_PATH)) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL
            )
        ''')

        cols = [r['name'] for r in conn.execute("PRAGMA table_info(vehicles)").fetchall()]
        new_cols = {
            'name': 'TEXT',
            # tracking: record a sighting whenever a match scores at least `threshold`
            'tracking': 'INTEGER NOT NULL DEFAULT 1',
            'threshold': f'REAL NOT NULL DEFAULT {DEFAULT_THRESHOLD}',
        }
        for col, decl in new_cols.items():
            if col in cols:
                continue
            try:
                conn.execute(f'ALTER TABLE vehicles ADD COLUMN {col} {decl}')
            except sqlite3.OperationalError:
                pass  # another node added it first
        conn.commit()

    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_uuid TEXT NOT NULL,
                camera_uuid TEXT NOT NULL,
                camera_name TEXT,
                score REAL NOT NULL,
                seen_at REAL NOT NULL,
                image TEXT NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS sightings_by_vehicle ON sightings (vehicle_uuid, seen_at)')
        conn.commit()


# ---- vehicles

def add_vehicle(uuid: str, name: str | None = None):
    with closing(_connect(DB_PATH)) as conn:
        try:
            conn.execute('INSERT INTO vehicles (uuid, name) VALUES (?, ?)', (uuid, name))
            conn.commit()
        except sqlite3.IntegrityError:
            pass


def delete_vehicle(uuid: str):
    with closing(_connect(DB_PATH)) as conn:
        conn.execute('DELETE FROM vehicles WHERE uuid = ?', (uuid,))
        conn.commit()


def get_vehicles():
    with closing(_connect(DB_PATH)) as conn:
        vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    return [dict(v) for v in vehicles]


def get_vehicle(uuid: str):
    with closing(_connect(DB_PATH)) as conn:
        row = conn.execute('SELECT * FROM vehicles WHERE uuid = ?', (uuid,)).fetchone()
    return dict(row) if row else None


def update_vehicle(uuid: str, **fields):
    """set any of name / tracking / threshold"""
    fields = {k: v for k, v in fields.items() if k in ('name', 'tracking', 'threshold')}
    if not fields:
        return
    assignments = ', '.join(f'{k} = ?' for k in fields)
    with closing(_connect(DB_PATH)) as conn:
        conn.execute(f'UPDATE vehicles SET {assignments} WHERE uuid = ?', (*fields.values(), uuid))
        conn.commit()


def touch_vehicles():
    """mark vehicles.db as changed (reference images are files, not rows) so every node reloads"""
    os.utime(DB_PATH)


# ---- sightings

def add_sighting(vehicle_uuid: str, camera_uuid: str, camera_name: str | None,
                 score: float, seen_at: float, image: str) -> int:
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        cur = conn.execute(
            'INSERT INTO sightings (vehicle_uuid, camera_uuid, camera_name, score, seen_at, image) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (vehicle_uuid, camera_uuid, camera_name, score, seen_at, image)
        )
        conn.commit()
        return cur.lastrowid


def update_sighting_score(sighting_id: int, score: float) -> bool:
    """False if the sighting no longer exists"""
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        cur = conn.execute('UPDATE sightings SET score = ? WHERE id = ?', (score, sighting_id))
        conn.commit()
        return cur.rowcount > 0


def get_sightings(vehicle_uuid: str | None = None, limit: int = 100, offset: int = 0, min_score: float = 0.0):
    """newest first; total counts every sighting that passes the filters, not just this page"""
    where, params = 'WHERE score >= ?', (min_score,)
    if vehicle_uuid:
        where, params = where + ' AND vehicle_uuid = ?', (*params, vehicle_uuid)
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        total = conn.execute(f'SELECT COUNT(*) FROM sightings {where}', params).fetchone()[0]
        rows = conn.execute(
            f'SELECT * FROM sightings {where} ORDER BY seen_at DESC, id DESC LIMIT ? OFFSET ?',
            (*params, limit, offset)
        ).fetchall()
    return total, [dict(r) for r in rows]


def get_sighting(sighting_id: int):
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        row = conn.execute('SELECT * FROM sightings WHERE id = ?', (sighting_id,)).fetchone()
    return dict(row) if row else None


def sighting_stats() -> dict[str, dict]:
    """vehicle uuid -> {count, last_seen}"""
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        rows = conn.execute(
            'SELECT vehicle_uuid, COUNT(*) AS count, MAX(seen_at) AS last_seen '
            'FROM sightings GROUP BY vehicle_uuid'
        ).fetchall()
    return {r['vehicle_uuid']: {'count': r['count'], 'last_seen': r['last_seen']} for r in rows}


def delete_sighting(sighting_id: int) -> str | None:
    """returns the image file name of the deleted sighting"""
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        row = conn.execute('SELECT image FROM sightings WHERE id = ?', (sighting_id,)).fetchone()
        if row is None:
            return None
        conn.execute('DELETE FROM sightings WHERE id = ?', (sighting_id,))
        conn.commit()
        return row['image']


def delete_sightings(vehicle_uuid: str) -> list[str]:
    """returns the image file names of the deleted sightings"""
    with closing(_connect(SIGHTINGS_DB_PATH)) as conn:
        rows = conn.execute('SELECT image FROM sightings WHERE vehicle_uuid = ?', (vehicle_uuid,)).fetchall()
        conn.execute('DELETE FROM sightings WHERE vehicle_uuid = ?', (vehicle_uuid,))
        conn.commit()
    return [r['image'] for r in rows]
