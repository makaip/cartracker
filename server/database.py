import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent / 'vehicles.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()

    conn.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE NOT NULL
        )
    ''')

    cur = conn.execute("PRAGMA table_info(vehicles)")
    cols = [r['name'] for r in cur.fetchall()]
    if 'name' not in cols:
        conn.execute('ALTER TABLE vehicles ADD COLUMN name TEXT')

    conn.commit()
    conn.close()

def add_vehicle(uuid: str, name: str | None = None):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO vehicles (uuid, name) VALUES (?, ?)', (uuid, name))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def delete_vehicle(uuid: str):
    conn = get_db_connection()
    conn.execute('DELETE FROM vehicles WHERE uuid = ?', (uuid,))
    conn.commit()
    conn.close()

def get_vehicles():
    conn = get_db_connection()
    vehicles = conn.execute('SELECT * FROM vehicles').fetchall()
    conn.close()
    return [dict(v) for v in vehicles]
