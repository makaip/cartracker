import sqlite3
import os

DB_PATH = 'vehicles.db'

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
    conn.commit()
    conn.close()

def add_vehicle(uuid: str):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO vehicles (uuid) VALUES (?)', (uuid,))
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
