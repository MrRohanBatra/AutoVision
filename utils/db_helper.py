import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), "database", "autovision.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE if not exists users (
        chat_id INTEGER PRIMARY KEY,
        username TEXT,
        registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE if not exists cars (
        plate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        plate_number TEXT NOT NULL,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chat_id) REFERENCES users(chat_id) ON DELETE CASCADE
    );

    CREATE TABLE if not exists detections (
        detect_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT NOT NULL,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        processed INTEGER DEFAULT 0,
        location TEXT DEFAULT 'unknown',
        image_path TEXT
    );
    """)
    conn.commit()
    conn.close()

def add_detection(plate_number, location="unknown", image_path=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (plate_number, location, image_path, processed) VALUES (?, ?, ?, 0)",
        (plate_number, location, image_path)
    )
    conn.commit()
    conn.close()

def get_user_chat_ids_for_plate(plate_number):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT u.chat_id 
        FROM users u
        JOIN cars c ON u.chat_id=c.chat_id
        WHERE c.plate_number=?
    """, (plate_number,))
    chat_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return chat_ids
