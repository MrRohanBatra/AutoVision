import os
import sqlite3
DB_PATH = os.path.join(os.getcwd(), "database", "autovision.db")
# -------------------------
# Database helpers
# -------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # users table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS users
                (
                    chat_id
                    INTEGER
                    PRIMARY
                    KEY,
                    username
                    TEXT,
                    registered_at
                    DATETIME
                    DEFAULT
                    CURRENT_TIMESTAMP
                )
                """)

    # cars table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS cars
                (
                    plate_id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    chat_id
                    INTEGER,
                    plate_number
                    TEXT
                    NOT
                    NULL,
                    added_at
                    DATETIME
                    DEFAULT
                    CURRENT_TIMESTAMP,
                    FOREIGN
                    KEY
                (
                    chat_id
                ) REFERENCES users
                (
                    chat_id
                ) ON DELETE CASCADE
                    )
                """)

    conn.commit()
    conn.close()


def add_user(chat_id: int, username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO users (chat_id, username) VALUES (?, ?)",
        (chat_id, username)
    )
    conn.commit()
    conn.close()


def add_car(chat_id: int, plate_number: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO cars (chat_id, plate_number) VALUES (?, ?)",
        (chat_id, plate_number.upper())
    )
    conn.commit()
    conn.close()

