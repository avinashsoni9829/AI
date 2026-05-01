import sqlite3
from pathlib import Path
from datetime import datetime



BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "rent_posts.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rental_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            raw_text_anonymized TEXT,
            city TEXT,
            area TEXT,
            bhk INTEGER,
            rent INTEGER,
            deposit INTEGER,
            furnishing TEXT,
            property_type TEXT,
            gender_preference TEXT,
            available_from TEXT,
            confidence REAL,
            status TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_post(parsed: dict):
    init_db()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO rental_posts (
            source, raw_text_anonymized, city, area, bhk, rent, deposit,
            furnishing, property_type, gender_preference, available_from,
            confidence, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        parsed.get("source"),
        parsed.get("raw_text_anonymized"),
        parsed.get("city"),
        parsed.get("area"),
        parsed.get("bhk"),
        parsed.get("rent"),
        parsed.get("deposit"),
        parsed.get("furnishing"),
        parsed.get("property_type"),
        parsed.get("gender_preference"),
        parsed.get("available_from"),
        parsed.get("confidence"),
        parsed.get("status"),
        parsed.get("created_at") or datetime.now().isoformat()
    ))

    conn.commit()
    post_id = cur.lastrowid
    conn.close()

    return post_id


def fetch_posts(limit: int = 20, status: str | None = None):
    init_db()

    conn = get_connection()
    cur = conn.cursor()

    if status:
        cur.execute("""
            SELECT * FROM rental_posts
            WHERE status = ?
            ORDER BY id DESC
            LIMIT ?
        """, (status, limit))
    else:
        cur.execute("""
            SELECT * FROM rental_posts
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    return rows