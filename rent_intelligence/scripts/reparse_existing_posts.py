import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rent_intelligence.parser.post_parser import parse_rental_post
from rent_intelligence.db.database import DB_PATH, init_db


def main():
    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, raw_text_anonymized FROM rental_posts")
    rows = cur.fetchall()

    for row in rows:
        parsed = parse_rental_post(row["raw_text_anonymized"])

        cur.execute("""
            UPDATE rental_posts
            SET
                area = ?,
                bhk = ?,
                rent = ?,
                deposit = ?,
                furnishing = ?,
                property_type = ?,
                listing_type = ?,
                gender_preference = ?,
                available_from = ?,
                confidence = ?,
                status = ?
            WHERE id = ?
        """, (
            parsed.get("area"),
            parsed.get("bhk"),
            parsed.get("rent"),
            parsed.get("deposit"),
            parsed.get("furnishing"),
            parsed.get("property_type"),
            parsed.get("listing_type"),
            parsed.get("gender_preference"),
            parsed.get("available_from"),
            parsed.get("confidence"),
            parsed.get("status"),
            row["id"],
        ))

    conn.commit()
    conn.close()

    print("Existing posts reparsed ✅")


if __name__ == "__main__":
    main()
