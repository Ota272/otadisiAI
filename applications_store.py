"""
SQLite-хранилище заявок: только «реальные» записи (не демо с кнопки тестовых заявок).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "applications.sqlite"


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                application_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_application(record: dict) -> None:
    """Сохраняет или обновляет заявку. Демо-заявки (is_demo=True) не пишутся."""
    if record.get("is_demo"):
        return
    aid = record.get("application_id")
    if not aid:
        return
    payload = json.dumps(record, ensure_ascii=False, default=str)
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO applications (application_id, payload, updated_at)
            VALUES (?, ?, ?)
            """,
            (str(aid), payload, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_all_applications() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT payload FROM applications ORDER BY updated_at ASC"
        )
        return [json.loads(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
