
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "applications.sqlite"


def _existing_column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_type: str
) -> None:
    if column in _existing_column_names(conn, table):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


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
        _add_column_if_missing(conn, "applications", "score_zone", "TEXT")
        _add_column_if_missing(conn, "applications", "final_score", "REAL")
        _add_column_if_missing(conn, "applications", "is_verified", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "applications", "verified_payload", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_zone ON applications(score_zone)"
        )
        conn.commit()
    finally:
        conn.close()


def _serialize_verified_payload(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def upsert_application(record: dict) -> None:
    if record.get("is_demo"):
        return
    aid = record.get("application_id")
    if not aid:
        return
    aid = str(aid)
    init_db()

    payload = json.dumps(record, ensure_ascii=False, default=str)
    now = datetime.now().isoformat()

    score_zone = record.get("score_zone") or record.get("zone")
    if score_zone is not None:
        score_zone = str(score_zone)

    final_score = record.get("final_score")
    if final_score is None and record.get("score") is not None:
        try:
            final_score = float(record["score"])
        except (TypeError, ValueError):
            final_score = None

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT is_verified, verified_payload FROM applications WHERE application_id = ?",
            (aid,),
        )
        existing = cur.fetchone()

        if "is_verified" in record and record["is_verified"] is not None:
            try:
                is_verified = 1 if int(record["is_verified"]) else 0
            except (TypeError, ValueError):
                is_verified = 1 if record["is_verified"] else 0
        elif existing is not None:
            is_verified = int(existing[0] if existing[0] is not None else 0)
        else:
            is_verified = 0

        verified_payload_out: Optional[str]
        if "verified_payload" in record:
            verified_payload_out = _serialize_verified_payload(record["verified_payload"])
        elif existing is not None:
            verified_payload_out = existing[1]
        else:
            verified_payload_out = None

        conn.execute(
            """
            INSERT OR REPLACE INTO applications (
                application_id, payload, updated_at,
                score_zone, final_score, is_verified, verified_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                aid,
                payload,
                now,
                score_zone,
                final_score,
                is_verified,
                verified_payload_out,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _merge_row_into_dict(
    payload_json: str,
    score_zone: Optional[str],
    final_score: Optional[float],
    is_verified: Optional[int],
    verified_payload: Optional[str],
) -> dict:
    d = json.loads(payload_json)
    if score_zone is not None:
        d["score_zone"] = score_zone
    if final_score is not None:
        d["final_score"] = final_score
    if is_verified is not None:
        d["is_verified"] = int(is_verified)
    if verified_payload:
        try:
            d["verified_payload"] = json.loads(verified_payload)
        except json.JSONDecodeError:
            d["verified_payload"] = verified_payload
    return d


def load_all_applications() -> list[dict]:
    if not DB_PATH.exists():
        return []
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            """
            SELECT payload, score_zone, final_score, is_verified, verified_payload
            FROM applications
            ORDER BY updated_at ASC
            """
        )
        out = []
        for row in cur.fetchall():
            out.append(
                _merge_row_into_dict(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                )
            )
        return out
    finally:
        conn.close()


def clear_all_applications() -> int:
    """Удаляет все строки из таблицы заявок. Перезапустите Uvicorn, чтобы очистить кэш в памяти (_applications_db)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM applications")
        n = int(cur.fetchone()[0])
        conn.execute("DELETE FROM applications")
        conn.commit()
        return n
    finally:
        conn.close()


def get_training_data(zone: Optional[str] = None) -> list[dict]:
    """
    Заявки для обучения: только с is_verified = 1.
    Если задан zone — дополнительно score_zone = zone ('green' | 'yellow' | 'red').
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        if zone is not None:
            cur = conn.execute(
                """
                SELECT payload, score_zone, final_score, is_verified, verified_payload
                FROM applications
                WHERE is_verified = 1 AND score_zone = ?
                ORDER BY updated_at ASC
                """,
                (str(zone),),
            )
        else:
            cur = conn.execute(
                """
                SELECT payload, score_zone, final_score, is_verified, verified_payload
                FROM applications
                WHERE is_verified = 1
                ORDER BY updated_at ASC
                """
            )
        return [
            _merge_row_into_dict(row[0], row[1], row[2], row[3], row[4])
            for row in cur.fetchall()
        ]
    finally:
        conn.close()
