"""
SmartAgro Score — SQLite Database Module
Модуль для работы с SQLite базой данных заявок
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
import os

# Путь к базе данных в папке data
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "data", "smartagro.db")


@contextmanager
def get_db_connection():
    """Контекстный менеджер для подключения к БД."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Инициализация базы данных — создание таблиц."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Таблица заявок
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id TEXT UNIQUE NOT NULL,
                company_name TEXT NOT NULL,
                bin_iin TEXT NOT NULL,
                region TEXT NOT NULL,
                subsidy_type TEXT NOT NULL,
                requested_amount REAL NOT NULL,
                application_date TEXT NOT NULL,
                akimat TEXT NOT NULL,
                direction TEXT NOT NULL,
                subsidy_name TEXT NOT NULL,
                normativ REAL NOT NULL,
                amount_due REAL NOT NULL,
                district TEXT NOT NULL,
                source_system TEXT DEFAULT 'manual',
                score REAL NOT NULL,
                score_category TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                shap_values TEXT NOT NULL,
                shap_explanation TEXT NOT NULL,
                calculated_at TEXT NOT NULL,
                model_version TEXT DEFAULT 'CatBoost-v1.0-production',
                decision TEXT,
                officer_name TEXT,
                decided_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Индексы для ускорения поиска
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_applications_score ON applications(score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_applications_category ON applications(score_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_applications_bin_iin ON applications(bin_iin)")
        
        conn.commit()


def create_application(data: dict) -> dict:
    """Создание новой заявки."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO applications (
                application_id, company_name, bin_iin, region, subsidy_type,
                requested_amount, application_date, akimat, direction, subsidy_name,
                normativ, amount_due, district, source_system, score, score_category,
                recommendation, shap_values, shap_explanation, calculated_at, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["application_id"],
            data["company_name"],
            data["bin_iin"],
            data["region"],
            data["subsidy_type"],
            data["requested_amount"],
            data["application_date"],
            data["akimat"],
            data["direction"],
            data["subsidy_name"],
            data["normativ"],
            data["amount_due"],
            data["district"],
            data.get("source_system", "manual"),
            data["score"],
            data["score_category"],
            data["recommendation"],
            json.dumps(data.get("shap_values", {})),
            json.dumps(data.get("shap_explanation", [])),
            data["calculated_at"],
            data.get("model_version", "CatBoost-v1.0-production")
        ))
        
        conn.commit()
        return get_application(data["application_id"])


def get_all_applications() -> list:
    """Получение всех заявок, отсортированных по баллу."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM applications ORDER BY score DESC
        """)
        
        applications = []
        for row in cursor.fetchall():
            app = dict(row)
            app["shap_values"] = json.loads(app["shap_values"])
            app["shap_explanation"] = json.loads(app["shap_explanation"])
            applications.append(app)
        
        return applications


def get_application(application_id: str) -> Optional[dict]:
    """Получение заявки по ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM applications WHERE application_id = ?", (application_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        app = dict(row)
        app["shap_values"] = json.loads(app["shap_values"])
        app["shap_explanation"] = json.loads(app["shap_explanation"])
        return app


def update_application_decision(application_id: str, decision: str, officer_name: str, decided_at: str) -> bool:
    """Обновление решения по заявке."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE applications 
            SET decision = ?, officer_name = ?, decided_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE application_id = ?
        """, (decision, officer_name, decided_at, application_id))
        
        conn.commit()
        return cursor.rowcount > 0


def get_application_count() -> int:
    """Получение количества заявок."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM applications")
        return cursor.fetchone()[0]


# Инициализация БД при импорте
init_db()
