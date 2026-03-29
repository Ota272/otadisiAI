"""
SmartAgro Score — FastAPI Scoring Engine
Хакатон Decentrathon 5.0 | AI for Government
Министерство сельского хозяйства РК
"""

import uuid
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import random
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier, Pool
import shap

# ─────────────────────────────────────────────
# Инициализация приложения
# ─────────────────────────────────────────────
app = FastAPI(
    title="SmartAgro Score API",
    description="Система merit-based скоринга субсидий для Министерства сельского хозяйства РК",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Хранилище данных (in-memory для MVP)
# ─────────────────────────────────────────────
API_KEYS: dict[str, dict] = {
    "sk-msgov-2025-demo-key-abc123": {
        "owner": "МСХ РК — Отдел субсидирования",
        "created": "2025-01-01",
    }
}

APPLICATIONS_DB: list[dict] = []
DECISIONS_DB: list[dict] = []

# ─────────────────────────────────────────────
# Pydantic схемы данных
# ─────────────────────────────────────────────

class ApplicationFeatures(BaseModel):
    """Признаки для расчёта скорингового балла."""

    # Основные идентификаторы
    bin_iin: str = Field(..., description="БИН/ИИН предприятия")
    company_name: str = Field(..., description="Наименование предприятия")
    region: str = Field(..., description="Область РК")
    subsidy_type: str = Field(..., description="Вид субсидии")
    requested_amount: float = Field(..., description="Запрашиваемая сумма (тенге)")

    # Фичи production модели (8 признаков)
    application_date: str = Field(..., description="Дата поступления (формат: DD.MM.YYYY HH:MM:SS)")
    akimat: str = Field(..., description="Акимат")
    direction: str = Field(..., description="Направление водства")
    subsidy_name: str = Field(..., description="Наименование субсидирования")
    normativ: float = Field(..., description="Норматив")
    amount_due: float = Field(..., description="Причитающая сумма")
    district: str = Field(..., description="Район хозяйства")

    # Метаданные (не идут в модель)
    source_system: Optional[str] = Field(
        default="manual",
        description="Источник: manual | giss | egov"
    )


class ScoreResponse(BaseModel):
    """Ответ Scoring Engine."""
    application_id: str
    company_name: str
    bin_iin: str
    region: str
    subsidy_type: str
    requested_amount: float
    score: float = Field(..., ge=0, le=100, description="Скоринговый балл (0–100)")
    score_category: str = Field(..., description="green | yellow | red")
    recommendation: str
    shap_values: dict[str, float]
    shap_explanation: list[dict]
    calculated_at: str
    model_version: str = "CatBoost-v1.0-production"


class DecisionRequest(BaseModel):
    """Решение комиссии по заявке."""
    application_id: str
    decision: str = Field(..., description="approved | rejected")
    comment: Optional[str] = None
    officer_name: str


class DecisionResponse(BaseModel):
    application_id: str
    decision: str
    officer_name: str
    decided_at: str
    comment: Optional[str]


class ApiKeyResponse(BaseModel):
    api_key: str
    owner: str
    created: str
    permissions: list[str]


# ─────────────────────────────────────────────
# Загрузка production модели CatBoost
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "Дата поступления",
    "Область",
    "Акимат",
    "Направление водства",
    "Наименование субсидирования",
    "Норматив",
    "Причитающая сумма",
    "Район хозяйства",
]

CAT_FEATURES = ["Дата поступления", "Область", "Акимат", "Направление водства",
                "Наименование субсидирования", "Район хозяйства"]
NUM_FEATURES = ["Норматив", "Причитающая сумма"]

# Пути к модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), "production_model.cbm")

def _load_model() -> tuple:
    """Загружает production модель и SHAP explainer."""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return model, explainer

_MODEL, _EXPLAINER = _load_model()


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _normalize_feature(name: str, value: float) -> float:
    """Нормализует признак в диапазон [0, 1]."""
    lo, hi = FEATURE_BOUNDS[name]
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))


def _compute_score(features: ApplicationFeatures) -> tuple[float, dict, list]:
    """
    Вычисляет скоринговый балл через production CatBoost модель.
    Score = вероятность одобрения × 100

    Returns:
        (score, shap_values_dict, shap_explanation_list)
    """
    # Подготовка данных для модели
    input_data = pd.DataFrame([{
        "Дата поступления": features.application_date,
        "Область": features.region,
        "Акимат": features.akimat,
        "Направление водства": features.direction,
        "Наименование субсидирования": features.subsidy_name,
        "Норматив": features.normativ,
        "Причитающая сумма": features.amount_due,
        "Район хозяйства": features.district,
    }])

    # Предикт вероятности (класс 1 = одобрено)
    proba = _MODEL.predict_proba(input_data)[0][1]
    score = float(np.clip(proba * 100, 0, 100))

    # SHAP-объяснения
    shap_vals = _EXPLAINER.shap_values(input_data)[0]
    shap_dict = {name: float(shap_vals[i]) for i, name in enumerate(FEATURE_NAMES)}

    # Человекочитаемые объяснения
    LABELS = {
        "Дата поступления":           "Дата подачи заявки",
        "Область":                    "Область",
        "Акимат":                     "Акимат",
        "Направление водства":        "Направление водства",
        "Наименование субсидирования":"Тип субсидии",
        "Норматив":                   "Норматив",
        "Причитающая сумма":          "Сумма к выплате",
        "Район хозяйства":            "Район",
    }

    raw_vals = [
        features.application_date,
        features.region,
        features.akimat,
        features.direction,
        features.subsidy_name,
        features.normativ,
        features.amount_due,
        features.district,
    ]

    explanation = []
    for name, shap_val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        raw_val = raw_vals[FEATURE_NAMES.index(name)]
        direction = "positive" if shap_val > 0 else "negative"
        sign = "+" if shap_val > 0 else ""
        explanation.append({
            "feature": name,
            "label": LABELS[name],
            "raw_value": str(raw_val) if isinstance(raw_val, str) else round(raw_val, 2),
            "shap_value": round(shap_val, 2),
            "direction": direction,
            "text": f"{sign}{shap_val:.1f}: {LABELS[name]} = {raw_val}",
        })

    return score, shap_dict, explanation


def _get_category(score: float) -> tuple[str, str]:
    """Возвращает (category_key, recommendation_text)."""
    if score >= 80:
        return "green", "Строго рекомендовано к одобрению"
    elif score >= 50:
        return "yellow", "Требуется рассмотрение комиссии"
    else:
        return "red", "Не рекомендовано (выявлены риски)"


def _verify_api_key(x_api_key: str = Header(...)) -> str:
    """Middleware для проверки API ключа."""
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Неверный API ключ")
    return x_api_key


# ─────────────────────────────────────────────
# Маршруты API
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Корневой маршрут — информация о системе."""
    return {
        "system": "SmartAgro Score",
        "version": "1.0.0",
        "description": "Merit-based scoring для субсидий МСХ РК",
        "docs": "/docs",
    }


@app.get("/health", tags=["Info"])
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_application(
    features: ApplicationFeatures,
    api_key: str = Depends(_verify_api_key),
):
    """
    Рассчитывает скоринговый балл для заявки.

    Принимает данные фермера из ГИСС/eGov, возвращает:
    - Балл (0–100)
    - Категорию (green/yellow/red)
    - SHAP-объяснение
    """
    app_id = str(uuid.uuid4())[:8].upper()
    score, shap_dict, explanation = _compute_score(features)
    category, recommendation = _get_category(score)

    result = {
        "application_id": app_id,
        "company_name": features.company_name,
        "bin_iin": features.bin_iin,
        "region": features.region,
        "subsidy_type": features.subsidy_type,
        "requested_amount": features.requested_amount,
        "score": round(score, 2),
        "score_category": category,
        "recommendation": recommendation,
        "shap_values": shap_dict,
        "shap_explanation": explanation,
        "calculated_at": datetime.now().isoformat(),
        "model_version": "CatBoost-v1.0-production",
        "application_date": features.application_date,
        "source_system": features.source_system or "manual",
    }

    APPLICATIONS_DB.append(result)
    return result


@app.get("/api/v1/applications", tags=["Applications"])
def get_all_applications(api_key: str = Depends(_verify_api_key)):
    """Возвращает все заявки, отсортированные по убыванию балла."""
    return sorted(APPLICATIONS_DB, key=lambda x: x["score"], reverse=True)


@app.get("/api/v1/applications/{application_id}", tags=["Applications"])
def get_application(application_id: str, api_key: str = Depends(_verify_api_key)):
    """Возвращает детали конкретной заявки."""
    for app in APPLICATIONS_DB:
        if app["application_id"] == application_id:
            return app
    raise HTTPException(status_code=404, detail="Заявка не найдена")


@app.post("/api/v1/giss/sync", tags=["GISS Integration"])
def sync_from_giss(api_key: str = Depends(_verify_api_key)):
    """
    Имитирует синхронизацию заявок из ГИСС.
    В продакшене: подключение к реальному API ГИСС.
    """
    REGIONS = [
        "Алматинская", "Акмолинская", "Атырауская", "Восточно-Казахстанская",
        "Жамбылская", "Карагандинская", "Костанайская", "Кызылординская",
        "Мангистауская", "Павлодарская", "Северо-Казахстанская", "Туркестанская",
        "Западно-Казахстанская", "Актюбинская",
    ]
    AKIMATS = ["Акимат г. Алматы", "Акимат г. Астаны", "Акимат Шыркент", "Акимат Тараз"]
    DIRECTIONS = ["Мясное", "Молочное", "Овцеводство", "Птицеводство"]
    SUBSIDY_NAMES = [
        "Субсидирование племенного КРС",
        "Субсидирование молочного стада",
        "Субсидирование овцеводства",
    ]
    DISTRICTS = ["Алматинский район", "Шуский район", "Талгарский район", "Карасайский район"]
    COMPANIES = [
        "ТОО «Алтай-Агро»", "ИП Сейткали А.Б.", "ТОО «СтепьПром»",
        "КХ «Береке»", "ТОО «АгроЗерно»", "ИП Жаксыбеков К.М.",
        "ТОО «КазАгро Холдинг»", "КФХ «Нур»",
    ]

    synced = []
    for _ in range(random.randint(3, 6)):
        app_date = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        normativ = random.uniform(500000, 5000000)
        amount = random.uniform(2_000_000, 50_000_000)

        features = ApplicationFeatures(
            bin_iin=f"{random.randint(100000000000, 999999999999)}",
            company_name=random.choice(COMPANIES),
            region=random.choice(REGIONS),
            subsidy_type=random.choice(SUBSIDY_NAMES),
            requested_amount=amount,
            application_date=app_date.strftime("%d.%m.%Y %H:%M:%S"),
            akimat=random.choice(AKIMATS),
            direction=random.choice(DIRECTIONS),
            subsidy_name=random.choice(SUBSIDY_NAMES),
            normativ=normativ,
            amount_due=amount,
            district=random.choice(DISTRICTS),
            source_system="giss",
        )
        result = score_application(features, api_key)
        synced.append(result)

    return {
        "synced_count": len(synced),
        "source": "ГИСС (mock)",
        "synced_at": datetime.now().isoformat(),
        "applications": synced,
    }


@app.post("/api/v1/decision", response_model=DecisionResponse, tags=["Decisions"])
def record_decision(
    decision_req: DecisionRequest,
    api_key: str = Depends(_verify_api_key),
):
    """Фиксирует решение комиссии (Human-in-the-loop)."""
    if decision_req.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision должен быть approved или rejected")

    record = {
        "application_id": decision_req.application_id,
        "decision": decision_req.decision,
        "officer_name": decision_req.officer_name,
        "decided_at": datetime.now().isoformat(),
        "comment": decision_req.comment,
    }
    DECISIONS_DB.append(record)

    # Обновляем статус в базе заявок
    for app in APPLICATIONS_DB:
        if app["application_id"] == decision_req.application_id:
            app["decision"] = decision_req.decision
            app["officer_name"] = decision_req.officer_name
            app["decided_at"] = record["decided_at"]
            break

    return record


@app.get("/api/v1/decisions", tags=["Decisions"])
def get_decisions(api_key: str = Depends(_verify_api_key)):
    """Возвращает историю решений комиссии."""
    return DECISIONS_DB


@app.post("/api/v1/keys/generate", response_model=ApiKeyResponse, tags=["API Keys"])
def generate_api_key(owner: str, api_key: str = Depends(_verify_api_key)):
    """Генерирует новый API ключ для внешней системы."""
    new_key = f"sk-msgov-{secrets.token_hex(12)}"
    API_KEYS[new_key] = {
        "owner": owner,
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    return {
        "api_key": new_key,
        "owner": owner,
        "created": API_KEYS[new_key]["created"],
        "permissions": ["score", "read", "sync"],
    }


@app.get("/api/v1/keys", tags=["API Keys"])
def list_api_keys(api_key: str = Depends(_verify_api_key)):
    """Список всех активных API ключей (маскированных)."""
    return [
        {
            "key_preview": f"{k[:12]}...{k[-4:]}",
            "owner": v["owner"],
            "created": v["created"],
        }
        for k, v in API_KEYS.items()
    ]