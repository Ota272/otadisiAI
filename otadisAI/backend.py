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

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
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

    # Признаки эффективности (фичи ML-модели)
    gross_output_growth: float = Field(
        ..., ge=-1.0, le=5.0,
        description="Рост валовой продукции за 2 года (доля, напр. 0.15 = +15%)"
    )
    pedigree_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Доля племенного поголовья в стаде (0–1)"
    )
    land_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Коэффициент использования земли (0–1)"
    )
    historical_survival_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Исторический показатель выживаемости скота (0–1)"
    )
    debt_load_ratio: float = Field(
        ..., ge=0.0, le=5.0,
        description="Долговая нагрузка (соотношение долг/EBITDA)"
    )
    subsidy_utilization_history: float = Field(
        ..., ge=0.0, le=1.0,
        description="Процент освоения прошлых субсидий (0–1)"
    )
    years_in_operation: int = Field(
        ..., ge=0, le=50,
        description="Лет в операционной деятельности"
    )
    veterinary_compliance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Выполнение ветеринарных норм (0–1)"
    )

    # Метаданные
    application_date: Optional[str] = Field(
        default=None,
        description="Дата подачи (ISO format)"
    )
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
    model_version: str = "GBM-v1.0-mock"


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
# Mock ML-модель (Gradient Boosting)
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "gross_output_growth",
    "pedigree_ratio",
    "land_utilization",
    "historical_survival_rate",
    "debt_load_ratio",
    "subsidy_utilization_history",
    "years_in_operation",
    "veterinary_compliance",
]

# Веса признаков для детерминированного скоринга
FEATURE_WEIGHTS = {
    "gross_output_growth":        0.25,
    "pedigree_ratio":             0.20,
    "land_utilization":           0.10,
    "historical_survival_rate":   0.15,
    "debt_load_ratio":           -0.10,  # Отрицательный: чем больше долг, тем хуже
    "subsidy_utilization_history":0.12,
    "years_in_operation":         0.05,
    "veterinary_compliance":      0.13,
}

# Нормализационные границы для каждого признака
FEATURE_BOUNDS = {
    "gross_output_growth":        (-0.5, 1.0),
    "pedigree_ratio":             (0.0, 1.0),
    "land_utilization":           (0.0, 1.0),
    "historical_survival_rate":   (0.0, 1.0),
    "debt_load_ratio":            (0.0, 5.0),
    "subsidy_utilization_history":(0.0, 1.0),
    "years_in_operation":         (0.0, 30.0),
    "veterinary_compliance":      (0.0, 1.0),
}

# Обучаем mock GBM на синтетических данных
def _train_mock_model() -> tuple:
    """Обучает mock GBM и возвращает (model, explainer)."""
    np.random.seed(42)
    n_samples = 500

    X = np.column_stack([
        np.random.uniform(-0.3, 0.8, n_samples),   # gross_output_growth
        np.random.uniform(0.0, 1.0, n_samples),     # pedigree_ratio
        np.random.uniform(0.3, 1.0, n_samples),     # land_utilization
        np.random.uniform(0.7, 1.0, n_samples),     # historical_survival_rate
        np.random.uniform(0.0, 4.0, n_samples),     # debt_load_ratio
        np.random.uniform(0.5, 1.0, n_samples),     # subsidy_utilization_history
        np.random.randint(0, 25, n_samples).astype(float),  # years_in_operation
        np.random.uniform(0.6, 1.0, n_samples),     # veterinary_compliance
    ])

    # Целевая переменная: взвешенная сумма + шум
    y = (
        X[:, 0] * 25 +
        X[:, 1] * 20 +
        X[:, 2] * 10 +
        X[:, 3] * 15 +
        (1 - X[:, 4] / 5) * 10 +
        X[:, 5] * 12 +
        np.clip(X[:, 6] / 30, 0, 1) * 5 +
        X[:, 7] * 13 +
        np.random.normal(0, 2, n_samples)
    )
    y = np.clip(y, 0, 100)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    return model, explainer


_MODEL, _EXPLAINER = _train_mock_model()


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _normalize_feature(name: str, value: float) -> float:
    """Нормализует признак в диапазон [0, 1]."""
    lo, hi = FEATURE_BOUNDS[name]
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))


def _compute_score(features: ApplicationFeatures) -> tuple[float, dict, list]:
    """
    Вычисляет скоринговый балл через ML-модель и SHAP.

    Returns:
        (score, shap_values_dict, shap_explanation_list)
    """
    raw = [
        features.gross_output_growth,
        features.pedigree_ratio,
        features.land_utilization,
        features.historical_survival_rate,
        features.debt_load_ratio,
        features.subsidy_utilization_history,
        float(features.years_in_operation),
        features.veterinary_compliance,
    ]

    X = np.array(raw).reshape(1, -1)
    score_raw = float(_MODEL.predict(X)[0])
    score = float(np.clip(score_raw, 0, 100))

    # SHAP-объяснения
    shap_vals = _EXPLAINER.shap_values(X)[0]
    shap_dict = {name: float(shap_vals[i]) for i, name in enumerate(FEATURE_NAMES)}

    # Человекочитаемые объяснения
    LABELS = {
        "gross_output_growth":        "Рост валовой продукции",
        "pedigree_ratio":             "Доля племенного поголовья",
        "land_utilization":           "Использование земельного фонда",
        "historical_survival_rate":   "Выживаемость скота (история)",
        "debt_load_ratio":            "Долговая нагрузка",
        "subsidy_utilization_history":"Освоение прошлых субсидий",
        "years_in_operation":         "Стаж работы предприятия",
        "veterinary_compliance":      "Ветеринарное соответствие",
    }

    explanation = []
    for name, shap_val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        raw_val = raw[FEATURE_NAMES.index(name)]
        direction = "positive" if shap_val > 0 else "negative"
        sign = "+" if shap_val > 0 else ""
        explanation.append({
            "feature": name,
            "label": LABELS[name],
            "raw_value": round(raw_val, 4),
            "shap_value": round(shap_val, 2),
            "direction": direction,
            "text": f"{sign}{shap_val:.1f} балл(ов): {LABELS[name]} = {raw_val:.2f}",
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
        "model_version": "GBM-v1.0-mock",
        "application_date": features.application_date or datetime.now().strftime("%Y-%m-%d"),
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
    SUBSIDY_TYPES = [
        "Приобретение племенного КРС",
        "Приобретение племенных овец",
        "Повышение продуктивности молочного стада",
        "Улучшение качества мясного производства",
        "Приобретение баранов-производителей",
    ]
    COMPANIES = [
        "ТОО «Алтай-Агро»", "ИП Сейткали А.Б.", "ТОО «СтепьПром»",
        "КХ «Береке»", "ТОО «АгроЗерно»", "ИП Жаксыбеков К.М.",
        "ТОО «КазАгро Холдинг»", "КФХ «Нур»",
    ]

    synced = []
    for _ in range(random.randint(3, 6)):
        growth = random.uniform(-0.3, 0.6)
        features = ApplicationFeatures(
            bin_iin=f"{random.randint(100000000000, 999999999999)}",
            company_name=random.choice(COMPANIES),
            region=random.choice(REGIONS),
            subsidy_type=random.choice(SUBSIDY_TYPES),
            requested_amount=random.uniform(2_000_000, 50_000_000),
            gross_output_growth=growth,
            pedigree_ratio=random.uniform(0.2, 1.0),
            land_utilization=random.uniform(0.4, 1.0),
            historical_survival_rate=random.uniform(0.6, 1.0),
            debt_load_ratio=random.uniform(0.0, 3.5),
            subsidy_utilization_history=random.uniform(0.5, 1.0),
            years_in_operation=random.randint(1, 20),
            veterinary_compliance=random.uniform(0.5, 1.0),
            source_system="giss",
            application_date=(
                datetime.now() - timedelta(days=random.randint(0, 30))
            ).strftime("%Y-%m-%d"),
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