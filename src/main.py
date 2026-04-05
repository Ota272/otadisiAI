
import copy
import io
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _load_env_vars():
    """Корень репозитория, затем frontend/.env (часто там лежит GEMINI при запуске только uvicorn)."""
    env_paths = [
        _REPO_ROOT / ".env",
        _REPO_ROOT / "frontend" / ".env",
    ]

    try:
        from dotenv import load_dotenv
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return
    except Exception:
        pass

    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass

_ENV_LOADED_FROM = _load_env_vars()

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.llm_routing import expert_opinion_available, primary_cloud_llm
from ml.shap_integration import (
    ScoringEngine,
    extract_features_from_documents_auto,
    extract_text_from_pdf,
    generate_gemini_expert_opinion,
)
from ml.compliance_checker import run_compliance_check, detect_subsidy_type
from src.store import applications_store

app = FastAPI(
    title="SmartAgro Score API",
    description="""
## Scoring Engine для субсидий МСХ РК

- **ML-модель**: XGBoost на 17 экономических признаках
- **Объяснимость**: SHAP TreeExplainer (не черный ящик!)
- **LLM-анализ**: Claude извлекает фичи из PDF-документов
- **Human-in-the-loop**: ИИ даёт рекомендацию, комиссия решает

⚠️ Система предоставляет рекомендации. Финальное решение — за комиссией МСХ РК.
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[ScoringEngine] = None

@app.on_event("startup")
async def startup_event():
    global _engine, _applications_db
    applications_store.init_db()
    _applications_db.clear()
    loaded = applications_store.load_all_applications()
    _applications_db.extend(loaded)
    print(f"SQLite: загружено {len(loaded)} заявок из хранилища")
    models_dir = _REPO_ROOT / "models"
    if not (models_dir / "xgb_scorer.joblib").exists():
        print("ВНИМАНИЕ: Модель не найдена. Запустите data_prep.py && train_model.py")
        _engine = None
    else:
        _engine = ScoringEngine(models_dir)
        print("SmartAgro Scoring Engine загружен и готов к работе")

def get_engine() -> ScoringEngine:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Scoring Engine не инициализирован. Обучите модель: python train_model.py",
        )
    return _engine

VALID_API_KEYS = {
    "sk-msgov-2025-demo-key-abc123": "МСХ РК — Отдел субсидирования",
    "sk-msgov-giss-integration-xyz": "Сервис тестовой нагрузки API",
}

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Неверный API ключ")
    return x_api_key

_applications_db: list[dict] = []
_decisions_db: list[dict] = []

def _register_application(record: dict, *, persist: bool | None = None) -> None:
    if persist is None:
        persist = not record.get("is_demo", False)
    stored = copy.deepcopy(record)
    _applications_db.append(stored)
    if persist:
        applications_store.upsert_application(stored)

def _persist_application_update(record: dict) -> None:
    if not record.get("is_demo"):
        applications_store.upsert_application(record)

class FarmerFeatures(BaseModel):

    bin_iin: str = Field(..., description="БИН/ИИН предприятия")
    company_name: str = Field(..., description="Наименование предприятия")
    region: str = Field(..., description="Область РК")
    subsidy_type: str = Field(..., description="Тип субсидии")
    requested_amount: float = Field(..., gt=0, description="Сумма (тенге)")

    gross_output_growth_yoy: Optional[float] = None
    land_to_livestock_ratio: Optional[float] = None
    historical_survival_rate: Optional[float] = None
    subsidy_dependence_index: Optional[float] = None
    veterinary_compliance: Optional[float] = None
    years_in_operation: Optional[int] = None
    pedigree_ratio: Optional[float] = None
    previous_subsidies_count: Optional[int] = None
    debt_load_ratio: Optional[float] = None

    normative: Optional[float] = None
    direction: Optional[str] = None
    source_system: Optional[str] = None
    application_date: Optional[str] = None

class DecisionRequest(BaseModel):
    application_id: str
    decision: str = Field(..., description="approved | rejected | review")
    comment: Optional[str] = None


class ExpertVerifyRequest(BaseModel):
    """Правки эксперта для Data Learning Loop (попадают в verified_payload и is_verified)."""
    verified_payload: Optional[dict] = None
    comment: Optional[str] = Field(
        None,
        description="Краткий комментарий; будет записан в verified_payload.expert_comment",
    )


def _merge_verified_payload(app: dict, fragment: dict) -> None:
    cur = app.get("verified_payload")
    base = dict(cur) if isinstance(cur, dict) else {}
    base.update(fragment)
    app["verified_payload"] = base


DIRECTION_CODE_MAP = {
    "Субсидирование в скотоводстве": 0,
    "Субсидирование в овцеводстве": 1,
    "Субсидирование в коневодстве": 2,
    "Субсидирование в птицеводстве": 3,
    "Субсидирование в верблюдоводстве": 4,
    "Субсидирование в свиноводстве": 5,
}

REGION_CODE_MAP = {
    "Алматинская область": 0, "Акмолинская область": 1, "Атырауская область": 2,
    "Восточно-Казахстанская область": 3, "Жамбылская область": 4,
    "Карагандинская область": 5, "Костанайская область": 6,
    "Кызылординская область": 7, "Мангистауская область": 8,
    "Павлодарская область": 9, "Северо-Казахстанская область": 10,
    "Туркестанская область": 11, "Западно-Казахстанская область": 12,
    "Актюбинская область": 13,     "область Абай": 14,
}

_MERGE_LOG_NULL_KEYS = frozenset({
    "years_in_operation",
    "debt_load_ratio",
    "subsidy_dependence_index",
    "gross_output_growth_yoy",
    "historical_survival_rate",
    "veterinary_compliance",
    "land_to_livestock_ratio",
    "livestock_count",
})


def _coerce_doc_scalar(val, key: str) -> float | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        v = float(val)
        if key == "gross_output_growth_yoy" and 1.0 < v <= 100.0:
            return v / 100.0
        return v
    if isinstance(val, str):
        s = val.strip().replace("\xa0", " ")
        sl = s.lower()
        if key == "years_in_operation":
            m = re.search(
                r"(?:основан[оаы]?\s+в|работает\s+с|с\s+года|с\s+)(\d{4})\s*(?:г\.?|года?)?",
                s,
                re.I,
            )
            if m:
                try:
                    y0 = int(m.group(1))
                    age = datetime.now().year - y0
                    if 1 <= age <= 80:
                        return float(age)
                except ValueError:
                    pass
        if ("менее" in sl or "до " in sl or "<" in sl) and re.search(r"%|\s*проц", sl):
            m = re.search(r"(?:менее|до|<)\s*([0-9]+(?:[.,][0-9]+)?)\s*%", sl)
            if m and (key == "subsidy_dependence_index" or "субсид" in sl or "зависим" in sl):
                try:
                    return min(0.99, float(m.group(1).replace(",", ".")) / 100.0)
                except ValueError:
                    pass
        m = re.search(r"(\d{1,2})\s*(?:лет|год|года|лет\s+работы)", s, re.I)
        if m and key == "years_in_operation":
            return float(m.group(1))
        if key == "gross_output_growth_yoy" and "%" in s:
            m = re.search(r"[-+]?\s*([0-9]+(?:[.,][0-9]+)?)\s*%", s)
            if m:
                try:
                    return float(m.group(1).replace(",", ".")) / 100.0
                except ValueError:
                    pass
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?=\s*%|\s*проц)", s)
        if m and "%" in s:
            try:
                pct = float(m.group(0).replace(",", "."))
                if key in ("historical_survival_rate", "veterinary_compliance", "pedigree_ratio", "subsidy_dependence_index"):
                    return pct / 100.0 if pct > 1.0 else pct
            except ValueError:
                pass
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s.replace(" ", ""))
        if m:
            try:
                return float(m.group(0).replace(",", "."))
            except ValueError:
                return None
    return None


def _merge_extracted_doc_features(
    feature_dict: dict,
    doc_features: dict,
    *,
    source_tag: str = "DOC",
    force: bool = False,          # REGEX_POST passes force=True to override any prior value
) -> None:
    if not doc_features:
        return
    for key, raw in doc_features.items():
        if key not in feature_dict:
            print(f"[{source_tag}_SKIP] {key}: ключ отсутствует в feature_dict")
            continue
        if raw is None:
            if key in _MERGE_LOG_NULL_KEYS:
                print(f"[{source_tag}_SKIP] {key}: null — текущее {feature_dict.get(key)!r}")
            continue
        coerced = _coerce_doc_scalar(raw, key)
        if coerced is None:
            print(f"[{source_tag}_SKIP] {key}: не удалось привести к числу, raw={raw!r}")
            continue
        old = feature_dict[key]
        try:
            old_f = float(old) if old is not None else None
            new_f = float(coerced)
        except (TypeError, ValueError):
            old_f, new_f = None, float(coerced)
        # Skip if value unchanged (unless force=True for REGEX_POST)
        if not force and old_f is not None and abs(old_f - new_f) <= 1e-9:
            continue
        feature_dict[key] = new_f
        print(f"[FEATURE_UPDATE] {key}: {old_f} → {new_f}  (source={source_tag})")

    lah_raw = doc_features.get("land_area_ha")
    lah = _coerce_doc_scalar(lah_raw, "land_area_ha") if lah_raw is not None else None
    if lah is None and lah_raw is not None:
        try:
            lah = float(lah_raw)
        except (TypeError, ValueError):
            lah = None
    lc = feature_dict.get("livestock_count")
    if (
        lah is not None
        and lc
        and float(lc) > 0
        and doc_features.get("land_to_livestock_ratio") is None
    ):
        try:
            coerced = float(lah) / float(lc)
            old = feature_dict.get("land_to_livestock_ratio")
            feature_dict["land_to_livestock_ratio"] = coerced
            try:
                changed = old is None or abs(float(old) - coerced) > 1e-6
            except (TypeError, ValueError):
                changed = True
            if changed:
                print(f"[{source_tag}_MERGE] land_to_livestock_ratio (из land_area_ha/livestock_count): {old} -> {coerced}")
        except (TypeError, ValueError):
            pass

def _regex_scoring_features_from_text(text: str) -> dict[str, float]:
    if not text or len(text) < 30:
        return {}
    out: dict[str, float] = {}

    def _in_negative_context(match_obj, window: int = 80) -> bool:
        """Проверяет что число не в отрицательном контексте."""
        start = max(0, match_obj.start() - window)
        end = min(len(text), match_obj.end() + window)
        ctx = text[start:end].lower()
        neg_contexts = ["не ", "нет ", "отсутствует", "пример", "напр.", "образец", "сноска"]
        return any(nc in ctx for nc in neg_contexts)

    m = re.search(r"долг\s*/\s*ebitda\s*[=:]\s*([0-9]+(?:[.,][0-9]+)?)", text, re.IGNORECASE)
    if m and not _in_negative_context(m):
        try: out["debt_load_ratio"] = float(m.group(1).replace(",", "."))
        except ValueError: pass
    else:
        if re.search(r"долг\s*/\s*ebitda\s*[=:]\s*(?:0(?:[.,]0+)?)(?:\s|$|[;.,])", text, re.IGNORECASE) or \
           re.search(r"долг\s*/\s*ebitda[\s\S]{0,50}?(?:нет|отсутствует|нулев|=\s*ноль)", text, re.IGNORECASE):
            out["debt_load_ratio"] = 0.0
        elif re.search(r"не\s+имеет\s+кредитн[\s\S]{0,40}нагруз", text, re.IGNORECASE) or \
             re.search(r"кредитной\s+нагрузк[\s\S]{0,30}?нет", text, re.IGNORECASE):
            out["debt_load_ratio"] = 0.0

    for pat in (
        r"([0-9]+(?:[.,][0-9]+)?)\s*га[\s\S]{0,25}/\s*голов",
        r"([0-9]+(?:[.,][0-9]+)?)\s*гектар[\s\S]{0,25}/\s*голов",
        r"[оО]беспеченност[\s\S]{0,50}([0-9]+(?:[.,][0-9]+)?)\s*га/\s*голов",
    ):
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try:
                out["land_to_livestock_ratio"] = float(mm.group(1).replace(",", "."))
            except ValueError: pass
            break

    for pat in (
        r"(?:рост|прирост)\s+валовой\s+продукц[\s\S]{0,70}?\+?\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
        r"валовой\s+продукц[\s\S]{0,80}?\+\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
        r"рост\s+продукц(?:ии|ия)[\s\S]{0,60}?составил[\s\S]{0,25}(\d+(?:[.,]\d+)?)\s*%",
        r"(?:рост|прирост)[\s\S]{0,40}?\+?\s*([0-9]+(?:[.,][0-9]+)?)\s*%\s*(?:за\s+год|г/г|год\s*к\s*году)",
        r"\+?\s*(\d+(?:[.,]\d+)?)\s*%\s*(?:рост|прирост)[\s\S]{0,30}?(?:продукц|валов)"
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                out["gross_output_growth_yoy"] = float(m.group(1).replace(",", ".")) / 100.0
                break
            except ValueError: pass

    for pat in (
        r"зависимост[\s\S]{0,55}?(?:менее|до\s*|<)\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
        r"зависимость\s+от\s+субсид[\s\S]{0,60}?([0-9]+(?:[.,][0-9]+)?)\s*%"
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).replace(",", ".")) / 100.0
                if 0 <= v <= 1: out["subsidy_dependence_index"] = v
                break
            except ValueError: pass

    for pat in (
        r"стаж\s+(\d{1,3})\s+лет",
        r"стаж[\s\S]{0,40}?(\d{1,3})\s*(?:лет|год|года)(?:\s+работы)?"
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                y = float(m.group(1))
                if 0 < y <= 80: out["years_in_operation"] = y
                break
            except ValueError: pass

    if "years_in_operation" not in out:
        m = re.search(r"(?:основан[оаы]?\s+в|работает\s+с|с\s+года\s+)\s*(\d{4})\s*(?:г\.?|года?)?", text, re.IGNORECASE)
        if m:
            try:
                y0 = int(m.group(1))
                age = datetime.now().year - y0
                if 1 <= age <= 80: out["years_in_operation"] = float(age)
            except ValueError: pass

    for pat, key in (
        (r"(?:выживаемост|сохранност)[\s\S]{0,70}?(\d{1,3}(?:[.,]\d+)?)\s*%", "historical_survival_rate"),
        (r"(?:сохранност|выживаемост)[\s\S]{0,30}[:=]\s*(\d{1,3}(?:[.,]\d+)?)\s*%", "historical_survival_rate"),
    ):
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try:
                pct = float(mm.group(1).replace(",", "."))
                out[key] = pct / 100.0 if pct > 1.0 else pct
            except ValueError: pass
            break

    if "historical_survival_rate" not in out:
        m = re.search(r"падеж[\s\S]{0,70}?(\d{1,2}(?:[.,]\d+)?)\s*%", text, re.IGNORECASE)
        if m:
            try:
                raw_p = float(m.group(1).replace(",", "."))
                frac = raw_p / 100.0 if raw_p > 1.0 else raw_p
                out["historical_survival_rate"] = max(0.0, min(1.0, 1.0 - frac))
            except ValueError: pass

    m = re.search(r"ветеринарн[\s\S]{0,40}соответств[\s\S]{0,40}?(\d{1,3}(?:[.,]\d+)?)\s*%", text, re.IGNORECASE)
    if not m:
        m = re.search(r"ветеринарное\s+соответствие\s*[:=]?\s*(\d{1,3}(?:[.,]\d+)?)\s*%?", text, re.IGNORECASE)
    if m:
        try:
            pct = float(m.group(1).replace(",", "."))
            out["veterinary_compliance"] = pct / 100.0 if pct > 1.0 else pct
        except ValueError: pass

    # ── pedigree_ratio: "Доля племенного поголовья: 75%" / "племенное поголовье 70%" ──
    for pat in (
        r"доля\s+племенн[\s\S]{0,40}?(\d{1,3}(?:[.,]\d+)?)\s*%",
        r"племенн[\s\S]{0,30}поголовь[\s\S]{0,30}?(\d{1,3}(?:[.,]\d+)?)\s*%",
        r"племенн[\s\S]{0,20}[:=]\s*(\d{1,3}(?:[.,]\d+)?)\s*%",
    ):
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try:
                pct = float(mm.group(1).replace(",", "."))
                out["pedigree_ratio"] = pct / 100.0 if pct > 1.0 else pct
                break
            except ValueError: pass

    # ── previous_subsidies_count: "получено субсидий: 5" / "предыдущие субсидии: 3" ──
    for pat in (
        r"(?:полученн[ыхую]|предыдущ[иеых]|количество)\s+(?:субсиди[йяе])[\s\S]{0,30}?[:=]?\s*(\d+)",
        r"(?:субсиди[йяе])[\s\S]{0,30}?(?:получен[оы]|ранее)[\s\S]{0,30}?(\d+)",
        r"(\d+)\s+(?:субсиди[йяе])\s+(?:получен[оы]|ранее|предыдущ)",
    ):
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try:
                cnt = int(mm.group(1))
                if 0 <= cnt <= 50:
                    out["previous_subsidies_count"] = float(cnt)
                    break
            except ValueError: pass

    # ── debt_load_ratio: усиленные паттерны ──
    if "debt_load_ratio" not in out:
        for pat in (
            r"долг\s*/\s*ebitda[\s\S]{0,20}?[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
            r"долговая\s+нагрузк[\s\S]{0,30}?[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
            r"(?:долг|задолженность)[\s\S]{0,20}(?:ebitda|долгов)[\s\S]{0,20}?([0-9]+(?:[.,][0-9]+)?)",
        ):
            mm = re.search(pat, text, re.IGNORECASE)
            if mm:
                try:
                    out["debt_load_ratio"] = float(mm.group(1).replace(",", "."))
                    break
                except ValueError: pass

    # ── subsidy_dependence_index: усиленные паттерны ──
    if "subsidy_dependence_index" not in out:
        for pat in (
            r"зависимост[\s\S]{0,40}от\s+субсид[\s\S]{0,30}?[:=]\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
            r"зависимост[\s\S]{0,40}[:=]\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
            r"субсиди[йяе]\s+зависимост[\s\S]{0,30}?(\d{1,3}(?:[.,]\d+)?)\s*%",
        ):
            mm = re.search(pat, text, re.IGNORECASE)
            if mm:
                try:
                    v = float(mm.group(1).replace(",", ".")) / 100.0
                    if 0 <= v <= 1:
                        out["subsidy_dependence_index"] = v
                        break
                except ValueError: pass

    # ── years_in_operation: усиленные паттерны ──
    if "years_in_operation" not in out:
        for pat in (
            r"стаж\s+работ[ыае][\s\S]{0,20}?[:=]?\s*(\d{1,3})\s*(?:лет|год|года)",
            r"работает\s+(\d{1,3})\s*(?:лет|год|года)",
            r"(\d{1,3})\s*(?:лет|год|года)\s+(?:работы|стажа|опыта)",
        ):
            mm = re.search(pat, text, re.IGNORECASE)
            if mm:
                try:
                    y = float(mm.group(1))
                    if 0 < y <= 80:
                        out["years_in_operation"] = y
                        break
                except ValueError: pass

    return out

_MODEL_FEATURES_NEEDING_IMPUTE = (
    "gross_output_growth_yoy",
    "land_to_livestock_ratio",
    "historical_survival_rate",
    "subsidy_dependence_index",
    "veterinary_compliance",
    "years_in_operation",
    "pedigree_ratio",
    "previous_subsidies_count",
    "debt_load_ratio",
    "grazing_norm_deviation",
    "natural_loss_risk_score",
)

_FEATURE_IMPUTE_IF_MISSING: dict[str, float] = {
    # Нейтральные дефолты: нет данных = среднее по рынку (не наказываем и не поощряем)
    "gross_output_growth_yoy": 0.03,      # небольшой рост (было -0.05 спад)
    "land_to_livestock_ratio": 6.0,       # средняя земля
    "historical_survival_rate": 0.88,     # 88% выживаемость (было 0.75)
    "subsidy_dependence_index": 0.25,     # умеренная зависимость (было 0.50)
    "veterinary_compliance": 0.70,        # 70% соответствие (было 0.50)
    "years_in_operation": 8.0,            # средний опыт (было 3.0)
    "pedigree_ratio": 0.30,               # умеренное племенное (было 0.15)
    "previous_subsidies_count": 2.0,      # немного истории (было 0.0)
    "debt_load_ratio": 1.2,               # умеренный долг (было 2.5)
    "grazing_norm_deviation": 0.1,        # небольшое отклонение (было 0.5)
    "natural_loss_risk_score": 1.1,       # чуть выше нормы (было 1.5)
}


def _impute_missing_model_features(feature_dict: dict, *, tag: str = "IMPUTE") -> None:
    for key in _MODEL_FEATURES_NEEDING_IMPUTE:
        if key not in feature_dict:
            continue
        raw = feature_dict[key]
        if raw is not None:
            continue
        default = _FEATURE_IMPUTE_IF_MISSING.get(key)
        if default is None:
            continue
        feature_dict[key] = default
        print(f"WARNING: Field {key} is missing, using default {default} ({tag})")


def _build_feature_dict(f: FarmerFeatures) -> dict:
    now = datetime.now()
    hour = float(now.hour)
    month = float(now.month)

    if f.application_date:
        try:

            dt_str = f.application_date.strip()
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                dt = datetime.strptime(dt_str, "%d.%m.%Y %H:%M:%S")
            hour = float(dt.hour)
            month = float(dt.month)
        except ValueError:
            pass

    normative = f.normative
    if normative is None:
        normative = 15000.0
        print("WARNING: Field normative is missing, using default 15000.0")

    direction = f.direction
    if direction is None:
        direction = "Субсидирование в скотоводстве"
        print(
            "WARNING: Field direction is missing, "
            "using default Субсидирование в скотоводстве"
        )

    if f.source_system is None:
        print("WARNING: Field source_system is missing, using default manual")

    log_amount = float(np.log1p(f.requested_amount))
    livestock_count = max(1.0, f.requested_amount / max(normative, 1))
    direction_code = float(DIRECTION_CODE_MAP.get(direction, 6))
    region_encoded = float(REGION_CODE_MAP.get(f.region, 7))
    dir_l = direction.lower()
    is_pedigree = 1.0 if ("племен" in dir_l or "племен" in f.subsidy_type.lower()) else 0.0
    is_producer = 1.0 if "производи" in f.subsidy_type.lower() else 0.0

    def _opt_float(v):
        return float(v) if v is not None else None

    return {
        "gross_output_growth_yoy":     _opt_float(f.gross_output_growth_yoy),
        "land_to_livestock_ratio":     _opt_float(f.land_to_livestock_ratio),
        "historical_survival_rate":    _opt_float(f.historical_survival_rate),
        "subsidy_dependence_index":    _opt_float(f.subsidy_dependence_index),
        "veterinary_compliance":       _opt_float(f.veterinary_compliance),
        "years_in_operation":          _opt_float(f.years_in_operation),
        "pedigree_ratio":              _opt_float(f.pedigree_ratio),
        "previous_subsidies_count":    _opt_float(f.previous_subsidies_count),
        "debt_load_ratio":             _opt_float(f.debt_load_ratio),
        "land_area_ha":                None,
        "has_vet_passport":            None,
        "log_amount":                  log_amount,
        "livestock_count":             livestock_count,
        "direction_code":              direction_code,
        "is_pedigree":                 is_pedigree,
        "is_producer":                 is_producer,
        "hour_submitted":              hour,
        "month_submitted":             month,
        "region_encoded":              region_encoded,
    }

@app.get("/", tags=["Info"])
def root():
    return {
        "system": "SmartAgro Score",
        "version": "2.0.0",
        "model_loaded": _engine is not None,
        "docs": "/docs",
    }

@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model_loaded": _engine is not None,
            "timestamp": datetime.now().isoformat()}

def _build_score_response(
    features: FarmerFeatures,
    engine: ScoringEngine,
    *,
    include_shap: bool = True,
) -> dict:
    feature_dict = _build_feature_dict(features)
    _impute_missing_model_features(feature_dict, tag="IMPUTE_FORM")
    result = engine.score_farmer(feature_dict, include_shap=include_shap)

    app_id = str(uuid.uuid4())[:8].upper()
    app_date = features.application_date or datetime.now().strftime("%Y-%m-%d")

    return {
        "application_id":        app_id,
        "company_name":          features.company_name,
        "bin_iin":               features.bin_iin,
        "region":                features.region,
        "subsidy_type":          features.subsidy_type,
        "direction":             features.direction,
        "score":                 result["score"],
        "score_ml":              result["score"],
        "score_doc":             None,
        "ml_weight_used":        1.0,
        "doc_weight_used":       0.0,
        "manual_review_required": False,
        "zone":                  result["zone"],
        "score_zone":            result["zone"],
        "final_score":           float(result["score"]),
        "is_verified":           0,
        "zone_label":            result["zone_label"],
        "recommendation":        result["recommendation"],
        "verdict":               result["verdict"],
        "top_positive_factors":  result["top_positive_factors"],
        "top_negative_factors":  result["top_negative_factors"],
        "all_shap_values":       result["all_shap_values"],
        "shap_base_value":       result.get("shap_base_value"),
        "raw_features_used":     result.get("raw_features_used", {}),
        "compliance":            None,
        "documents_text_chars":  0,
        "documents_extracted_ok": False,
        "documents_pdf_count":   0,
        "documents_extracted_text": None,
        "documents_extraction_note": None,
        "requested_amount":      features.requested_amount,
        "application_date":      app_date,
        "source_system":         features.source_system,
        "model_version":         result["model_version"],
        "calculated_at":         datetime.now().isoformat(),
        "is_demo":               False,
    }

@app.post("/api/v1/score", tags=["Scoring"], summary="Рассчитать скоринговый балл")
def score_application(
    features: FarmerFeatures,
    api_key: str = Depends(verify_api_key),
    engine: ScoringEngine = Depends(get_engine),
):
    response_data = _build_score_response(features, engine)
    _register_application(response_data)
    return response_data

def _wait_gemini_file_ready(uploaded, genai_mod, timeout_s: float = 120.0):
    from google.generativeai import protos as _protos

    t0 = time.time()
    f = uploaded
    while f.state != _protos.File.State.ACTIVE:
        if f.state == _protos.File.State.FAILED:
            err = getattr(f, "error", None)
            raise RuntimeError(f"Файл Gemini FAILED: {err}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Файл не перешёл в ACTIVE за отведённое время")
        time.sleep(0.7)
        f = genai_mod.get_file(f.name)
    return f

def _is_gemini_quota_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    if "429" in str(exc) or "quota" in s or "resource exhausted" in s:
        return True
    return "ResourceExhausted" in type(exc).__name__

async def _gemini_ocr_pdfs(pdf_items: list[tuple[str, bytes]], api_key: str) -> tuple[str, str | None]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    uploaded_refs = []
    results       = []

    for fname, pdf_bytes in pdf_items:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            print(f"[gemini_ocr] Загружаю '{fname}' ({len(pdf_bytes)//1024} КБ) через Files API…")
            uploaded = genai.upload_file(
                path=tmp_path,
                mime_type="application/pdf",
                display_name=fname,
            )
            uploaded = _wait_gemini_file_ready(uploaded, genai)
            uploaded_refs.append(uploaded)

            response = model.generate_content([
                uploaded,
                (
                    "Извлеки полный текст из этого документа. "
                    "Сохраняй структуру: заголовки, таблицы, числа, даты. "
                    "Выводи только содержимое документа — без своих комментариев."
                ),
            ])
            text = (response.text or "").strip()
            if text:
                results.append(f"=== {fname} ===\n{text}")
                print(f"[gemini_ocr] '{fname}': {len(text)} симв. извлечено")
            else:
                print(f"[gemini_ocr] '{fname}': Gemini вернул пустой ответ")

        except Exception as exc:
            print(f"[gemini_ocr] Ошибка для '{fname}': {exc}")
            if _is_gemini_quota_error(exc):
                for u in uploaded_refs:
                    try:
                        genai.delete_file(u.name)
                    except Exception:
                        pass
                return "\n\n".join(results), (
                    "Gemini API: квота исчерпана (HTTP 429). Облачный OCR остановлен. "
                    "Текст из PDF при возможности извлекается локально (PyMuPDF). "
                    "Для облака: включите оплату в Google AI Studio / снизьте число запросов."
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    for uploaded in uploaded_refs:
        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass

    return "\n\n".join(results), None


def _groq_ocr_pdfs_sync(pdf_items: list[tuple[str, bytes]]) -> tuple[str, str | None]:
    import fitz
    from ml.llm_routing import groq_vision_ocr_page

    results: list[str] = []
    try:
        max_pages = max(1, int(os.getenv("GROQ_OCR_MAX_PAGES", "4")))
    except ValueError:
        max_pages = 4

    for fname, pdf_bytes in pdf_items:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                n = min(max_pages, doc.page_count)
                parts: list[str] = []
                for i in range(n):
                    page = doc.load_page(i)
                    w, h = page.rect.width, page.rect.height
                    scale = min(2.0, 1200.0 / max(w, h, 1.0))
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    png = pix.tobytes("png")
                    text = groq_vision_ocr_page(
                        png_bytes=png,
                        prompt=(
                            "Извлеки весь читаемый текст с изображения страницы документа. "
                            "Сохраняй порядок строк, числа и даты. "
                            "Выводи только текст документа, без комментариев."
                        ),
                    )
                    if text:
                        parts.append(text)
                if parts:
                    joined = "\n\n".join(parts)
                    results.append(f"=== {fname} ===\n{joined}")
                    print(f"[groq_ocr] '{fname}': {len(joined)} симв. с {n} стр.")
            finally:
                doc.close()
        except Exception as exc:
            print(f"[groq_ocr] Ошибка для '{fname}': {exc}")

    note = None
    if pdf_items and not results:
        note = (
            "Groq Vision OCR не вернул текст (проверьте GROQ_OCR_MODEL, лимиты или качество скана)."
        )
    return "\n\n".join(results), note


async def _groq_ocr_pdfs(pdf_items: list[tuple[str, bytes]]) -> tuple[str, str | None]:
    import asyncio

    return await asyncio.to_thread(_groq_ocr_pdfs_sync, pdf_items)


@app.post("/api/v1/score-with-documents", tags=["Scoring"],
          summary="Скоринг + LLM-анализ PDF документов")
async def score_with_documents(
    features_json: str = Form(...),
    documents: list[UploadFile] = File(default=[]),
    api_key: str = Depends(verify_api_key),
    engine: ScoringEngine = Depends(get_engine),
):
    import json as json_lib

    try:
        features_dict = json_lib.loads(features_json)
        features = FarmerFeatures(**features_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка парсинга данных: {e}")

    feature_dict = _build_feature_dict(features)
    llm_summary = None
    combined_text     = ""                                       
    combined_text_llm = ""                                               
    extraction_note: str | None = None                                                  

    if documents:
        all_texts   = []
        pdf_items   = []                                                  

        for doc in documents:
            if not doc.filename:
                continue
            content    = await doc.read()
            fname_lower = doc.filename.lower()

            if fname_lower.endswith(".pdf"):
                pdf_items.append((doc.filename, content))

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    text = extract_text_from_pdf(tmp_path)
                    if text.strip():
                        all_texts.append(f"=== {doc.filename} ===\n{text}")
                except Exception as _e:
                    print(f"[pdfplumber] {doc.filename}: {_e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

            elif fname_lower.endswith(".txt"):
                text = content.decode("utf-8", errors="ignore")
                all_texts.append(f"=== {doc.filename} ===\n{text}")

        gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        doc_backend = primary_cloud_llm()
        missing = [item for item in pdf_items
                   if not any(item[0] in t for t in all_texts)]

        if missing:
            ocr_text, ocr_note = "", None
            if doc_backend == "groq" and groq_key:
                print(f"[groq_ocr] локально нет текста по {len(missing)} PDF — Groq Vision…")
                ocr_text, ocr_note = await _groq_ocr_pdfs(missing)
            elif gemini_api_key:
                print(f"[gemini_ocr] локально нет текста по {len(missing)} PDF — Gemini Files API…")
                ocr_text, ocr_note = await _gemini_ocr_pdfs(missing, gemini_api_key)
            elif groq_key:
                print(f"[groq_ocr] fallback: Groq Vision для {len(missing)} PDF (нет GEMINI_API_KEY)…")
                ocr_text, ocr_note = await _groq_ocr_pdfs(missing)

            extraction_note = ocr_note
            if (ocr_text or "").strip():
                all_texts.append(ocr_text)
                print(f"[cloud_ocr] добавлено {len(ocr_text)} симв.")
            elif not groq_key and not gemini_api_key:
                extraction_note = (
                    f"Нет GROQ_API_KEY и GEMINI_API_KEY — {len(missing)} файл(ов) со сканами без текстового слоя не распознаны."
                )
                print("[cloud_ocr] нет ключей для облачного OCR")

        if all_texts:
            combined_text = "\n\n".join(all_texts)
            MAX_CHARS = 60_000
            combined_text_llm = combined_text[:MAX_CHARS] if len(combined_text) > MAX_CHARS else combined_text
            print(f"[docs] итого текст: {len(combined_text)} → {len(combined_text_llm)} симв. для LLM")
            _llm_doc = "Groq" if (doc_backend == "groq" and groq_key) else (
                "Gemini" if gemini_api_key else ("Groq" if groq_key else "нет")
            )
            print(f"[docs] облачный LLM для документов (LLM_PROVIDER / ключи): {_llm_doc}")

            rx_pre = _regex_scoring_features_from_text(combined_text_llm)
            if rx_pre:
                _merge_extracted_doc_features(feature_dict, rx_pre, source_tag="REGEX_PRE")
                print(f"[docs] regex (до LLM): {rx_pre}")

            extraction = extract_features_from_documents_auto(combined_text_llm)
            llm_summary = extraction.get("llm_summary")
            doc_features = extraction.get("features") or {}
            if doc_features:
                _merge_extracted_doc_features(feature_dict, doc_features, source_tag="LLM_DOC")
            if extraction.get("extraction_status") != "success":
                print(
                    f"[docs] JSON-извлечение из PDF: {extraction.get('extraction_status')} "
                    f"(полей в ответе: {len(doc_features)})"
                )

            if doc_features.get("has_vet_passport") == 1.0:
                vc0 = feature_dict.get("veterinary_compliance")
                try:
                    base_vc = float(vc0) if vc0 is not None else 0.88
                except (TypeError, ValueError):
                    base_vc = 0.88
                feature_dict["veterinary_compliance"] = min(1.0, base_vc + 0.05)
                print(f"[docs] has_vet_passport: veterinary_compliance скорректировано до {feature_dict['veterinary_compliance']}")

            rx_post = _regex_scoring_features_from_text(combined_text_llm)
            if rx_post:
                _merge_extracted_doc_features(
                    feature_dict, rx_post, source_tag="REGEX_POST", force=True
                )
                print(f"[docs] REGEX_POST (приоритет над LLM): {rx_post}")

    _impute_missing_model_features(feature_dict, tag="IMPUTE_DOC")
    result = engine.score_farmer(feature_dict, llm_context=llm_summary)
    base_score = result["score"]

    llm_expert_opinion = None

    compliance_report   = None
    final_score         = base_score
    score_doc           = None
    ml_weight           = 1.0
    doc_weight          = 0.0
    manual_review_flag  = False

    if combined_text.strip():
        compliance_report = run_compliance_check(
            documents_text=combined_text_llm,
            subsidy_name=features.subsidy_type,
            direction=features.direction,
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            use_embeddings=True,
        )

        score_doc = float(compliance_report.get("overall_score_pct", 50.0))

        doc_completeness = compliance_report.get("doc_completeness")
        if doc_completeness is None:
            doc_completeness = score_doc / 100.0

        if doc_completeness >= 0.70:
            ml_weight, doc_weight = 0.30, 0.70
        elif doc_completeness >= 0.40:
            ml_weight, doc_weight = 0.50, 0.50
        else:
            ml_weight, doc_weight = 0.70, 0.30
            manual_review_flag = True

        raw_final  = ml_weight * base_score + doc_weight * score_doc
        final_score = round(float(max(1.0, min(100.0, raw_final))), 1)

        if final_score >= 80:
            final_zone          = "green"
            final_zone_label    = "Зелёная зона (80–100)"
            final_recommendation = "Строго рекомендовано к включению в шорт-лист"
        elif final_score >= 50:
            final_zone          = "yellow"
            final_zone_label    = "Жёлтая зона (50–79)"
            final_recommendation = "Рекомендуется дополнительное рассмотрение комиссией"
        else:
            final_zone          = "red"
            final_zone_label    = "Красная зона (0–49)"
            final_recommendation = "Не рекомендовано — выявлены существенные риски"
    else:
        final_zone          = result["zone"]
        final_zone_label    = result["zone_label"]
        final_recommendation = result["recommendation"]

    app_id = str(uuid.uuid4())[:8].upper()

    _MAX_DOC_TEXT_STORE = 280_000
    _doc_text_store = (combined_text[:_MAX_DOC_TEXT_STORE] if combined_text.strip() else None)
    if not combined_text.strip() and documents and not extraction_note:
        extraction_note = (
            "Текст из PDF не извлечён: возможно сканы без текстового слоя; проверьте GROQ_API_KEY (OCR) или GEMINI_API_KEY."
        )

    response_data = {
        "application_id":         app_id,
        "company_name":           features.company_name,
        "bin_iin":                features.bin_iin,
        "region":                 features.region,

        "score_ml":               base_score,
        "score_doc":              score_doc,
        "score":                  final_score,
        "ml_weight_used":         round(ml_weight, 2),
        "doc_weight_used":        round(doc_weight, 2),
        "manual_review_required": manual_review_flag,
        "zone":                   final_zone,
        "score_zone":             final_zone,
        "final_score":            float(final_score),
        "is_verified":            0,
        "zone_label":             final_zone_label,
        "recommendation":         final_recommendation,

        "verdict":                result["verdict"],
        "top_positive_factors":   result["top_positive_factors"],
        "top_negative_factors":   result["top_negative_factors"],
        "all_shap_values":        result["all_shap_values"],
        "shap_base_value":        result.get("shap_base_value"),
        "raw_features_used":      result.get("raw_features_used", {}),

        "compliance": compliance_report,
        "llm_expert_opinion": llm_expert_opinion,  # только объяснение

        "llm_document_analysis":  llm_summary,
        "documents_processed":    len(documents),
        "documents_text_chars":   len(combined_text),
        "documents_extracted_ok": bool(combined_text.strip()),
        "documents_pdf_count":    sum(
            1 for d in documents if d.filename and d.filename.lower().endswith(".pdf")
        ),
        "documents_extracted_text": _doc_text_store,
        "documents_extraction_note": extraction_note,
        "requested_amount":       features.requested_amount,
        "subsidy_type":           features.subsidy_type,
        "direction":              features.direction,
        "calculated_at":          datetime.now().isoformat(),
        "model_version":          result["model_version"],
        "source_system":          features.source_system,
        "is_demo":                False,
    }

    if combined_text.strip() and expert_opinion_available():
        try:
            llm_expert_opinion = generate_gemini_expert_opinion(
                response_data,
                os.getenv("GEMINI_API_KEY"),
            )
            response_data["llm_expert_opinion"] = llm_expert_opinion
        except Exception as e:
            print(f"LLM expert opinion ошибка: {e}")

    _register_application(response_data)
    return response_data

@app.get("/api/v1/applications", tags=["Applications"])
def get_applications(
    api_key: str = Depends(verify_api_key),
    zone: Optional[str] = None,
    min_score: Optional[float] = None,
):
    apps = list(_applications_db)
    if zone:
        apps = [a for a in apps if a.get("zone") == zone]
    if min_score is not None:
        apps = [a for a in apps if a.get("score", 0) >= min_score]
    return sorted(apps, key=lambda x: x.get("score", 0), reverse=True)

@app.get("/api/v1/applications/{application_id}", tags=["Applications"])
def get_application(application_id: str, api_key: str = Depends(verify_api_key)):
    for app in _applications_db:
        if app.get("application_id") == application_id:
            return app
    raise HTTPException(status_code=404, detail="Заявка не найдена")

@app.post("/api/v1/decision", tags=["Decisions"])
def record_decision(
    decision_req: DecisionRequest,
    api_key: str = Depends(verify_api_key),
):
    if decision_req.decision not in ("approved", "rejected", "review"):
        raise HTTPException(status_code=400,
                            detail="decision: approved | rejected | review")

    record = {
        "application_id": decision_req.application_id,
        "decision":        decision_req.decision,
        "comment":         decision_req.comment,
        "decided_at":      datetime.now().isoformat(),
    }
    _decisions_db.append(record)

    for app in _applications_db:
        if app.get("application_id") == decision_req.application_id:
            app.update({
                "decision":        decision_req.decision,
                "decided_at":      record["decided_at"],
                "officer_comment": decision_req.comment,
            })
            _merge_verified_payload(
                app,
                {
                    "commission_decision": {
                        "decision": decision_req.decision,
                        "comment": decision_req.comment,
                        "decided_at": record["decided_at"],
                    }
                },
            )
            app["is_verified"] = 1
            _persist_application_update(app)
            break

    return record


@app.post(
    "/api/v1/applications/{application_id}/expert-verify",
    tags=["Applications"],
    summary="Отметить заявку проверенной экспертом",
)
def expert_verify_application(
    application_id: str,
    body: ExpertVerifyRequest,
    api_key: str = Depends(verify_api_key),
):
    """Ставит is_verified=1 и сохраняет JSON правок; строка попадает в get_training_data / training-samples."""
    for app in _applications_db:
        if app.get("application_id") == application_id:
            fragment = dict(body.verified_payload or {})
            if body.comment is not None:
                fragment.setdefault("expert_comment", body.comment)
            fragment["verified_at"] = datetime.now().isoformat()
            _merge_verified_payload(app, fragment)
            app["is_verified"] = 1
            _persist_application_update(app)
            return app
    raise HTTPException(status_code=404, detail="Заявка не найдена")


@app.get(
    "/api/v1/training-samples",
    tags=["Analytics"],
    summary="Выборка для обучения (проверенные заявки)",
)
def list_training_samples(
    api_key: str = Depends(verify_api_key),
    zone: Optional[str] = None,
):
    """
    Все заявки с is_verified=1; при zone=green|yellow|red — только эта зона.
    Проверка без SQL: сравните count с ожиданием после expert-verify / decision.
    """
    items = applications_store.get_training_data(zone=zone)
    return {
        "zone_filter": zone,
        "count": len(items),
        "items": items,
    }


@app.post("/api/v1/giss/sync", tags=["Demo"], summary="Сгенерировать тестовые заявки")
def sync_demo_applications(
    api_key: str = Depends(verify_api_key),
    engine: ScoringEngine = Depends(get_engine),
):
    import random

    REGIONS    = ["Алматинская область", "Акмолинская область", "Жамбылская область",
                  "Западно-Казахстанская область", "область Абай", "Атырауская область"]
    COMPANIES  = ["ТОО «Алтай-Агро»", "ИП Сейткали А.Б.", "ТОО «СтепьПром»",
                  "КХ «Береке»", "ТОО «АгроЗерно»", "ТОО «КазАгро Холдинг»"]
    DIRECTIONS = ["Субсидирование в скотоводстве", "Субсидирование в овцеводстве",
                  "Субсидирование в птицеводстве"]

    synced = []
    for _ in range(random.randint(3, 7)):
        f = FarmerFeatures(
            bin_iin=str(random.randint(100000000000, 999999999999)),
            company_name=random.choice(COMPANIES),
            region=random.choice(REGIONS),
            subsidy_type="Субсидия на племенное поголовье",
            requested_amount=random.uniform(2_000_000, 50_000_000),
            direction=random.choice(DIRECTIONS),
            normative=random.choice([15000, 150000, 20000]),
            gross_output_growth_yoy=random.uniform(-0.2, 0.5),
            land_to_livestock_ratio=random.uniform(0.5, 6.0),
            historical_survival_rate=random.uniform(0.65, 0.98),
            subsidy_dependence_index=random.uniform(0.1, 0.8),
            veterinary_compliance=random.uniform(0.6, 1.0),
            years_in_operation=random.randint(1, 20),
            pedigree_ratio=random.uniform(0.2, 1.0),
            previous_subsidies_count=random.randint(0, 10),
            debt_load_ratio=random.uniform(0.3, 4.0),
            source_system="demo",
        )
        rec = _build_score_response(f, engine, include_shap=False)
        rec["is_demo"] = True
        rec["source_system"] = "demo"
        _register_application(rec, persist=False)
        synced.append(rec)

    return {
        "synced_count": len(synced),
        "source":       "Тестовые заявки (демо)",
        "synced_at":    datetime.now().isoformat(),
        "applications": synced,
    }

@app.get("/api/v1/analytics/summary", tags=["Analytics"])
def get_analytics_summary(api_key: str = Depends(verify_api_key)):
    real = [a for a in _applications_db if not a.get("is_demo")]
    if not real:
        return {"message": "Нет данных для анализа"}

    scores  = [a.get("score", 0) for a in real]
    amounts = [a.get("requested_amount", 0) for a in real]
    zone_counts = {"green": 0, "yellow": 0, "red": 0}
    for a in real:
        z = a.get("zone", "yellow")
        if z in zone_counts:
            zone_counts[z] += 1

    return {
        "total_applications":    len(real),
        "total_requested_tenge": sum(amounts),
        "average_score":         round(float(np.mean(scores)), 1),
        "median_score":          round(float(np.median(scores)), 1),
        "zone_distribution":     zone_counts,
        "total_decisions":       len(_decisions_db),
        "approved":  sum(1 for d in _decisions_db if d["decision"] == "approved"),
        "rejected":  sum(1 for d in _decisions_db if d["decision"] == "rejected"),
    }


@app.get("/api-docs", include_in_schema=False)
def api_docs_v2():
    """Отдаёт красивую HTML-страницу с документацией API v2."""
    from fastapi.responses import HTMLResponse
    import os

    docs_path = os.path.join(os.path.dirname(__file__), "api_docs_v2.html")
    if not os.path.exists(docs_path):
        docs_path = os.path.join(os.path.dirname(__file__), "..", "api_docs_v2.html")

    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        raise HTTPException(status_code=404, detail="API documentation not found")
