
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

def _load_env_vars():
    env_path = Path(__file__).resolve().with_name(".env")

    try:
        from dotenv import load_dotenv                
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return str(env_path)
    except Exception:
        pass

    if not env_path.exists():
        return None

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
        return str(env_path)
    except Exception:
        return None

_ENV_LOADED_FROM = _load_env_vars()

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shap_integration import ScoringEngine, extract_features_from_documents, extract_text_from_pdf
from compliance_checker import run_compliance_check, detect_subsidy_type
import applications_store

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
    models_dir = Path("models")
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
    _applications_db.append(record)
    if persist:
        applications_store.upsert_application(record)

def _persist_application_update(record: dict) -> None:
    if not record.get("is_demo"):
        applications_store.upsert_application(record)

class FarmerFeatures(BaseModel):

    bin_iin: str = Field(..., description="БИН/ИИН предприятия")
    company_name: str = Field(..., description="Наименование предприятия")
    region: str = Field(..., description="Область РК")
    subsidy_type: str = Field(..., description="Тип субсидии")
    requested_amount: float = Field(..., gt=0, description="Сумма (тенге)")

    gross_output_growth_yoy: float = Field(default=0.0, ge=-0.5, le=2.0)
    land_to_livestock_ratio: float = Field(default=2.0, ge=0.1, le=15.0)
    historical_survival_rate: float = Field(default=0.88, ge=0.0, le=1.0)
    subsidy_dependence_index: float = Field(default=0.3, ge=0.0, le=1.0)
    veterinary_compliance: float = Field(default=0.85, ge=0.0, le=1.0)
    years_in_operation: int = Field(default=5, ge=0, le=50)
    pedigree_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    previous_subsidies_count: int = Field(default=3, ge=0, le=20)
    debt_load_ratio: float = Field(default=1.5, ge=0.0, le=10.0)

    normative: float = Field(default=15000.0, gt=0)
    direction: str = Field(default="Субсидирование в скотоводстве")
    source_system: str = Field(default="manual")
    application_date: Optional[str] = None

class DecisionRequest(BaseModel):
    application_id: str
    decision: str = Field(..., description="approved | rejected | review")
    officer_name: str
    comment: Optional[str] = None

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
    "Актюбинская область": 13, "область Абай": 14,
}

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

    log_amount = float(np.log1p(f.requested_amount))
    livestock_count = max(1.0, f.requested_amount / max(f.normative, 1))
    direction_code = float(DIRECTION_CODE_MAP.get(f.direction, 6))
    region_encoded = float(REGION_CODE_MAP.get(f.region, 7))
    is_pedigree = 1.0 if ("племен" in f.direction.lower() or
                           "племен" in f.subsidy_type.lower()) else 0.0
    is_producer = 1.0 if "производи" in f.subsidy_type.lower() else 0.0

    return {
        "gross_output_growth_yoy":     f.gross_output_growth_yoy,
        "land_to_livestock_ratio":     f.land_to_livestock_ratio,
        "historical_survival_rate":    f.historical_survival_rate,
        "subsidy_dependence_index":    f.subsidy_dependence_index,
        "veterinary_compliance":       f.veterinary_compliance,
        "years_in_operation":          float(f.years_in_operation),
        "pedigree_ratio":              f.pedigree_ratio,
        "previous_subsidies_count":    float(f.previous_subsidies_count),
        "debt_load_ratio":             f.debt_load_ratio,
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

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        missing = [item for item in pdf_items
                   if not any(item[0] in t for t in all_texts)]

        if missing and gemini_api_key:
            print(f"[gemini_ocr] локальные движки не дали текст по {len(missing)} файл(ам) — пробуем Gemini Files API…")
            ocr_text, ocr_note = await _gemini_ocr_pdfs(missing, gemini_api_key)
            extraction_note = ocr_note
            if ocr_text.strip():
                all_texts.append(ocr_text)
                print(f"[gemini_ocr] получено {len(ocr_text)} симв. через Gemini OCR")
        elif not gemini_api_key and missing:
            extraction_note = (
                f"GEMINI_API_KEY не задан — {len(missing)} файл(ов) без локального текста не обработаны облаком."
            )
            print(f"[gemini_ocr] GEMINI_API_KEY не задан — {len(missing)} сканированных файл(ов) пропущены")

        if all_texts:
            combined_text = "\n\n".join(all_texts)
            MAX_CHARS = 60_000
            combined_text_llm = combined_text[:MAX_CHARS] if len(combined_text) > MAX_CHARS else combined_text
            print(f"[docs] итого текст: {len(combined_text)} → {len(combined_text_llm)} симв. для LLM")

            if gemini_api_key:

                extraction = extract_features_from_documents(combined_text_llm, gemini_api_key)

                if extraction["extraction_status"] == "success":
                    doc_features = extraction["features"]
                    llm_summary = extraction.get("llm_summary")

                    for llm_field, our_field in [
                        ("veterinary_compliance", "veterinary_compliance"),
                        ("pedigree_ratio",        "pedigree_ratio"),
                        ("livestock_count",       "livestock_count"),
                        ("years_in_operation",    "years_in_operation"),
                    ]:
                        val = doc_features.get(llm_field)
                        if val is not None:
                            feature_dict[our_field] = float(val)

                    if doc_features.get("has_vet_passport") == 1.0:
                        feature_dict["veterinary_compliance"] = min(
                            1.0, feature_dict["veterinary_compliance"] + 0.05
                        )

    result = engine.score_farmer(feature_dict, llm_context=llm_summary)
    base_score = result["score"]

    compliance_report   = None
    final_score         = base_score
    score_doc           = None
    ml_weight           = 1.0
    doc_weight          = 0.0
    manual_review_flag  = False

    if combined_text.strip():
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        compliance_report = run_compliance_check(
            documents_text=combined_text_llm,
            subsidy_name=features.subsidy_type,
            direction=features.direction,
            gemini_api_key=gemini_api_key if gemini_api_key else None,
        )

        score_doc = float(compliance_report.get("overall_score_pct", 50.0))

        doc_completeness = compliance_report.get("doc_completeness")
        if doc_completeness is None:
            char_count = len(combined_text)
            doc_completeness = min(1.0, char_count / 8000)

        if doc_completeness >= 0.70:
            ml_weight, doc_weight = 0.55, 0.45
        elif doc_completeness >= 0.40:
            ml_weight, doc_weight = 0.75, 0.25
        else:
            ml_weight, doc_weight = 0.90, 0.10
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
            "Текст из PDF не извлечён: возможно только изображения-сканы без слоя текста и квота Gemini исчерпана."
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
        "zone_label":             final_zone_label,
        "recommendation":         final_recommendation,

        "verdict":                result["verdict"],
        "top_positive_factors":   result["top_positive_factors"],
        "top_negative_factors":   result["top_negative_factors"],
        "all_shap_values":        result["all_shap_values"],
        "shap_base_value":        result.get("shap_base_value"),
        "raw_features_used":      result.get("raw_features_used", {}),

        "compliance": compliance_report,

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
        "officer_name":    decision_req.officer_name,
        "comment":         decision_req.comment,
        "decided_at":      datetime.now().isoformat(),
    }
    _decisions_db.append(record)

    for app in _applications_db:
        if app.get("application_id") == decision_req.application_id:
            app.update({
                "decision":        decision_req.decision,
                "officer_name":    decision_req.officer_name,
                "decided_at":      record["decided_at"],
                "officer_comment": decision_req.comment,
            })
            _persist_application_update(app)
            break

    return record

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
