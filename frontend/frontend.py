
import json
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.shap_integration import generate_gemini_expert_opinion

def _load_env_vars():
    env_path = ROOT_DIR / ".env"

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

_load_env_vars()

API_BASE = os.getenv("SMARTAGRO_API_BASE", "http://localhost:8001")
API_KEY = os.getenv("SMARTAGRO_API_KEY", "sk-msgov-2025-demo-key-abc123")
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(
    page_title="SmartAgro Score | МСХ РК",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALID_REGIONS = [
    "Алматинская область", "Акмолинская область", "Атырауская область", "Восточно-Казахстанская область",
    "Жамбылская область", "Карагандинская область", "Костанайская область", "Кызылординская область",
    "Мангистауская область", "Павлодарская область", "Северо-Казахстанская область",
    "Туркестанская область", "Западно-Казахстанская область", "Актюбинская область",
    "область Абай",
]
VALID_AKIMATS = [
    "Акимат г. Алматы", "Акимат г. Астаны", "Акимат Шыркент", "Акимат Тараз",
    "Акимат г. Шымкент", "Акимат г. Караганда", "Акимат г. Актобе",
]
VALID_DIRECTIONS = [
    "Мясное", "Молочное", "Овцеводство", "Птицеводство",
    "Свиноводство", "Коневодство", "Верблюдоводство",
]
VALID_SUBSIDY_NAMES = [
    "Субсидирование племенного КРС",
    "Субсидирование молочного стада",
    "Субсидирование овцеводства",
    "Субсидирование птицеводства",
    "Субсидирование кормовых культур",
]
VALID_DISTRICTS = [
    "Алматинский район", "Шуский район", "Талгарский район", "Карасайский район",
    "Енбекшиказахский район", "Илийский район", "Уйгурский район",
]

FEATURE_NAMES_ORDER = [
    "Дата поступления",
    "Область",
    "Акимат",
    "Направление водства",
    "Наименование субсидирования",
    "Норматив",
    "Причитающая сумма",
    "Район хозяйства",
]

FEATURE_LABELS_SHORT = {
    "Дата поступления": "Дата подачи",
    "Область": "Область",
    "Акимат": "Акимат",
    "Направление водства": "Направление",
    "Наименование субсидирования": "Тип субсидии",
    "Норматив": "Норматив",
    "Причитающая сумма": "Сумма к выплате",
    "Район хозяйства": "Район",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&family=Montserrat:wght@400;500;600;700;800&display=swap');

:root {
    --primary:     #003580;
    --primary-lt:  #1a4d9e;
    --accent:      #0072CE;
    --gold:        #C8952A;
    --success:     #1a7a4a;
    --success-bg:  #d4f0e0;
    --warn:        #b36200;
    --warn-bg:     #fff3cd;
    --danger:      #b5001f;
    --danger-bg:   #fde8eb;
    --bg:          #f4f6fb;
    --surface:     #ffffff;
    --border:      #dde3ef;
    --text:        #1a2340;
    --text-muted:  #6b7a99;
    --sidebar-bg:  #002166;
}

html, body, [class*="css"] {
    font-family: 'Golos Text', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* ── Сайдбар ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 3px solid var(--gold);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #cdd8f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Шапка ── */
.gov-header {
    background: linear-gradient(135deg, #002166 0%, #003580 60%, #0050b3 100%);
    border-bottom: 4px solid var(--gold);
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 20px rgba(0,53,128,0.25);
}
.gov-header .logo-text {
    font-family: 'Montserrat', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
    line-height: 1.1;
}
.gov-header .logo-sub {
    font-size: 13px;
    color: #a8c4f0;
    font-weight: 400;
    margin-top: 3px;
}
.gov-header .badge {
    background: var(--gold);
    color: #1a1200;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    margin-left: auto;
}

/* ── Метрики ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .m-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.metric-card .m-value {
    font-family: 'Montserrat', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: var(--primary);
    line-height: 1.1;
    margin-top: 4px;
}
.metric-card .m-delta {
    font-size: 13px;
    margin-top: 6px;
    color: var(--text-muted);
}

/* ── Таблица шорт-листа ── */
.shortlist-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.shortlist-table th {
    background: var(--primary);
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.shortlist-table td { padding: 9px 14px; border-bottom: 1px solid var(--border); }
.shortlist-table tr:hover td { background: #f0f4ff; }

.row-green td { background: #e8f8f0 !important; border-left: 4px solid var(--success) !important; }
.row-yellow td { background: #fffbea !important; border-left: 4px solid #e8a800 !important; }
.row-red   td { background: #fff0f2 !important; border-left: 4px solid var(--danger) !important; }
.row-cutoff td { opacity: 0.55; background: #f8f8f8 !important; border-left: 4px solid #aaa !important; }

.score-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
}
.pill-green  { background: var(--success-bg); color: var(--success); }
.pill-yellow { background: var(--warn-bg);    color: var(--warn); }
.pill-red    { background: var(--danger-bg);  color: var(--danger); }

/* ── SHAP ── */
.shap-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
}
.shap-pos { background: #e8f8f0; border-left: 4px solid var(--success); }
.shap-neg { background: #fff0f2; border-left: 4px solid var(--danger); }
.shap-val { font-weight: 700; min-width: 60px; }
.shap-pos .shap-val { color: var(--success); }
.shap-neg .shap-val { color: var(--danger); }

/* ── Compliance чеклист ── */
.compliance-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-top: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.compliance-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.compliance-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 14px;
    color: var(--primary);
}
.compliance-badge {
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
}
.badge-ok    { background: var(--success-bg); color: var(--success); }
.badge-warn  { background: var(--warn-bg);    color: var(--warn); }
.badge-fail  { background: var(--danger-bg);  color: var(--danger); }
.badge-disq  { background: #2d0010; color: #ff6b8a; }

.check-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 7px;
    margin-bottom: 5px;
    font-size: 13px;
    line-height: 1.4;
}
.check-ok   { background: #f0faf5; border-left: 3px solid var(--success); }
.check-fail { background: #fff2f4; border-left: 3px solid var(--danger); }
.check-warn { background: #fffbea; border-left: 3px solid #e8a800; }

.check-emoji { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
.check-text  { flex: 1; }
.check-label { font-weight: 600; color: var(--text); }
.check-evidence { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.check-source { font-size: 10px; color: #aab; font-style: italic; }
.check-critical { font-size: 10px; font-weight: 700; color: var(--danger); margin-left: 4px; }

.score-breakdown {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f4f6fb;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 13px;
}
.score-ml   { color: var(--text-muted); }
.score-arrow { color: #aaa; }
.score-bonus { font-weight: 700; }
.score-bonus.pos { color: var(--success); }
.score-bonus.neg { color: var(--danger); }
.score-final { font-family:'Montserrat',sans-serif; font-weight:800; font-size:18px; color: var(--primary); }

/* ── Кнопки Human-in-the-loop ── */
.hitl-block {
    background: linear-gradient(135deg, #f0f4ff, #e8f0ff);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 24px;
}
.hitl-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    color: var(--primary);
    font-size: 15px;
    margin-bottom: 4px;
}
.hitl-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 14px; }

/* ── Бюджетная черта ── */
.budget-cutoff-row {
    text-align: center;
    font-size: 13px;
    font-weight: 700;
    color: #555;
    background: linear-gradient(90deg, transparent, #ddd 20%, #ddd 80%, transparent);
    padding: 6px;
    letter-spacing: 0.5px;
}

/* ── Инфо-блоки ── */
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.info-box h4 { color: var(--primary); font-weight: 700; margin-bottom: 10px; font-size: 15px; }

/* ── Code ── */
.code-block {
    background: #0f1a2e;
    color: #7de3a0;
    padding: 16px 20px;
    border-radius: 8px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    overflow-x: auto;
    border: 1px solid #1e3060;
    line-height: 1.6;
}

/* ── Статус ГИСС ── */
.giss-status {
    display: flex; align-items: center; gap: 8px;
    background: #e8f8f0; border: 1px solid #a3dfc0;
    border-radius: 8px; padding: 10px 16px; font-size: 13px;
    color: var(--success); font-weight: 600;
}
.giss-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--success);
    animation: pulse 1.5s infinite; display: inline-block; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

/* ── Streamlit override ── */
div[data-testid="stTabs"] button {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
div.stButton > button {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    transition: all .2s !important;
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    border-radius: 8px !important;
    border: 1.5px solid var(--border) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

if "applications" not in st.session_state:
    st.session_state.applications = []
if "selected_app_id" not in st.session_state:
    st.session_state.selected_app_id = None
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "api_key_display" not in st.session_state:
    st.session_state.api_key_display = API_KEY

def _api_post(
    endpoint: str,
    payload: dict,
    *,
    timeout: float | tuple[float, float] = 10,
) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            headers=HEADERS,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        st.error(
            "⏱️ Сервер не ответил в срок. Убедитесь, что uvicorn запущен, и попробуйте снова."
        )
        return None
    except Exception as e:
        st.error(f"Ошибка API: {e}")
        return None

def _api_post_multipart(endpoint: str, data: dict, files: list[tuple[str, tuple[str, bytes, str]]]) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            data=data,
            files=files,
            headers=HEADERS,

            timeout=(10, 300),
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        st.error(
            "⏱️ Сервер не успел обработать документы за 5 минут. "
            "Попробуйте загрузить меньше файлов или уменьшить их размер."
        )
        return None
    except Exception as e:
        st.error(f"Ошибка API: {e}")
        return None

def _api_get(endpoint: str) -> dict | list | None:
    try:

        _to = 120 if endpoint.rstrip("/").endswith("applications") else 10
        r = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, timeout=_to)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка API: {e}")
        return None

def _score_pill(score: float, category: str) -> str:
    cls = {"green": "pill-green", "yellow": "pill-yellow", "red": "pill-red"}.get(category, "")
    return f'<span class="score-pill {cls}">{score:.0f}</span>'

def _fmt_tenge(val: float) -> str:
    return f"{val/1_000_000:.2f} млн ₸"

FEAT_LABELS_SHAP = {
    "gross_output_growth_yoy": "Рост продукции",
    "pedigree_ratio": "Племенное поголовье",
    "historical_survival_rate": "Выживаемость стада",
    "veterinary_compliance": "Ветеринария",
    "debt_load_ratio": "Долговая нагрузка",
    "subsidy_dependence_index": "Независимость от субсидий",
    "land_to_livestock_ratio": "Обеспеченность землёй",
    "years_in_operation": "Стаж работы",
    "previous_subsidies_count": "История субсидий",
    "livestock_count": "Поголовье (расч.)",
    "log_amount": "Масштаб заявки",
    "direction_code": "Направление",
    "region_encoded": "Регион",
    "is_pedigree": "Племенная субсидия",
    "is_producer": "Производители",
    "hour_submitted": "Час подачи",
    "month_submitted": "Месяц подачи",
}

def _shap_max_abs(all_shap: dict) -> float:
    if not all_shap:
        return 1e-9
    return max(abs(float(v)) for v in all_shap.values()) or 1e-9

def _shap_to_display_points(shap_val: float, max_abs: float, scale: float = 20.0) -> int:
    if max_abs < 1e-9:
        return 0
    return int(round(scale * float(shap_val) / max_abs))

def _normalized_profile_from_raw(raw: dict) -> dict[str, float]:
    if not raw:
        return {}
    g = float(raw.get("gross_output_growth_yoy", 0.0))
    growth_norm = float(np.clip((g + 0.35) / 1.0 * 100, 0, 100))
    pr = float(raw.get("pedigree_ratio", 0.5))
    land = float(raw.get("land_to_livestock_ratio", 2.0))
    surv = float(raw.get("historical_survival_rate", 0.85))
    debt = float(raw.get("debt_load_ratio", 1.5))
    sub = float(raw.get("subsidy_dependence_index", 0.3))
    years = float(raw.get("years_in_operation", 5.0))
    vet = float(raw.get("veterinary_compliance", 0.85))
    return {
        "Рост продукции": growth_norm,
        "Племенное поголовье": pr * 100,
        "Земля": min(land / 8.0 * 100, 100),
        "Выживаемость": surv * 100,
        "Долг (инверт.)": max(0.0, (1.0 - debt / 5.0)) * 100,
        "Независимость": (1.0 - sub) * 100,
        "Стаж": min(years / 20.0 * 100, 100),
        "Ветеринария": vet * 100,
    }

def _radar_order_labels() -> list[str]:
    return [
        "Рост продукции",
        "Племенное поголовье",
        "Земля",
        "Выживаемость",
        "Долг (инверт.)",
        "Независимость",
        "Стаж",
        "Ветеринария",
    ]

def _plot_score_summary_no_waterfall(app: dict, score_ml: float):
    base = app.get("shap_base_value")
    all_shap = app.get("all_shap_values") or {}
    if base is None and not all_shap:
        return
    c1, c2, c3 = st.columns(3)
    if base is not None:
        c1.metric("Базовое ожидание модели", f"{float(base):.1f} б.")
    if all_shap:
        s = sum(float(v) for v in all_shap.values())
        c2.metric("Сумма вкладов SHAP (Σ)", f"{s:+.1f} б.")
    c3.metric("Балл XGBoost (ML)", f"{score_ml:.1f} / 100")
    st.caption(
        "Итоговый балл модели ≈ базовое ожидание + сумма вкладов признаков (SHAP). "
        "Документы и заключение LLM могут влиять на отдельный бонус соответствия."
    )

def _refresh_apps():
    data = _api_get("/api/v1/applications")
    if data is not None:
        st.session_state.applications = data

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:40px;">🌾</div>
        <div style="font-family:'Montserrat',sans-serif; font-size:18px; font-weight:800; color:#fff; margin-top:6px;">
            SmartAgro Score
        </div>
        <div style="font-size:11px; color:#8ab4e8; margin-top:4px; letter-spacing:0.5px;">
            МСХ РК | Система скоринга субсидий
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**🔐 Авторизованный пользователь**")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.08); padding:10px 14px; border-radius:8px; font-size:13px;">
        👤 Минсельхоз<br>
        <span style="color:#8ab4e8; font-size:12px;">Министерство сельского хозяйства РК</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    total = len(st.session_state.applications)
    green = sum(1 for a in st.session_state.applications if a.get("zone") == "green")
    yellow = sum(1 for a in st.session_state.applications if a.get("zone") == "yellow")
    red = sum(1 for a in st.session_state.applications if a.get("zone") == "red")

    st.markdown("**📊 Статистика очереди**")
    st.markdown(f"""
    <div style="font-size:13px; line-height:2;">
        📋 Всего заявок: <b>{total}</b><br>
        🟢 Рекомендовано: <b>{green}</b><br>
        🟡 На рассмотрении: <b>{yellow}</b><br>
        🔴 Не рекомендовано: <b>{red}</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<span style='font-size:11px; color:#8ab4e8;'>🕐 Последнее обновление: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)

st.markdown("""
<div class="gov-header">
    <div>
        <div class="logo-text">🌾 SmartAgro Score</div>
        <div class="logo-sub">Информационно-аналитическая система merit-based скоринга субсидий</div>
    </div>
    <div class="badge">ВНУТРЕННИЙ ПОРТАЛ МСХ РК</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Поток заявок",
    "📊 Шорт-лист и бюджет",
    "🔍 Профиль фермера (XAI)",
    "🔌 Интеграция API",
])

with tab1:
    st.markdown("### 📥 Поток заявок — API и ручная загрузка")

    col_sync, col_status = st.columns([1, 2])

    with col_sync:
        if st.button("🧪 Тестовые заявки", type="primary", use_container_width=True):
            with st.spinner("Загрузка тестовых заявок…"):
                time.sleep(1.2)
                data = _api_post(
                    "/api/v1/giss/sync",
                    {},
                    timeout=(10, 120),
                )
                if data:
                    st.success(f"✅ Добавлено тестовых заявок: {data['synced_count']}")
                    _refresh_apps()
                    st.rerun()

    with col_status:
        st.markdown("""
        <div class="giss-status">
            <span class="giss-dot"></span>
            API скоринга: онлайн | eGov: онлайн | Тестовые заявки — по кнопке слева
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.expander("✏️ Подать заявку на скоринг", expanded=True):

        st.markdown("### 🅰️ Вариант А — Основные данные")
        st.caption("Обязательно для заполнения. На основе этих данных рассчитывается базовый скоринговый балл.")

        a1, a2, a3 = st.columns(3)
        with a1:
            man_bin = st.text_input("БИН / ИИН предприятия *", placeholder="123456789012")
            man_company = st.text_input("Наименование предприятия *", placeholder="ТОО «Агро-Нур»")
        with a2:
            man_region = st.selectbox("Область *", [
                "Алматинская область", "Акмолинская область", "Атырауская область",
                "Восточно-Казахстанская область", "Жамбылская область",
                "Карагандинская область", "Костанайская область", "Кызылординская область",
                "Мангистауская область", "Павлодарская область", "Северо-Казахстанская область",
                "Туркестанская область", "Западно-Казахстанская область",
                "Актюбинская область", "область Абай",
            ])
            man_direction = st.selectbox("Направление животноводства *", VALID_DIRECTIONS)
        with a3:
            man_subsidy = st.selectbox("Вид субсидии *", [
                "Субсидия на племенное маточное поголовье КРС",
                "Субсидия на быков-производителей",
                "Удешевление производства молока",
                "Субсидия на племенных овец / баранов-производителей",
                "Субсидия на коневодство",
                "Субсидия на верблюдоводство",
                "Субсидия на свиноводство",
                "Субсидия на птицеводство",
                "Иное",
            ])
            man_amount = st.number_input(
                "Запрашиваемая сумма субсидии (тенге) *",
                min_value=100_000, max_value=500_000_000,
                value=10_000_000, step=500_000,
            )

        st.divider()

        st.markdown("### 🅱️ Вариант Б — Уточняющие данные *(опционально)*")
        st.caption(
            "Заполнение не обязательно — если оставить «Не знаю / Не указано», "
            "модель применит статистические дефолты. "
            "Чем точнее данные — тем точнее итоговый балл."
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            farm_size = st.selectbox(
                "Размер хозяйства *(опционально)*",
                ["Не указано", "Малое (до 50 голов / до 100 га)", "Среднее (50–500 голов / 100–1000 га)", "Крупное (500+ голов / 1000+ га)"],
            )
            debt_level = st.selectbox(
                "Долговая нагрузка *(опционально)*",
                ["Не знаю", "Низкая — Долг/EBITDA < 1.5", "Умеренная — Долг/EBITDA 1.5–3.0", "Высокая — Долг/EBITDA > 3.0"],
            )
        with b2:
            subsidy_exp = st.selectbox(
                "Опыт участия в субсидировании *(опционально)*",
                ["Не знаю", "Нет — подаю впервые", "1–2 раза ранее", "3 и более раз"],
            )
            vet_status = st.selectbox(
                "Ветеринарное состояние хозяйства *(опционально)*",
                ["Не знаю", "Нарушений нет — все справки актуальны", "Есть незначительные замечания", "Есть серьёзные нарушения / запреты"],
            )
        with b3:
            growth_choice = st.selectbox(
                "Динамика производства за прошлый год *(опционально)*",
                ["Не знаю", "Спад (< 0%)", "Без изменений (0–5%)", "Умеренный рост (5–20%)", "Высокий рост (> 20%)"],
            )
            pedigree_choice = st.selectbox(
                "Доля племенного поголовья *(опционально)*",
                ["Не знаю", "Нет или менее 20%", "20–60%", "60–90%", "Более 90%"],
            )

        st.divider()

        st.markdown("### 📂 Документы заявки")
        st.caption(
            "Перетащите PDF-файлы или нажмите «Browse files». "
            "LLM проанализирует каждый документ на соответствие правилам субсидирования МСХ РК "
            "и уточнит признаки для XGBoost-модели. **Лимит: 200 МБ суммарно.**"
        )

        uploaded_docs = st.file_uploader(
            "Загрузите любые документы заявки — ветсправки, ЭСФ, землеустройство, банковские выписки и т.д.",
            type=["pdf"],
            accept_multiple_files=True,
            key="bulk_docs",
        )

        if uploaded_docs:
            total_mb = sum(len(d.getvalue()) for d in uploaded_docs) / 1_048_576
            if total_mb > 200:
                st.error(
                    f"❌ Суммарный объём {total_mb:.1f} МБ превышает лимит 200 МБ. "
                    "Удалите часть файлов и попробуйте снова."
                )
                uploaded_docs = []
            else:
                st.success(f"✅ {len(uploaded_docs)} файл(ов) готово к отправке — {total_mb:.1f} МБ / 200 МБ")
                with st.expander("Список загруженных файлов", expanded=False):
                    for d in uploaded_docs:
                        sz_kb = len(d.getvalue()) / 1024
                        st.caption(f"📄 {d.name} — {sz_kb:.1f} КБ")

        if st.button("📤 Отправить заявку на скоринг", type="primary", use_container_width=True):
            if not man_bin.strip() or not man_company.strip():
                st.warning("⚠️ Заполните обязательные поля Варианта А: БИН/ИИН и наименование предприятия.")
            else:

                _debt_map = {
                    "Не знаю":                           1.5,
                    "Низкая — Долг/EBITDA < 1.5":        0.8,
                    "Умеренная — Долг/EBITDA 1.5–3.0":   2.2,
                    "Высокая — Долг/EBITDA > 3.0":       3.8,
                }
                _vet_map = {
                    "Не знаю":                                      0.85,
                    "Нарушений нет — все справки актуальны":        0.97,
                    "Есть незначительные замечания":                0.72,
                    "Есть серьёзные нарушения / запреты":           0.45,
                }
                _growth_map = {
                    "Не знаю":                 0.05,
                    "Спад (< 0%)":            -0.12,
                    "Без изменений (0–5%)":    0.03,
                    "Умеренный рост (5–20%)":  0.12,
                    "Высокий рост (> 20%)":    0.28,
                }
                _pedigree_map = {
                    "Не знаю":           0.50,
                    "Нет или менее 20%": 0.10,
                    "20–60%":            0.40,
                    "60–90%":            0.75,
                    "Более 90%":         0.95,
                }
                _subsidy_exp_map = {
                    "Не знаю":             3,
                    "Нет — подаю впервые": 0,
                    "1–2 раза ранее":      1,
                    "3 и более раз":       5,
                }
                _farm_years_map = {
                    "Не указано":                            5,
                    "Малое (до 50 голов / до 100 га)":       3,
                    "Среднее (50–500 голов / 100–1000 га)":  7,
                    "Крупное (500+ голов / 1000+ га)":      15,
                }

                payload = {
                    "bin_iin":                  man_bin.strip(),
                    "company_name":             man_company.strip(),
                    "region":                   man_region,
                    "subsidy_type":             man_subsidy,
                    "direction":                man_direction,
                    "requested_amount":         man_amount,
                    "source_system":            "manual",

                    "debt_load_ratio":          _debt_map.get(debt_level, 1.5),
                    "veterinary_compliance":    _vet_map.get(vet_status, 0.85),
                    "gross_output_growth_yoy":  _growth_map.get(growth_choice, 0.05),
                    "pedigree_ratio":           _pedigree_map.get(pedigree_choice, 0.50),
                    "previous_subsidies_count": _subsidy_exp_map.get(subsidy_exp, 3),
                    "years_in_operation":       _farm_years_map.get(farm_size, 5),
                    "normative":                15_000.0,
                }

                if uploaded_docs:
                    total_mb_chk = sum(len(d.getvalue()) for d in uploaded_docs) / 1_048_576
                    if total_mb_chk > 200:
                        st.error("❌ Суммарный размер файлов превышает 200 МБ — уменьшите набор документов.")
                    else:
                        files = [
                            ("documents", (d.name, d.getvalue(), "application/pdf"))
                            for d in uploaded_docs
                        ]
                        with st.spinner(
                            f"🧠 LLM анализирует {len(uploaded_docs)} документ(ов) "
                            f"+ XGBoost скоринг — может занять 15–30 секунд…"
                        ):
                            result = _api_post_multipart(
                                "/api/v1/score-with-documents",
                                data={"features_json": json.dumps(payload, ensure_ascii=False)},
                                files=files,
                            )
                else:
                    with st.spinner("⚙️ XGBoost скоринг (без документов)…"):
                        result = _api_post("/api/v1/score", payload)

                if result:
                    _z  = result.get("zone", "yellow")
                    _ic = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(_z, "⚪")
                    _w_ml  = result.get("ml_weight_used",  1.0)
                    _w_doc = result.get("doc_weight_used", 0.0)
                    _score_ml  = result.get("score_ml",  result.get("score", 0))
                    _score_doc = result.get("score_doc", None)
                    _manual    = result.get("manual_review_required", False)

                    score_detail = f"**{result['score']:.0f}** / 100"
                    if _w_doc > 0:
                        score_detail += (
                            f" _(ML {_score_ml:.0f}×{_w_ml:.0%}"
                            + (f" + Документы {_score_doc:.0f}×{_w_doc:.0%}" if _score_doc is not None else "")
                            + ")_"
                        )

                    st.success(
                        f"{_ic} Заявка принята. ID: **{result['application_id']}** | "
                        f"Итоговый балл: {score_detail}"
                    )
                    _chars = result.get("documents_text_chars") or 0
                    _ext_note = result.get("documents_extraction_note")
                    if uploaded_docs and _chars > 0:
                        st.caption(f"В ответе сервера: **{_chars:,}** симв. текста из PDF — на вкладке «Профиль фермера» выберите эту заявку по ID.")
                    elif uploaded_docs:
                        st.warning(
                            "⚠️ Файлы отправлены, но полный текст из PDF не сохранён. "
                            + (_ext_note if _ext_note else "Проверьте логи uvicorn (квота LLM API 429, сканы без текста).")
                        )
                    st.session_state.selected_app_id = result["application_id"]
                    if _manual:
                        st.warning(
                            "⚠️ Документы содержат мало машиночитаемых данных. "
                            "Рекомендуется ручная проверка комиссией."
                        )
                    _refresh_apps()
                    st.rerun()

    st.divider()
    st.markdown("#### 📋 Все заявки в очереди")

    _refresh_apps()

    if not st.session_state.applications:
        st.info("Заявок нет. Нажмите «Тестовые заявки» или добавьте вручную.")
    else:
        for app in st.session_state.applications:
            cat = app.get("zone", "yellow")
            icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(cat, "⚪")
            score = app.get("score", 0)
            decision_status = st.session_state.decisions.get(app["application_id"], "")
            dec_badge = ""
            if decision_status == "approved":
                dec_badge = "✅ Одобрено"
            elif decision_status == "rejected":
                dec_badge = "❌ Отказано"

            with st.container():
                c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
                with c1:
                    _demo_badge = " **[тест]**" if app.get("is_demo") else ""
                    st.markdown(f"{icon} **{app['company_name']}**{_demo_badge} `{app['bin_iin']}`")
                with c2:
                    st.caption(f"{app['region']} | {app['subsidy_type'][:30]}...")
                with c3:
                    st.caption(f"{_fmt_tenge(app['requested_amount'])}")
                with c4:
                    st.markdown(f"**{score:.0f}** / 100")
                with c5:
                    if dec_badge:
                        st.caption(dec_badge)
                    elif st.button("Открыть", key=f"open_{app['application_id']}"):
                        st.session_state.selected_app_id = app["application_id"]
                        st.info("Перейдите на вкладку '🔍 Профиль фермера (XAI)'")

with tab2:
    st.markdown("### 📊 Шорт-лист и распределение бюджета")

    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        budget = st.number_input(
            "💰 Бюджет транша (тенге)",
            min_value=1_000_000,
            max_value=10_000_000_000,
            value=100_000_000,
            step=5_000_000,
            help="Введите доступный бюджет для данного транша субсидий",
        )
        budget_mln = budget / 1_000_000
        st.markdown(f"<div style='font-size:20px; font-weight:800; color:#003580;'>= {budget_mln:.1f} млн ₸</div>",
                    unsafe_allow_html=True)

    with col_b2:
        apps = st.session_state.applications
        if apps:
            total_sum = sum(a.get("requested_amount", 0) for a in apps)
            approved_count = sum(1 for a in apps if a.get("score", 0) >= 50)
            coverage = min(budget / total_sum * 100, 100) if total_sum > 0 else 0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="m-label">Всего заявлено</div>
                    <div class="m-value">{_fmt_tenge(total_sum)}</div>
                    <div class="m-delta">{len(apps)} заявок</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="m-label">Покрытие бюджетом</div>
                    <div class="m-value">{coverage:.0f}%</div>
                    <div class="m-delta">от общего объёма</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="m-label">Средний балл</div>
                    <div class="m-value">{np.mean([a['score'] for a in apps]):.0f}</div>
                    <div class="m-delta">по всей очереди</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    apps_sorted = sorted(st.session_state.applications, key=lambda x: x.get("score", 0), reverse=True)

    if not apps_sorted:
        st.info("Заявок нет. Загрузите тестовые заявки или добавьте вручную.")
    else:

        rows_html = ""
        cumulative = 0.0
        cutoff_drawn = False

        for app in apps_sorted:
            app_id = app["application_id"]
            score = app.get("score", 0)
            cat = app.get("zone", "yellow")
            amt = app.get("requested_amount", 0)
            decision = st.session_state.decisions.get(app_id, "")

            would_exceed = (cumulative + amt) > budget

            if would_exceed and not cutoff_drawn:
                remaining = budget - cumulative
                rows_html += f"""
                <tr>
                    <td colspan="7" class="budget-cutoff-row">
                        ─── БЮДЖЕТ ИСЧЕРПАН ({_fmt_tenge(budget)}) ──
                        Остаток: {_fmt_tenge(remaining)} | Заявки ниже не попадают в транш ───
                    </td>
                </tr>"""
                cutoff_drawn = True

            if not would_exceed:
                cumulative += amt

            row_class = "row-green" if cat == "green" else ("row-yellow" if cat == "yellow" else "row-red")
            if would_exceed and cutoff_drawn:
                row_class = "row-cutoff"

            pill = _score_pill(score, cat)
            dec_cell = ""
            if decision == "approved":
                dec_cell = "<span style='color:green; font-weight:700;'>✅ Одобрено</span>"
            elif decision == "rejected":
                dec_cell = "<span style='color:red; font-weight:700;'>❌ Отказано</span>"
            else:
                dec_cell = "<span style='color:#aaa;'>Ожидание</span>"

            date_str = app.get("application_date", app.get("calculated_at", "")[:10])
            _demo_tag = ' <small style="color:#888;">[тест]</small>' if app.get("is_demo") else ""

            rows_html += f"""
            <tr class="{row_class}">
                <td>{date_str}</td>
                <td><b>{app['company_name']}</b>{_demo_tag}<br><small style='color:#888;'>{app_id}</small></td>
                <td>{app['region']}</td>
                <td style='max-width:180px;'>{app['subsidy_type']}</td>
                <td><b>{_fmt_tenge(amt)}</b></td>
                <td>{pill}</td>
                <td>{dec_cell}</td>
            </tr>"""

        table_html = f"""
        <table class="shortlist-table">
            <thead>
                <tr>
                    <th>Дата</th>
                    <th>Предприятие</th>
                    <th>Область</th>
                    <th>Вид субсидии</th>
                    <th>Сумма</th>
                    <th>Балл</th>
                    <th>Решение</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div style="margin-top:10px; font-size:12px; color:#888;">
            🟢 80–100: строго рекомендовано &nbsp;|&nbsp;
            🟡 50–79: требует рассмотрения &nbsp;|&nbsp;
            🔴 &lt;50: не рекомендовано
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.divider()
        scores = [a["score"] for a in apps_sorted]
        fig = go.Figure()
        colors = ["#1a7a4a" if s >= 80 else ("#e8a800" if s >= 50 else "#b5001f") for s in scores]
        fig.add_trace(go.Bar(
            x=[a["company_name"][:20] for a in apps_sorted],
            y=scores,
            marker_color=colors,
            text=[f"{s:.0f}" for s in scores],
            textposition="outside",
        ))
        fig.add_hline(y=80, line_dash="dot", line_color="#1a7a4a",
                      annotation_text="Порог рекомендации (80)", annotation_position="right")
        fig.add_hline(y=50, line_dash="dot", line_color="#e8a800",
                      annotation_text="Порог рассмотрения (50)", annotation_position="right")
        fig.update_layout(
            title="Распределение скоринговых баллов",
            xaxis_title="Предприятие",
            yaxis_title="Балл (0–100)",
            height=350,
            plot_bgcolor="#f8faff",
            paper_bgcolor="#ffffff",
            font=dict(family="Golos Text", size=12),
            margin=dict(t=50, b=60),
            yaxis=dict(range=[0, 110]),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🔍 Профиль фермера — Explainable AI")

    apps = st.session_state.applications

    if not apps:
        st.info("Нет заявок для анализа. Сначала добавьте заявку вручную или нажмите «Тестовые заявки».")
    else:
        app_names = {
            a["application_id"]: (
                f"{a['company_name']} [тест] [{a['application_id']}]" if a.get("is_demo")
                else f"{a['company_name']} [{a['application_id']}]"
            )
            for a in apps
        }
        default_id = st.session_state.selected_app_id or list(app_names.keys())[0]
        if default_id not in app_names:
            default_id = list(app_names.keys())[0]

        selected_id = st.selectbox(
            "Выберите предприятие для анализа",
            options=list(app_names.keys()),
            format_func=lambda x: app_names[x],
            index=list(app_names.keys()).index(default_id),
        )
        st.session_state.selected_app_id = selected_id

        app = next((a for a in apps if a["application_id"] == selected_id), None)

        if app is None:
            st.warning("Заявка не найдена.")
        else:
            cat = app.get("zone", "yellow")
            score = app.get("score", 0)
            score_ml = app.get("score_ml", score)
            compliance_bonus = app.get("compliance_bonus", 0)
            cat_colors = {"green": "#1a7a4a", "yellow": "#b36200", "red": "#b5001f"}
            cat_bg = {"green": "#e8f8f0", "yellow": "#fffbea", "red": "#fff0f2"}
            cat_labels = {"green": "✅ СТРОГО РЕКОМЕНДОВАНО", "yellow": "⚠️ ТРЕБУЕТ РАССМОТРЕНИЯ", "red": "🚫 НЕ РЕКОМЕНДОВАНО"}

            bonus_str = ""
            if compliance_bonus != 0:
                sign = "+" if compliance_bonus > 0 else ""
                bonus_color = "#1a7a4a" if compliance_bonus > 0 else "#b5001f"
                bonus_str = f'<span style="font-size:13px; color:{bonus_color}; font-weight:700; margin-left:10px;">({sign}{compliance_bonus:.1f} от проверки документов)</span>'

            st.markdown(f"""
            <div style="background:{cat_bg.get(cat,'#f8f8f8')}; border: 2px solid {cat_colors.get(cat,'#888')};
                 border-radius:12px; padding:20px 24px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-family:'Montserrat',sans-serif; font-size:22px; font-weight:800;
                             color:#1a2340;">{app['company_name']}</div>
                        <div style="color:#6b7a99; font-size:13px; margin-top:4px;">
                            БИН/ИИН: {app['bin_iin']} &nbsp;|&nbsp; {app.get('region','')} &nbsp;|&nbsp; {app.get('subsidy_type', app.get('direction',''))}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:52px; font-weight:900; color:{cat_colors.get(cat,'#333')};
                             font-family:'Montserrat',sans-serif; line-height:1;">{score:.0f}</div>
                        <div style="font-size:11px; color:#888; letter-spacing:0.5px;">/ 100 баллов {bonus_str}</div>
                        <div style="font-size:13px; font-weight:700; color:{cat_colors.get(cat,'#333')};
                             margin-top:4px;">{cat_labels.get(cat,'')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            gemini_key = os.getenv("GEMINI_API_KEY", "")
            opinion_key = f"gemini_opinion_{selected_id}"

            if opinion_key not in st.session_state:
                st.session_state[opinion_key] = None

            _det = (app.get("documents_extracted_text") or "")
            _note = app.get("documents_extraction_note")
            if isinstance(_det, str) and len(_det.strip()) > 0:
                st.success(
                    f"📄 В заявке сохранён текст PDF: **{len(_det):,}** симв. — он будет передан в LLM для заключения."
                )
            else:
                if _note:
                    st.warning(f"**Почему нет текста:** {_note}")
                else:
                    st.info(
                        "Текст PDF для этой записи **не сохранён** (заявка без файлов, тестовая заявка без вложений или старая сессия). "
                        "Подайте заявку с PDF через «Поток заявок»."
                    )

            col_btn, col_status_g = st.columns([1, 3])
            with col_btn:
                if st.button("🤖 Получить заключение LLM", use_container_width=True, key=f"gemini_btn_{selected_id}"):
                    if gemini_key:
                        with st.spinner("LLM: анкета + сохранённый текст PDF (если есть)…"):
                            opinion = generate_gemini_expert_opinion(app, gemini_key)
                            st.session_state[opinion_key] = opinion
                    else:
                        st.warning("Установите переменную GEMINI_API_KEY для работы LLM.")

            opinion_text = st.session_state.get(opinion_key)
            if opinion_text:
                verdict_color = {"green": "#1a7a4a", "yellow": "#b36200", "red": "#b5001f"}.get(cat, "#333")
                st.markdown(f"""
                <div style="background:#f9faff; border:1.5px solid {verdict_color};
                     border-left: 5px solid {verdict_color};
                     border-radius:10px; padding:20px 24px; margin: 12px 0 20px 0;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                        <span style="font-size:20px;">🤖</span>
                        <span style="font-family:'Montserrat',sans-serif; font-weight:700;
                             font-size:14px; color:{verdict_color};">
                            Экспертное заключение — LLM
                        </span>
                        <span style="font-size:11px; color:#aaa; margin-left:auto;">
                            на основе Правил субсидирования МСХ РК (Приказ № 108)
                        </span>
                    </div>
                    <div style="font-size:14px; line-height:1.8; color:#1a2340; white-space:pre-wrap;">{opinion_text}</div>
                </div>
                """, unsafe_allow_html=True)

            prof_col1, prof_col2 = st.columns([3, 2])

            with prof_col1:
                raw_features = app.get("raw_features_used") or {}
                pdf_ok = bool((app.get("documents_extracted_text") or "").strip())
                st.markdown("#### 📈 Показатели и условная динамика")
                st.caption(
                    "Графики строятся по **числовым признакам**, переданным в модель скоринга "
                    "(анкета заявки; при успешном разборе PDF — уточнённые значения). "
                    "Это не внешняя бухгалтерская отчётность, а входные данные расчёта по заявке."
                )
                if pdf_ok:
                    st.caption("📄 Для этой заявки сохранён текст PDF — он мог использоваться при извлечении признаков и в заключении LLM.")

                prof = _normalized_profile_from_raw(raw_features)
                bar_labels = list(prof.keys())
                bar_vals = [prof[k] for k in bar_labels]
                fig_bars = go.Figure(go.Bar(
                    x=bar_vals,
                    y=bar_labels,
                    orientation="h",
                    marker_color="#0072CE",
                    text=[f"{v:.0f}" for v in bar_vals],
                    textposition="outside",
                ))
                fig_bars.update_layout(
                    title="Ключевые показатели (нормализация 0–100 для сравнения)",
                    height=max(280, len(bar_labels) * 28),
                    plot_bgcolor="#f8faff",
                    paper_bgcolor="#ffffff",
                    margin=dict(t=50, b=30, l=10, r=50),
                    font=dict(family="Golos Text", size=11),
                    xaxis=dict(range=[0, 110], title="Шкала, б.н."),
                )
                st.plotly_chart(fig_bars, use_container_width=True)

                yoy = float(raw_features.get("gross_output_growth_yoy", 0.0))
                cy = datetime.now().year
                idx_prev, idx_curr = 100.0, 100.0 * (1.0 + yoy)
                fig_yoy = go.Figure()
                fig_yoy.add_trace(go.Scatter(
                    x=[cy - 1, cy],
                    y=[idx_prev, idx_curr],
                    name="Условный индекс (база 100)",
                    line=dict(color="#003580", width=3),
                    mode="lines+markers",
                    marker=dict(size=10),
                ))
                fig_yoy.update_layout(
                    title="Условная динамика по росту валовой продукции (YoY из заявки)",
                    height=240,
                    plot_bgcolor="#f8faff",
                    paper_bgcolor="#ffffff",
                    margin=dict(t=50, b=30, l=40, r=20),
                    font=dict(family="Golos Text", size=11),
                    yaxis_title="Индекс (условн.)",
                    xaxis_title="Год",
                    showlegend=False,
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

                radar_order = _radar_order_labels()
                radar_vals = [prof.get(lab, 0.0) for lab in radar_order]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_order + [radar_order[0]],
                    fill="toself",
                    fillcolor="rgba(0,114,206,0.18)",
                    line=dict(color="#0072CE", width=2.5),
                ))
                fig_radar.update_layout(
                    title="Профиль эффективности (те же данные, что в модели)",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=340, paper_bgcolor="#ffffff",
                    font=dict(family="Golos Text", size=11),
                    margin=dict(t=50),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with prof_col2:

                st.markdown("#### 🧠 Объяснение AI-решения (модель + SHAP)")
                st.markdown("""
                <div style="background:#f0f4ff; border:1px solid #dde3ef; border-radius:8px;
                     padding:10px 14px; font-size:12px; color:#6b7a99; margin-bottom:12px;">
                    <b>Как это связано с PDF и скорингом:</b> балл считает <b>XGBoost</b> по числовым признакам заявки
                    (часть значений может быть уточнена из текста PDF). <b>SHAP</b> показывает вклад каждого признака
                    в итоговый балл. Ниже вклад переведён в шкалу <b>±20 баллов</b> относительно самого сильного фактора
                    в этой заявке. Заключение <b>LLM</b> (кнопка выше) дополняет проверкой правил и контекстом PDF.
                </div>
                """, unsafe_allow_html=True)
                _dc = int(app.get("documents_text_chars") or 0)
                if _dc > 0:
                    st.caption(f"Документы: извлечено {_dc} симв. текста из PDF (для compliance и уточнения признаков).")
                elif app.get("documents_pdf_count", 0):
                    st.caption("Документы: PDF загружались, но машиночитаемый текст не получен (возможны сканы).")
                else:
                    st.caption("Документы: к этой записи файлы не прикреплялись или это тестовая заявка без вложений.")

                top_pos = app.get("top_positive_factors", [])
                top_neg = app.get("top_negative_factors", [])
                expl_by_feature = {item["feature"]: item for item in (top_pos + top_neg) if item.get("feature")}

                all_shap = app.get("all_shap_values") or {}
                max_abs = _shap_max_abs(all_shap)
                raw_for_shap = app.get("raw_features_used") or {}

                shap_html = ""
                if all_shap:
                    ordered = sorted(all_shap.items(), key=lambda x: abs(float(x[1])), reverse=True)
                    for fname, shap_raw in ordered:
                        pts = _shap_to_display_points(float(shap_raw), max_abs, 20.0)
                        direction = "positive" if float(shap_raw) > 0 else "negative"
                        cls = "shap-pos" if direction == "positive" else "shap-neg"
                        sign = "+" if pts > 0 else ""
                        label = FEAT_LABELS_SHAP.get(fname, fname)
                        src = expl_by_feature.get(fname)
                        if src:
                            explanation = src.get("explanation", "")
                            raw_hint = src.get("raw_value", raw_for_shap.get(fname))
                        else:
                            explanation = (
                                f"Значение признака в модели: {raw_for_shap.get(fname, '—')}. "
                                f"Вклад в итоговый балл (SHAP): {float(shap_raw):+.3f} в шкале модели."
                            )
                            raw_hint = raw_for_shap.get(fname)
                        shap_html += f"""
                        <div class="shap-item {cls}">
                            <span class="shap-val">{sign}{pts} б.</span>
                            <div>
                                <div style="font-weight:600;">{label}</div>
                                <div style="font-size:12px; color:#888;">{explanation}</div>
                                <div style="font-size:11px; color:#aaa;">исходное значение: {raw_hint}</div>
                            </div>
                        </div>"""
                if shap_html:
                    st.markdown(shap_html, unsafe_allow_html=True)
                else:
                    st.caption("SHAP-объяснения недоступны для этой заявки")

                if all_shap:
                    sorted_shap = sorted(all_shap.items(), key=lambda x: abs(float(x[1])), reverse=True)[:6]
                    labels_shap = [FEAT_LABELS_SHAP.get(k, k)[:28] for k, _ in sorted_shap]
                    pts_shap = [_shap_to_display_points(float(v), max_abs, 20.0) for _, v in sorted_shap]
                    colors_shap = ["#1a7a4a" if v > 0 else "#b5001f" for v in pts_shap]

                    fig_shap = go.Figure(go.Bar(
                        x=pts_shap,
                        y=labels_shap,
                        orientation="h",
                        marker_color=colors_shap,
                        text=[f"{v:+d} б." for v in pts_shap],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        title="Топ-6 факторов (вклад в балл, шкала ±20 отн. сильнейшего)",
                        height=300,
                        plot_bgcolor="#f8faff",
                        paper_bgcolor="#ffffff",
                        margin=dict(t=50, b=20, l=10, r=70),
                        font=dict(family="Golos Text", size=11),
                        xaxis_title="Баллы влияния (условные)",
                        xaxis=dict(range=[-22, 22]),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                st.divider()
                _plot_score_summary_no_waterfall(app, score_ml)

                st.markdown("#### 💼 Финансовая сводка")

                if compliance_bonus != 0:
                    bonus_cls = "pos" if compliance_bonus >= 0 else "neg"
                    bonus_sign = "+" if compliance_bonus >= 0 else ""
                    st.markdown(f"""
                    <div class="score-breakdown">
                        <span class="score-ml">XGBoost: <b>{score_ml:.0f}</b></span>
                        <span class="score-arrow">→</span>
                        <span class="score-bonus {bonus_cls}">документы: {bonus_sign}{compliance_bonus:.1f}</span>
                        <span class="score-arrow">=</span>
                        <span class="score-final">{score:.0f} / 100</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <table style="width:100%; font-size:13px; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:6px 4px; color:#888;">Запрошено</td>
                        <td style="text-align:right; font-weight:700;">{_fmt_tenge(app.get('requested_amount',0))}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:6px 4px; color:#888;">Регион</td>
                        <td style="text-align:right;">{app.get('region','—')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:6px 4px; color:#888;">Источник</td>
                        <td style="text-align:right;">{app.get('source_system','manual').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 4px; color:#888;">Дата подачи</td>
                        <td style="text-align:right;">{app.get('application_date', app.get('calculated_at','')[:10])}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

            compliance = app.get("compliance")
            if compliance:
                st.markdown("---")
                st.markdown("#### 📋 Проверка соответствия Правилам субсидирования (Приказ МСХ РК № 108)")

                c_status = compliance.get("overall_status", "")
                c_score_pct = compliance.get("overall_score_pct", 0)
                c_bonus = compliance.get("compliance_bonus", 0)
                c_checks = compliance.get("checks", [])
                c_disq = compliance.get("disqualifiers_found", [])
                c_name = compliance.get("subsidy_name", "")

                badge_cls = {
                    "СООТВЕТСТВУЕТ": "badge-ok",
                    "ЧАСТИЧНО": "badge-warn",
                    "НЕ СООТВЕТСТВУЕТ": "badge-fail",
                    "ДИСКВАЛИФИКАЦИЯ": "badge-disq",
                }.get(c_status, "badge-warn")

                bonus_sign = "+" if c_bonus >= 0 else ""
                bonus_color = "#1a7a4a" if c_bonus >= 0 else "#b5001f"

                st.markdown(f"""
                <div class="compliance-block">
                    <div class="compliance-header">
                        <div>
                            <span class="compliance-title">📑 {c_name}</span>
                            <span style="font-size:12px; color:#888; margin-left:10px;">
                                Выполнено {c_score_pct:.0f}% требований
                            </span>
                        </div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:13px; font-weight:700; color:{bonus_color};">
                                {bonus_sign}{c_bonus:.1f} к баллу
                            </span>
                            <span class="compliance-badge {badge_cls}">{c_status}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                bar_color = "#1a7a4a" if c_score_pct >= 85 else ("#e8a800" if c_score_pct >= 60 else "#b5001f")
                st.markdown(f"""
                    <div style="background:#eee; border-radius:6px; height:8px; margin-bottom:14px;">
                        <div style="background:{bar_color}; width:{c_score_pct}%; height:8px; border-radius:6px;
                             transition: width 0.5s;"></div>
                    </div>
                """, unsafe_allow_html=True)

                if c_disq:
                    st.markdown(f"""
                    <div style="background:#2d0010; border:1px solid #8b002a; border-radius:8px;
                         padding:12px 16px; margin-bottom:12px;">
                        <div style="color:#ff6b8a; font-weight:700; font-size:13px; margin-bottom:6px;">
                            🚫 ДИСКВАЛИФИЦИРУЮЩИЕ УСЛОВИЯ — субсидия невозможна:
                        </div>
                        {''.join(f"<div style='color:#ffaaaa; font-size:12px; margin-top:4px;'>• {d}</div>" for d in c_disq)}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)                              

                c_col1, c_col2 = st.columns(2)
                half = (len(c_checks) + 1) // 2

                for col, chunk in [(c_col1, c_checks[:half]), (c_col2, c_checks[half:])]:
                    with col:
                        for chk in chunk:
                            status = chk.get("status", "НЕ НАЙДЕНО")
                            emoji = chk.get("emoji", "❓")
                            text = chk.get("text", "")
                            evidence = chk.get("evidence", "")
                            source = chk.get("source", "")
                            is_critical = chk.get("is_critical", False)

                            if status == "ВЫПОЛНЕНО":
                                item_cls = "check-ok"
                            elif status in ("ЧАСТИЧНО", "ПРЕДУПРЕЖДЕНИЕ"):
                                item_cls = "check-warn"
                            else:
                                item_cls = "check-fail"

                            critical_badge = '<span class="check-critical">КРИТИЧНО</span>' if is_critical else ""

                            st.markdown(f"""
                            <div class="check-item {item_cls}">
                                <span class="check-emoji">{emoji}</span>
                                <div class="check-text">
                                    <div class="check-label">{text}{critical_badge}</div>
                                    <div class="check-evidence">{evidence}</div>
                                    <div class="check-source">{source}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                c_rec = compliance.get("recommendation", "")
                if c_rec:
                    st.markdown(f"""
                    <div style="background:#f4f6fb; border:1px solid #dde3ef; border-radius:8px;
                         padding:12px 16px; margin-top:12px; font-size:13px; color:#444;">
                        {c_rec}
                    </div>
                    """, unsafe_allow_html=True)

                llm_doc = app.get("llm_document_analysis")
                if llm_doc:
                    st.markdown(f"""
                    <div style="background:#f0f4ff; border:1px solid #c5d5f0; border-radius:8px;
                         padding:12px 16px; margin-top:8px; font-size:13px;">
                        <b style="color:#003580;">📄 ИИ-анализ документов:</b><br>
                        <span style="color:#444;">{llm_doc}</span>
                    </div>
                    """, unsafe_allow_html=True)

            elif app.get("documents_processed", 0) == 0:
                st.markdown("""
                <div style="background:#f8f8f8; border:1px dashed #ccc; border-radius:8px;
                     padding:14px 18px; margin-top:16px; font-size:13px; color:#888; text-align:center;">
                    📎 Документы не загружены — проверка соответствия правилам недоступна.<br>
                    Используйте эндпоинт <code>/api/v1/score-with-documents</code> для полного анализа.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="hitl-block">
                <div class="hitl-title">👤 Решение комиссии (Human-in-the-Loop)</div>
                <div class="hitl-desc">
                    ИИ предоставляет рекомендацию, однако окончательное решение принимается уполномоченным
                    сотрудником комиссии Министерства сельского хозяйства РК. Ваше решение будет зафиксировано
                    в системе с указанием ФИО и временной меткой.
                </div>
            </div>
            """, unsafe_allow_html=True)

            hitl_col1, hitl_col2, hitl_col3 = st.columns([3, 1, 1])
            with hitl_col1:
                comment = st.text_area("Комментарий (необязательно)", height=60,
                                       key=f"comment_{selected_id}")
            with hitl_col2:
                if st.button("✅ Одобрить выплату", type="primary",
                             key=f"approve_{selected_id}", use_container_width=True):
                    payload = {
                        "application_id": selected_id,
                        "decision": "approved",
                        "comment": comment,
                    }
                    result = _api_post("/api/v1/decision", payload)
                    if result:
                        st.session_state.decisions[selected_id] = "approved"
                        st.success(f"✅ Выплата одобрена. Зафиксировано в системе.")
                        _refresh_apps()
                        st.rerun()
            with hitl_col3:
                if st.button("❌ Отказать", type="secondary",
                             key=f"reject_{selected_id}", use_container_width=True):
                    payload = {
                        "application_id": selected_id,
                        "decision": "rejected",
                        "comment": comment,
                    }
                    result = _api_post("/api/v1/decision", payload)
                    if result:
                        st.session_state.decisions[selected_id] = "rejected"
                        st.warning("❌ Отказ зафиксирован в системе.")
                        _refresh_apps()
                        st.rerun()

            current_decision = st.session_state.decisions.get(selected_id)
            if current_decision:
                d_label = "✅ Одобрено" if current_decision == "approved" else "❌ Отказано"
                d_color = "#1a7a4a" if current_decision == "approved" else "#b5001f"
                st.markdown(f"""
                <div style="margin-top:12px; background:#f8f8f8; border:1px solid #ddd;
                     border-radius:8px; padding:10px 16px; font-size:13px; color:{d_color}; font-weight:700;">
                    {d_label} — Решение зафиксировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}
                </div>
                """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 🔌 Интеграция API — Примеры запросов")

    st.markdown("""
    <div class="info-box">
        <h4>📌 Базовая информация</h4>
        <p style="font-size:14px; color:#444; line-height:1.7;">
            <b>Base URL:</b> <code>http://localhost:8003</code><br>
            <b>Авторизация:</b> API Key в заголовке <code>X-API-Key</code><br>
            <b>Формат:</b> JSON<br>
            <b>Rate Limit:</b> 1000 запросов/час
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # POST /api/v1/score
    st.markdown("#### 1. POST /api/v1/score — Базовый скоринг")
    st.caption("Расчёт скорингового балла по числовым данным")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Запрос:**")
        st.code("""curl -X POST http://localhost:8003/api/v1/score \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \\
  -d '{
    "bin_iin": "123456789012",
    "company_name": "ТОО Агро-Нур",
    "region": "Алматинская область",
    "subsidy_type": "Приобретение племенного КРС",
    "requested_amount": 15000000,
    "pedigree_ratio": 0.85,
    "veterinary_compliance": 0.98,
    "years_in_operation": 7
  }'""", language="bash")

    with col2:
        st.markdown("**Ответ:**")
        st.code("""{
  "application_id": "A7F2B1C0",
  "score_ml": 84.5,
  "zone": "green",
  "recommendation": "Строго рекомендовано",
  "top_positive_factors": [
    {"label": "Доля племенного поголовья", "shap_value": 14.2},
    {"label": "Ветеринария", "shap_value": 8.5}
  ],
  "top_negative_factors": [
    {"label": "Долговая нагрузка", "shap_value": -3.1}
  ]
}""", language="json")

    st.divider()

    # POST /api/v1/score-with-documents
    st.markdown("#### 2. POST /api/v1/score-with-documents — Скоринг с документами")
    st.caption("Скоринг + LLM-анализ PDF-документов + проверка правил")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Запрос:**")
        st.code("""curl -X POST http://localhost:8003/api/v1/score-with-documents \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \\
  -F 'features_json={
    "bin_iin": "123456789012",
    "company_name": "ТОО Агро-Нур",
    "region": "Алматинская область",
    "requested_amount": 15000000
  }' \\
  -F 'documents=@ustav.pdf' \\
  -F 'documents=@veterinary_cert.pdf'""", language="bash")

    with col2:
        st.markdown("**Ответ:**")
        st.code("""{
  "application_id": "A7F2B1C0",
  "score_ml": 84.5,
  "score_bonus": 5.0,
  "score_final": 89.5,
  "zone": "green",
  "compliance_status": "passed",
  "llm_analysis": {
    "documents_processed": 2,
    "compliance_score": 95,
    "issues": []
  },
  "recommendation": "Строго рекомендовано"
}""", language="json")

    st.divider()

    # GET /api/v1/applications
    st.markdown("#### 3. GET /api/v1/applications — Список заявок")
    st.caption("Получение списка всех заявок (сортировка по баллу)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Запрос:**")
        st.code("""curl "http://localhost:8003/api/v1/applications?zone=green&min_score=80" \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" """, language="bash")

    with col2:
        st.markdown("**Ответ:**")
        st.code("""[
  {
    "application_id": "A7F2B1C0",
    "company_name": "ТОО Агро-Нур",
    "region": "Алматинская область",
    "score": 84.5,
    "zone": "green",
    "status": "pending"
  },
  {
    "application_id": "B3E8D2F1",
    "company_name": "ТОО Байтерек",
    "region": "Туркестанская область",
    "score": 82.0,
    "zone": "green",
    "status": "approved"
  }
]""", language="json")

    st.divider()

    # POST /api/v1/decision
    st.markdown("#### 4. POST /api/v1/decision — Решение комиссии")
    st.caption("Фиксация решения комиссии (Human-in-the-Loop)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Запрос:**")
        st.code("""curl -X POST http://localhost:8003/api/v1/decision \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \\
  -d '{
    "application_id": "A7F2B1C0",
    "decision": "approved",
    "comment": "Соответствует критериям, рекомендовано к одобрению"
  }'""", language="bash")

    with col2:
        st.markdown("**Ответ:**")
        st.code("""{
  "status": "success",
  "application_id": "A7F2B1C0",
  "decision": "approved",
  "comment": "Соответствует критериям, рекомендовано к одобрению",
  "timestamp": "2025-04-02T14:30:00Z",
  "message": "Решение зафиксировано"
}""", language="json")

    st.divider()

    # POST /api/v1/giss/sync
    st.markdown("#### 5. POST /api/v1/giss/sync — Тестовые заявки")
    st.caption("Синхронизация тестовых заявок (демо-режим)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Запрос:**")
        st.code("""curl -X POST http://localhost:8003/api/v1/giss/sync \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" """, language="bash")

    with col2:
        st.markdown("**Ответ:**")
        st.code("""{
  "status": "success",
  "synced_count": 15,
  "message": "Тестовые заявки добавлены"
}""", language="json")

    st.divider()

    # Таблица зон скоринга
    st.markdown("#### 📊 Зоны скоринга")

    zones_data = {
        "Зона": ["🟢 green", "🟡 yellow", "🔴 red"],
        "Диапазон баллов": ["80–100", "50–79", "0–49"],
        "Рекомендация": [
            "Строго рекомендовано",
            "Рассмотрение комиссией",
            "Не рекомендовано"
        ],
        "Вероятность одобрения": [
            "Высокая (>90%)",
            "Средняя (40–60%)",
            "Низкая (<10%)"
        ],
    }
    st.dataframe(pd.DataFrame(zones_data), hide_index=True, use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align:center; font-size:12px; color:#9aacce; padding:10px 0;">
    SmartAgro Score v2.0 | Министерство сельского хозяйства РК |
    Decentrathon 5.0 — AI for Government |
    ИИ предоставляет рекомендацию, финальное решение принимается комиссией
</div>
""", unsafe_allow_html=True)
