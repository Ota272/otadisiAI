"""
SmartAgro Score — Дашборд Министерства сельского хозяйства РК
Хакатон Decentrathon 5.0 | AI for Government
"""

import json
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

API_BASE = "http://localhost:8002"
API_KEY = "sk-msgov-2025-demo-key-abc123"
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(
    page_title="SmartAgro Score | МСХ РК",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Государственный стиль
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Сессионное состояние
# ─────────────────────────────────────────────

if "applications" not in st.session_state:
    st.session_state.applications = []
if "selected_app_id" not in st.session_state:
    st.session_state.selected_app_id = None
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "api_key_display" not in st.session_state:
    st.session_state.api_key_display = API_KEY
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

ITEMS_PER_PAGE = 10

# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

class ValidationErrorInfo:
    """Информация об ошибке валидации."""
    field: str
    message: str
    valid_values: list[str] | None = None


def _parse_validation_error(error_text: str) -> list[dict]:
    """
    Парсит ответ API с ошибками валидации.
    Возвращает список ошибок с полями и сообщениями.
    """
    try:
        # Пытаемся извлечь JSON из текста ошибки
        import json
        # Ищем JSON в тексте ошибки
        start = error_text.find('{')
        if start == -1:
            return []
        json_str = error_text[start:]
        # Находим закрывающую скобку
        depth = 0
        end = start
        for i, c in enumerate(json_str[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = json_str[:end]

        data = json.loads(json_str)
        errors = []

        if "detail" in data and isinstance(data["detail"], list):
            for err in data["detail"]:
                field = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "")

                # Извлекаем допустимые значения из сообщения
                valid_values = None
                if "Допустимые значения:" in msg:
                    parts = msg.split("Допустимые значения: ")
                    if len(parts) > 1:
                        valid_values = [v.strip() for v in parts[1].split(", ")]

                errors.append({
                    "field": field,
                    "message": msg,
                    "valid_values": valid_values,
                })
        return errors
    except Exception:
        return []


def _show_validation_error(error_text: str):
    """
    Красиво отображает ошибки валидации.
    """
    errors = _parse_validation_error(error_text)

    if not errors:
        st.error(f"❌ Ошибка валидации: {error_text}")
        return

    st.error("❌ **Обнаружены ошибки валидации:**")

    # Словарь с человеческими названиями полей
    FIELD_LABELS = {
        "body.bin_iin": "БИН/ИИН",
        "body.company_name": "Наименование предприятия",
        "body.region": "Область",
        "body.subsidy_type": "Вид субсидии",
        "body.requested_amount": "Запрашиваемая сумма",
        "body.application_date": "Дата подачи заявки",
        "body.akimat": "Акимат",
        "body.direction": "Направление водства",
        "body.subsidy_name": "Наименование субсидии",
        "body.normativ": "Норматив",
        "body.amount_due": "Причитающаяся сумма",
        "body.district": "Район хозяйства",
        "body.source_system": "Источник",
        "body": "Заявка",
    }

    for err in errors:
        field = err["field"]
        message = err["message"]
        valid_values = err.get("valid_values")

        # Получаем человеческое название поля
        field_label = FIELD_LABELS.get(field, field.replace("body.", ""))

        # Формируем сообщение
        if valid_values:
            # Показываем первые 5 допустимых значений
            show_values = valid_values[:5]
            more_text = f" и ещё {len(valid_values) - 5}" if len(valid_values) > 5 else ""

            st.markdown(f"""
            <div style="background: #fff3cd; border: 1px solid #b36200;
                 border-left: 4px solid #b36200; border-radius: 8px;
                 padding: 12px 16px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: #b36200; margin-bottom: 6px;">
                    ⚠️ {field_label}
                </div>
                <div style="color: #664d03; font-size: 14px; margin-bottom: 8px;">
                    {message.split('.')[0]}.
                </div>
                <div style="font-size: 13px; color: #856404;">
                    <b>Допустимые значения{more_text}:</b><br>
                    {', '.join(show_values)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #fff3cd; border: 1px solid #b36200;
                 border-left: 4px solid #b36200; border-radius: 8px;
                 padding: 12px 16px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: #b36200;">
                    ⚠️ {field_label}
                </div>
                <div style="color: #664d03; font-size: 14px; margin-top: 4px;">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True)


def _api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, headers=HEADERS, timeout=10)

        # Обрабатываем ошибку валидации 422
        if r.status_code == 422:
            _show_validation_error(r.text)
            return None

        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 422:
            _show_validation_error(e.response.text)
        else:
            st.error(f"❌ Ошибка API: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка подключения: {e}")
        return None


def _api_get(endpoint: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, timeout=10)

        # Обрабатываем ошибку валидации 422
        if r.status_code == 422:
            _show_validation_error(r.text)
            return None

        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 422:
            _show_validation_error(e.response.text)
        else:
            st.error(f"❌ Ошибка API: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка подключения: {e}")
        return None


def _score_pill(score: float, category: str) -> str:
    cls = {"green": "pill-green", "yellow": "pill-yellow", "red": "pill-red"}.get(category, "")
    return f'<span class="score-pill {cls}">{score:.0f}</span>'


def _fmt_tenge(val: float) -> str:
    return f"{val/1_000_000:.2f} млн ₸"


def _refresh_apps():
    data = _api_get("/api/v1/applications")
    if data is not None:
        st.session_state.applications = data


# ─────────────────────────────────────────────
# Боковая панель
# ─────────────────────────────────────────────

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
        👤 Аманжолов Д.С.<br>
        <span style="color:#8ab4e8; font-size:12px;">Главный специалист | Отдел субсидирования</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    total = len(st.session_state.applications)
    green = sum(1 for a in st.session_state.applications if a.get("score_category") == "green")
    yellow = sum(1 for a in st.session_state.applications if a.get("score_category") == "yellow")
    red = sum(1 for a in st.session_state.applications if a.get("score_category") == "red")

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

# ─────────────────────────────────────────────
# Шапка
# ─────────────────────────────────────────────

st.markdown("""
<div class="gov-header">
    <div>
        <div class="logo-text">🌾 SmartAgro Score</div>
        <div class="logo-sub">Информационно-аналитическая система merit-based скоринга субсидий</div>
    </div>
    <div class="badge">ВНУТРЕННИЙ ПОРТАЛ МСХ РК</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Вкладки
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Поток заявок",
    "📊 Шорт-лист и бюджет",
    "🔍 Профиль фермера (XAI)",
    "🔌 Интеграция API",
])

# ══════════════════════════════════════════════
# ВКЛАДКА 1: Поток заявок
# ══════════════════════════════════════════════

with tab1:
    st.markdown("### 📥 Поток заявок — API и ручная загрузка")

    col_sync, col_status = st.columns([1, 2])

    with col_sync:
        if st.button("🔄 Синхронизировать с ГИСС (API)", type="primary", use_container_width=True):
            with st.spinner("Подключение к ГИСС..."):
                time.sleep(1.2)
                data = _api_post("/api/v1/giss/sync", {})
                if data:
                    st.success(f"✅ Синхронизировано {data['synced_count']} заявок из ГИСС")
                    _refresh_apps()
                    st.rerun()

    with col_status:
        st.markdown("""
        <div class="giss-status">
            <span class="giss-dot"></span>
            ГИСС: онлайн | eGov: онлайн | Последняя синхронизация: только что
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Ручной ввод ──
    with st.expander("✏️ Ручной ввод заявки (форма)", expanded=False):
        st.markdown("#### Данные предприятия")

        c1, c2, c3 = st.columns(3)
        with c1:
            man_bin = st.text_input("БИН/ИИН*", placeholder="123456789012")
            man_company = st.text_input("Наименование предприятия*", placeholder="ТОО «Агро-Нур»")
        with c2:
            man_region = st.selectbox("Область*", [
                "Алматинская", "Акмолинская", "Атырауская", "Восточно-Казахстанская",
                "Жамбылская", "Карагандинская", "Костанайская", "Кызылординская",
                "Мангистауская", "Павлодарская", "Северо-Казахстанская",
                "Туркестанская", "Западно-Казахстанская", "Актюбинская",
            ])
            man_subsidy = st.selectbox("Вид субсидии*", [
                "Субсидирование племенного КРС",
                "Субсидирование молочного стада",
                "Субсидирование овцеводства",
            ])
        with c3:
            man_amount = st.number_input("Сумма заявки (тенге)*", min_value=100_000, max_value=500_000_000,
                                          value=10_000_000, step=500_000)

        st.markdown("#### Данные для модели")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            man_app_date = st.text_input("Дата подачи*", value=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                         help="Формат: DD.MM.YYYY HH:MM:SS")
            man_akimat = st.selectbox("Акимат*", ["Акимат г. Алматы", "Акимат г. Астаны", "Акимат Шыркент", "Акимат Тараз"])
        with f2:
            man_direction = st.selectbox("Направление водства*", ["Мясное", "Молочное", "Овцеводство", "Птицеводство"])
            man_subsidy_name = st.selectbox("Наименование субсидии*", [
                "Субсидирование племенного КРС",
                "Субсидирование молочного стада",
                "Субсидирование овцеводства",
            ])
        with f3:
            man_normativ = st.number_input("Норматив*", min_value=0.0, value=1000000.0, step=100000.0)
            man_district = st.selectbox("Район хозяйства*", ["Алматинский район", "Шуский район", "Талгарский район", "Карасайский район"])
        with f4:
            man_amount_due = st.number_input("Причитающая сумма*", min_value=0.0, value=float(man_amount), step=100000.0)

        st.markdown("#### 📎 Загрузка документов")

        doc_col1, doc_col2, doc_col3 = st.columns(3)
        with doc_col1:
            st.markdown("**Учредительные документы**")
            st.file_uploader("Свидетельство ТОО/ИП", key="doc_reg", type=["pdf", "jpg", "png"])
            st.file_uploader("Справка о налоговом учёте (БИН/ИИН)", key="doc_tax", type=["pdf"])
            st.file_uploader("Справка с банка", key="doc_bank", type=["pdf"])
            st.file_uploader("Лицевой счёт в ГИСС", key="doc_giss_acc", type=["pdf"])
        with doc_col2:
            st.markdown("**Земля и животноводство**")
            st.file_uploader("Сведения о земельных участках", key="doc_land", type=["pdf"])
            st.file_uploader("Чеки на корма/ГСМ", key="doc_feed", type=["pdf", "jpg"])
            st.file_uploader("Справка о ветеринарном благополучии", key="doc_vet", type=["pdf"])
            st.file_uploader("Акты/ЭСФ на племенной скот", key="doc_livestock", type=["pdf"])
            st.file_uploader("Обязательство на 2 года (племенной молодняк)", key="doc_oblig", type=["pdf"])
        with doc_col3:
            st.markdown("**Дополнительные**")
            st.file_uploader("Договор лизинга/займа", key="doc_lease", type=["pdf"])

        if st.button("📤 Отправить заявку на скоринг", type="primary"):
            if not man_bin or not man_company:
                st.warning("Заполните обязательные поля (БИН/ИИН и наименование).")
            else:
                with st.spinner("Расчёт скорингового балла..."):
                    payload = {
                        "bin_iin": man_bin,
                        "company_name": man_company,
                        "region": man_region,
                        "subsidy_type": man_subsidy,
                        "requested_amount": man_amount,
                        "application_date": man_app_date,
                        "akimat": man_akimat,
                        "direction": man_direction,
                        "subsidy_name": man_subsidy_name,
                        "normativ": man_normativ,
                        "amount_due": man_amount_due,
                        "district": man_district,
                        "source_system": "manual",
                    }
                    result = _api_post("/api/v1/score", payload)
                    if result:
                        st.success(f"✅ Заявка принята. ID: **{result['application_id']}** | Балл: **{result['score']:.0f}**")
                        _refresh_apps()
                        st.rerun()

    # ── Список заявок ──
    st.divider()
    st.markdown("#### 📋 Все заявки в очереди")

    # Кнопка обновления и счётчик
    ref_col, cnt_col = st.columns([1, 4])
    with ref_col:
        if st.button("🔄 Обновить", use_container_width=True):
            _refresh_apps()
            st.rerun()
    with cnt_col:
        st.caption(f"Всего заявок: {len(st.session_state.applications)}")

    _refresh_apps()

    if not st.session_state.applications:
        st.info("Заявок нет. Нажмите «Синхронизировать с ГИСС» или добавьте вручную.")
    else:
        # Фильтр по категории
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
        with filter_col1:
            show_green = st.checkbox("🟢 Рекомендовано", value=True)
        with filter_col2:
            show_yellow = st.checkbox("🟡 На рассмотрении", value=True)
        with filter_col3:
            show_red = st.checkbox("🔴 Не рекомендовано", value=True)

        # Фильтрация
        filtered_apps = [
            app for app in st.session_state.applications
            if (
                (app.get("score_category") == "green" and show_green) or
                (app.get("score_category") == "yellow" and show_yellow) or
                (app.get("score_category") == "red" and show_red)
            )
        ]

        if not filtered_apps:
            st.warning("Нет заявок по выбранным фильтрам.")
        else:
            # ═══════════════════════════════════════
            # ПАГИНАЦИЯ
            # ═══════════════════════════════════════
            total_items = len(filtered_apps)
            total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

            # Если текущая страница больше последней — возвращаем на последнюю
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = max(1, total_pages)

            # Навигация (стрелки + номер страницы)
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 1, 4])

            with nav_col1:
                if st.button("⬅️ Назад", disabled=(st.session_state.current_page <= 1), use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()

            with nav_col2:
                st.markdown(f"<div style='text-align:center; padding-top:10px; font-weight:600;'>Страница {st.session_state.current_page} из {total_pages}</div>", unsafe_allow_html=True)

            with nav_col3:
                if st.button("Вперёд ➡️", disabled=(st.session_state.current_page >= total_pages), use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()

            with nav_col4:
                # Индикатор диапазона
                start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE + 1
                end_idx = min(st.session_state.current_page * ITEMS_PER_PAGE, total_items)
                st.caption(f"Показано {start_idx}–{end_idx} из {total_items}")

            # ═══════════════════════════════════════
            # ОТОБРАЖЕНИЕ ЗАЯВОК (только текущая страница)
            # ═══════════════════════════════════════
            start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            page_apps = filtered_apps[start_idx:end_idx]

            for app in page_apps:
                cat = app.get("score_category", "yellow")
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
                        st.markdown(f"{icon} **{app['company_name'][:30]}** `{app['bin_iin']}`")
                    with c2:
                        st.caption(f"{app['region']} | {app['subsidy_type'][:25]}...")
                    with c3:
                        st.caption(f"{_fmt_tenge(app['requested_amount'])}")
                    with c4:
                        st.markdown(f"**{score:.0f}** / 100")
                    with c5:
                        if dec_badge:
                            st.caption(dec_badge)
                        elif st.button("🔍", key=f"open_{app['application_id']}", help="Открыть профиль"):
                            st.session_state.selected_app_id = app["application_id"]
                            st.info("Перейдите на вкладку '🔍 Профиль фермера (XAI)'")

# ══════════════════════════════════════════════
# ВКЛАДКА 2: Шорт-лист и бюджет
# ══════════════════════════════════════════════

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
        st.info("Заявок нет. Синхронизируйте с ГИСС или добавьте вручную.")
    else:
        # Строим таблицу шорт-листа
        rows_html = ""
        cumulative = 0.0
        cutoff_drawn = False

        for app in apps_sorted:
            app_id = app["application_id"]
            score = app.get("score", 0)
            cat = app.get("score_category", "yellow")
            amt = app.get("requested_amount", 0)
            decision = st.session_state.decisions.get(app_id, "")

            # Определяем, вписывается ли в бюджет
            would_exceed = (cumulative + amt) > budget

            # Черта бюджета
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

            rows_html += f"""
            <tr class="{row_class}">
                <td>{date_str}</td>
                <td><b>{app['company_name']}</b><br><small style='color:#888;'>{app_id}</small></td>
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

        # Мини-гистограмма баллов
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

# ══════════════════════════════════════════════
# ВКЛАДКА 3: Профиль фермера (XAI)
# ══════════════════════════════════════════════

with tab3:
    st.markdown("### 🔍 Профиль фермера — Explainable AI")

    apps = st.session_state.applications

    if not apps:
        st.info("Нет заявок для анализа. Сначала синхронизируйте с ГИСС.")
    else:
        app_names = {a["application_id"]: f"{a['company_name']} [{a['application_id']}]" for a in apps}
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
            cat = app.get("score_category", "yellow")
            score = app.get("score", 0)
            cat_colors = {"green": "#1a7a4a", "yellow": "#b36200", "red": "#b5001f"}
            cat_bg = {"green": "#e8f8f0", "yellow": "#fffbea", "red": "#fff0f2"}
            cat_labels = {"green": "✅ СТРОГО РЕКОМЕНДОВАНО", "yellow": "⚠️ ТРЕБУЕТ РАССМОТРЕНИЯ", "red": "🚫 НЕ РЕКОМЕНДОВАНО"}

            # Заголовок профиля
            st.markdown(f"""
            <div style="background:{cat_bg.get(cat,'#f8f8f8')}; border: 2px solid {cat_colors.get(cat,'#888')};
                 border-radius:12px; padding:20px 24px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-family:'Montserrat',sans-serif; font-size:22px; font-weight:800;
                             color:#1a2340;">{app['company_name']}</div>
                        <div style="color:#6b7a99; font-size:13px; margin-top:4px;">
                            БИН/ИИН: {app['bin_iin']} &nbsp;|&nbsp; {app['region']} &nbsp;|&nbsp; {app['subsidy_type']}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:52px; font-weight:900; color:{cat_colors.get(cat,'#333')};
                             font-family:'Montserrat',sans-serif; line-height:1;">{score:.0f}</div>
                        <div style="font-size:11px; color:#888; letter-spacing:0.5px;">/ 100 баллов</div>
                        <div style="font-size:13px; font-weight:700; color:{cat_colors.get(cat,'#333')};
                             margin-top:4px;">{cat_labels.get(cat,'')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            prof_col1, prof_col2 = st.columns([3, 2])

            with prof_col1:
                # ── Динамика валовой продукции (mock) ──
                st.markdown("#### 📈 Динамика показателей предприятия")

                np.random.seed(hash(app["bin_iin"]) % (2**31))
                growth = app.get("gross_output_growth", 0.1)
                years = [2020, 2021, 2022, 2023, 2024, 2025]
                base = random.uniform(50, 150)
                output_vals = [base * (1 + growth * (i / 2 + random.uniform(-0.05, 0.05))) for i in range(6)]
                land_vals = [app.get("land_utilization", 0.75) * 100 + random.uniform(-5, 5) for _ in years]
                survival_vals = [app.get("historical_survival_rate", 0.85) * 100 + random.uniform(-3, 3) for _ in years]

                fig_dyn = go.Figure()
                fig_dyn.add_trace(go.Scatter(
                    x=years, y=output_vals, name="Валовая продукция (млн ₸)",
                    line=dict(color="#003580", width=3), fill="tozeroy",
                    fillcolor="rgba(0,53,128,0.08)", mode="lines+markers",
                ))
                fig_dyn.update_layout(
                    height=220, plot_bgcolor="#f8faff", paper_bgcolor="#ffffff",
                    margin=dict(t=20, b=30, l=30, r=10),
                    font=dict(family="Golos Text", size=11),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_dyn, use_container_width=True)

                # Утилизация земли и выживаемость
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=years, y=land_vals, name="Утилизация земли (%)",
                                       marker_color="#0072CE", opacity=0.8))
                fig2.add_trace(go.Scatter(x=years, y=survival_vals, name="Выживаемость (%)",
                                           line=dict(color="#C8952A", width=2.5),
                                           mode="lines+markers", yaxis="y2"))
                fig2.update_layout(
                    height=220, plot_bgcolor="#f8faff", paper_bgcolor="#ffffff",
                    margin=dict(t=20, b=30),
                    font=dict(family="Golos Text", size=11),
                    legend=dict(orientation="h"),
                    yaxis=dict(title="Земля (%)", range=[0, 120]),
                    yaxis2=dict(title="Выживаемость (%)", overlaying="y", side="right", range=[0, 120]),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Radar chart
                shap_vals = app.get("shap_values", {})
                feature_labels = {
                    "Дата поступления": "Дата подачи",
                    "Область": "Область",
                    "Акимат": "Акимат",
                    "Направление водства": "Направление",
                    "Наименование субсидирования": "Тип субсидии",
                    "Норматив": "Норматив",
                    "Причитающая сумма": "Сумма",
                    "Район хозяйства": "Район",
                }

                radar_labels = list(feature_labels.values())
                # Нормализация для визуализации
                radar_vals_raw = {
                    "Дата поступления": 50,  # mock
                    "Область": 70,  # mock
                    "Акимат": 60,  # mock
                    "Направление водства": 80,  # mock
                    "Наименование субсидирования": 75,  # mock
                    "Норматив": min(app.get("normativ", 1000000) / 5000000 * 100, 100),
                    "Причитающая сумма": min(app.get("amount_due", 5000000) / 50000000 * 100, 100),
                    "Район хозяйства": 65,  # mock
                }
                radar_vals = [radar_vals_raw.get(k, 50) for k in feature_labels.keys()]

                fig_radar = go.Figure(go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    fillcolor="rgba(0,114,206,0.18)",
                    line=dict(color="#0072CE", width=2.5),
                ))
                fig_radar.update_layout(
                    title="Профиль данных заявки",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=320, paper_bgcolor="#ffffff",
                    font=dict(family="Golos Text", size=11),
                    margin=dict(t=50),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with prof_col2:
                # ── SHAP объяснение ──
                st.markdown("#### 🧠 Объяснение AI-решения (SHAP)")
                st.markdown(f"""
                <div style="background:#f0f4ff; border:1px solid #dde3ef; border-radius:8px;
                     padding:10px 14px; font-size:12px; color:#6b7a99; margin-bottom:12px;">
                    <b>ℹ️ О SHAP:</b> Значения показывают, как каждый фактор влияет на итоговый балл
                    по сравнению со средним по всем заявкам. ИИ предоставляет рекомендацию —
                    финальное решение принимает комиссия.
                </div>
                """, unsafe_allow_html=True)

                explanations = app.get("shap_explanation", [])
                shap_html = ""
                for item in explanations:
                    direction = item.get("direction", "positive")
                    cls = "shap-pos" if direction == "positive" else "shap-neg"
                    sign = "+" if item["shap_value"] > 0 else ""
                    raw_val = item.get('raw_value', 'N/A')
                    # Фикс: строки не форматируем как float
                    try:
                        val_str = f"{float(raw_val):.3f}" if raw_val != 'N/A' else 'N/A'
                    except (ValueError, TypeError):
                        val_str = str(raw_val)
                    shap_html += f"""
                    <div class="shap-item {cls}">
                        <span class="shap-val">{sign}{item['shap_value']:.1f}</span>
                        <div>
                            <div style="font-weight:600;">{item['label']}</div>
                            <div style="font-size:12px; color:#888;">Значение: {val_str}</div>
                        </div>
                    </div>"""

                st.markdown(shap_html, unsafe_allow_html=True)

                # SHAP waterfall (Plotly)
                if explanations:
                    labels_shap = [e["label"][:22] for e in explanations[:6]]
                    vals_shap = [e["shap_value"] for e in explanations[:6]]
                    colors_shap = ["#1a7a4a" if v > 0 else "#b5001f" for v in vals_shap]

                    fig_shap = go.Figure(go.Bar(
                        x=vals_shap,
                        y=labels_shap,
                        orientation="h",
                        marker_color=colors_shap,
                        text=[f"{v:+.1f}" for v in vals_shap],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        title="Топ-6 факторов (SHAP)",
                        height=280,
                        plot_bgcolor="#f8faff",
                        paper_bgcolor="#ffffff",
                        margin=dict(t=40, b=20, l=10, r=60),
                        font=dict(family="Golos Text", size=11),
                        xaxis_title="Влияние на балл",
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                # ── Финансовая сводка ──
                st.markdown("#### 💼 Финансовая сводка")
                st.markdown(f"""
                <table style="width:100%; font-size:13px; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #eee;"><td style="padding:6px 4px; color:#888;">Запрошено</td>
                        <td style="text-align:right; font-weight:700;">{_fmt_tenge(app['requested_amount'])}</td></tr>
                    <tr style="border-bottom:1px solid #eee;"><td style="padding:6px 4px; color:#888;">Регион</td>
                        <td style="text-align:right;">{app['region']}</td></tr>
                    <tr style="border-bottom:1px solid #eee;"><td style="padding:6px 4px; color:#888;">Источник</td>
                        <td style="text-align:right;">{app.get('source_system','manual').upper()}</td></tr>
                    <tr><td style="padding:6px 4px; color:#888;">Дата подачи</td>
                        <td style="text-align:right;">{app.get('application_date', app.get('calculated_at','')[:10])}</td></tr>
                </table>
                """, unsafe_allow_html=True)

            # ── Human-in-the-loop ──
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

            hitl_col1, hitl_col2, hitl_col3 = st.columns([2, 1, 1])
            with hitl_col1:
                officer = st.text_input("ФИО уполномоченного сотрудника", value="Аманжолов Д.С.",
                                        key=f"officer_{selected_id}")
                comment = st.text_area("Комментарий (необязательно)", height=60,
                                       key=f"comment_{selected_id}")
            with hitl_col2:
                if st.button("✅ Одобрить выплату", type="primary",
                             key=f"approve_{selected_id}", use_container_width=True):
                    payload = {
                        "application_id": selected_id,
                        "decision": "approved",
                        "officer_name": officer,
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
                        "officer_name": officer,
                        "comment": comment,
                    }
                    result = _api_post("/api/v1/decision", payload)
                    if result:
                        st.session_state.decisions[selected_id] = "rejected"
                        st.warning("❌ Отказ зафиксирован в системе.")
                        _refresh_apps()
                        st.rerun()

            # Уже принятое решение
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

# ══════════════════════════════════════════════
# ВКЛАДКА 4: Интеграция API
# ══════════════════════════════════════════════

with tab4:
    st.markdown("### 🔌 Интеграция API — Инструкция для внешних систем")

    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("""
        <div class="info-box">
            <h4>📌 О Scoring Engine API</h4>
            <p style="font-size:14px; color:#444; line-height:1.7;">
                SmartAgro Score API предоставляет REST-интерфейс для интеграции с государственными
                информационными системами ГИСС и eGov. Система принимает данные заявок, выполняет
                ML-скоринг с генерацией SHAP-объяснений и возвращает структурированный результат.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🚀 Быстрый старт")

        st.markdown("**Шаг 1. Получите API ключ**")
        st.markdown("""<div class="code-block">
# Ваш API ключ (передавать в заголовке каждого запроса)<br>
X-API-Key: sk-msgov-2025-demo-key-abc123
</div>""", unsafe_allow_html=True)

        st.markdown("**Шаг 2. Отправьте данные заявки на скоринг**")
        st.markdown("""<div class="code-block">
curl -X POST https://smartagro.msxrk.kz/api/v1/score \\<br>
&nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
&nbsp;&nbsp;-H "X-API-Key: YOUR_KEY" \\<br>
&nbsp;&nbsp;-d '{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"bin_iin": "123456789012",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"company_name": "ТОО Агро-Нур",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"region": "Алматинская",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"subsidy_type": "Приобретение племенного КРС",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"requested_amount": 15000000,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"gross_output_growth": 0.18,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"pedigree_ratio": 0.85,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"land_utilization": 0.9,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"historical_survival_rate": 0.92,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"debt_load_ratio": 1.2,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"subsidy_utilization_history": 0.95,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"years_in_operation": 7,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"veterinary_compliance": 0.98<br>
&nbsp;&nbsp;}'
</div>""", unsafe_allow_html=True)

        st.markdown("**Шаг 3. Получите результат скоринга**")
        st.markdown("""<div class="code-block">
{<br>
&nbsp;&nbsp;"application_id": "A7F2B1C0",<br>
&nbsp;&nbsp;"company_name": "ТОО Агро-Нур",<br>
&nbsp;&nbsp;"score": 84.5,<br>
&nbsp;&nbsp;"score_category": "green",<br>
&nbsp;&nbsp;"recommendation": "Строго рекомендовано к одобрению",<br>
&nbsp;&nbsp;"shap_explanation": [<br>
&nbsp;&nbsp;&nbsp;&nbsp;{ "label": "Доля племенного поголовья", "shap_value": 14.2, "direction": "positive" },<br>
&nbsp;&nbsp;&nbsp;&nbsp;{ "label": "Долговая нагрузка", "shap_value": -3.1, "direction": "negative" }<br>
&nbsp;&nbsp;],<br>
&nbsp;&nbsp;"model_version": "GBM-v1.0-mock"<br>
}
</div>""", unsafe_allow_html=True)

        st.markdown("#### 📋 Доступные эндпоинты")
        endpoints_data = {
            "Метод": ["POST", "GET", "POST", "POST", "POST", "GET"],
            "Эндпоинт": [
                "/api/v1/score",
                "/api/v1/applications",
                "/api/v1/giss/sync",
                "/api/v1/decision",
                "/api/v1/keys/generate",
                "/api/v1/decisions",
            ],
            "Описание": [
                "Скоринг одной заявки",
                "Список всех заявок",
                "Синхронизация с ГИСС",
                "Фиксация решения комиссии",
                "Генерация нового API ключа",
                "История решений",
            ],
            "Авторизация": ["✅"] * 6,
        }
        st.dataframe(pd.DataFrame(endpoints_data), hide_index=True, use_container_width=True)

    with c_right:
        st.markdown("#### 🔑 Управление API ключами")

        st.markdown(f"""
        <div style="background:#0f1a2e; border:1px solid #1e3060; border-radius:10px;
             padding:18px 20px; margin-bottom:16px;">
            <div style="color:#8ab4e8; font-size:11px; margin-bottom:8px; letter-spacing:0.5px;">
                АКТИВНЫЙ API КЛЮЧ (��СХ РК)
            </div>
            <div style="color:#7de3a0; font-family:'Courier New',monospace; font-size:14px;
                 word-break:break-all; line-height:1.8;">
                {st.session_state.api_key_display}
            </div>
            <div style="color:#6b7a99; font-size:11px; margin-top:8px;">
                Создан: 2025-01-01 | Доступ: score, read, sync
            </div>
        </div>
        """, unsafe_allow_html=True)

        new_owner = st.text_input("Название организации", placeholder="ГИСС — Р��гионал��ный офис Алматы")
        if st.button("🔑 Сгенерировать новый ключ", use_container_width=True):
            if new_owner:
                result = _api_get(f"/api/v1/keys/generate?owner={new_owner}")
                if result:
                    st.session_state.api_key_display = result.get("api_key", "")
                    st.success(f"Новый ключ сгенерирован для: {new_owner}")
                    st.rerun()
            else:
                st.warning("Введите название организации")

        st.divider()
        st.markdown("#### ⚙️ Системные требования")
        st.markdown("""
        <div style="font-size:13px; line-height:2.0; color:#444;">
            📦 <b>Версия API:</b> v1.0<br>
            🔒 <b>Авторизация:</b> API Key (Header: X-API-Key)<br>
            📄 <b>Формат:</b> JSON (Content-Type: application/json)<br>
            ⏱ <b>Rate limit:</b> 1 000 запросов/час<br>
            🔐 <b>Шифрование:</b> TLS 1.3 (HTTPS)<br>
            📊 <b>Модель:</b> Gradient Boosting + SHAP v0.43<br>
            🌐 <b>Base URL:</b> https://smartagro.msxrk.kz<br>
            📚 <b>Документация:</b> /docs (Swagger UI)
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🏗 Архитектура интеграции")
        st.markdown("""
        <div style="font-size:12px; color:#555; background:#f4f6fb; padding:14px 16px;
             border-radius:8px; border: 1px solid #dde3ef; line-height:1.9;">
            ГИСС / eGov<br>
            &nbsp;&nbsp;&nbsp;↓ POST /api/v1/score<br>
            SmartAgro Score API (FastAPI)<br>
            &nbsp;&nbsp;&nbsp;↓ ML Inference (GBM)<br>
            &nbsp;&nbsp;&nbsp;↓ SHAP Explanation<br>
            &nbsp;&nbsp;&nbsp;↓ JSON Response<br>
            Дашборд МСХ РК (Streamlit)<br>
            &nbsp;&nbsp;&nbsp;↓ Human-in-the-Loop<br>
            Решение комиссии → ГИСС
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Подвал
# ─────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="text-align:center; font-size:12px; color:#9aacce; padding:10px 0;">
    SmartAgro Score v1.0 | Министерство сельского хозяйства РК |
    Decentrathon 5.0 — AI for Government |
    ИИ предоставляет рекомендацию, финальное решение принимается комиссией
</div>
""", unsafe_allow_html=True)