
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
from frontend.locales import t as _t

if "lang" not in st.session_state:
    st.session_state.lang = "ru"

def lang():
    return st.session_state.lang

def T(key, override_lang=None):
    L = override_lang if override_lang else lang()
    return _t(key, L)

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

API_BASE = os.getenv("SMARTAGRO_API_BASE", "http://localhost:8003")
API_KEY = os.getenv("SMARTAGRO_API_KEY", "sk-msgov-2025-demo-key-abc123")
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(
    page_title="SmartAgro Score | МСХ РК",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALID_REGIONS = [
    "Акмолинская область", "Актюбинская область", "Алматинская область",
    "Атырауская область", "Восточно-Казахстанская область", "Жамбылская область",
    "Западно-Казахстанская область", "Карагандинская область", "Костанайская область",
    "Кызылординская область", "Мангистауская область", "Павлодарская область",
    "Северо-Казахстанская область", "Туркестанская область",
    "г.Шымкент", "область Абай", "область Жетісу", "область Ұлытау",
]
VALID_AKIMATS = [
    'ГУ "Управление сельского хозяйства и земельных отношений области Ұлытау"',
    'ГУ "Управление сельского хозяйства Алматинской области"',
    'ГУ "Управление сельского хозяйства Восточно-Казахстанской области"',
    'ГУ "Управление сельского хозяйства Западно-Казахстанской области"',
    'ГУ "Управление сельского хозяйства Кызылординской области"',
    'ГУ "Управление сельского хозяйства Мангистауской области"',
    'ГУ "Управление сельского хозяйства Павлодарской области"',
    'ГУ "Управление сельского хозяйства Туркестанской области"',
    'ГУ "Управление сельского хозяйства и ветеринарии города Шымкент"',
    'ГУ "Управление сельского хозяйства и земельных отношении Акмолинской области"',
    'ГУ "Управление сельского хозяйства и земельных отношений Актюбинской области"',
    'ГУ "Управление сельского хозяйства и земельных отношений Атырауской области"',
    'ГУ "Управление сельского хозяйства и земельных отношений Карагандинской области"',
    'ГУ "Управление сельского хозяйства и земельных отношений акимата Костанайской области"',
    'ГУ "Управление сельского хозяйства области Абай"',
    'ГУ "Управление сельского хозяйства области Жетiсу"',
    'КГУ "Управление сельского хозяйства акимата Жамбылской области"',
    'КГУ "Управление сельского хозяйства и земельных отношений акимата Северо-Казахстанской области"',
]
VALID_DIRECTIONS = [
    "Субсидирование в верблюдоводстве",
    "Субсидирование в козоводстве",
    "Субсидирование в коневодстве",
    "Субсидирование в овцеводстве",
    "Субсидирование в птицеводстве",
    "Субсидирование в пчеловодстве",
    "Субсидирование в свиноводстве",
    "Субсидирование в скотоводстве",
    "Субсидирование затрат по искусственному осеменению",
]
VALID_SUBSIDY_NAMES = [
    "Ведение селекционной и племенной работы с племенным маточным поголовьем отечественных пород лошадей верхового и верхово-упряжного направлений",
    "Возмещение затрат на содержание племенного поголовья пород лошадей верхового и верхово-упряжного направлений, выведенных на территории Республики Казахстан",
    "Заявка на получение субсидий за приобретение импортированного племенного маточного поголовья крупного рогатого скота из Австралии, стран Северной и Южной Америки, стран Европы (молочных, молочно-мясных пород)",
    "Заявка на получение субсидий за приобретение импортированного племенного маточного поголовья крупного рогатого скота из Австралии, стран Северной и Южной Америки, стран Европы (мясных и мясо-молочных пород)",
    "Заявка на получение субсидий за приобретение импортированного племенного маточного поголовья крупного рогатого скота из стран Содружества Независимых Государств, Украины (молочных, молочно-мясных пород)",
    "Заявка на получение субсидий за приобретение отечественного племенного маточного поголовья крупного рогатого скота (молочных, молочно-мясных пород)",
    "Заявка на получение субсидий за приобретение отечественного племенного маточного поголовья крупного рогатого скота (мясных и мясо-молочных пород)",
    "Заявка на получение субсидий за приобретение отечественных племенных овец",
    "Заявка на получение субсидий за приобретение племенного жеребца-производителя продуктивного направления",
    "Заявка на получение субсидий за приобретение племенного маточного поголовья коз",
    "Заявка на получение субсидий за приобретение племенного поголовья свиней",
    "Заявка на получение субсидий за приобретение племенного суточного молодняка родительской/прародительской формы мясного направления птиц",
    "Заявка на получение субсидий за приобретение племенных быков-производителей мясных и мясо-молочных пород",
    "Заявка на получение субсидий за приобретение суточного молодняка финальной формы яичного направления, полученного от племенной птицы",
    "Заявка на получение субсидий за приобретенное двуполое семя племенных быков молочного/молочно-мясного и мясного/мясо-молочного направлений",
    "Заявка на получение субсидий за приобретенное однополое семя племенных быков молочного/молочно-мясного и мясного/мясо-молочного направлений",
    "Заявка на получение субсидий на Удешевление стоимости затрат на корма сельскохозяйственным животным (маточное поголовье крупного рогатого скота молочного и молочно-мясного направления) по Туркестанской области",
    "Заявка на получение субсидий на ведение селекционной и племенной работы с маточным и ремонтным поголовьем свиней",
    "Заявка на получение субсидий на ведение селекционной и племенной работы с племенным маточным поголовьем крупного рогатого скота",
    "Заявка на получение субсидий на ведение селекционной и племенной работы с товарным маточным поголовьем крупного рогатого скота",
    "Заявка на получение субсидий на удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье верблюдов)",
    "Заявка на получение субсидий на удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье крупного рогатого скота молочного и молочно-мясного направления)",
    "Заявка на получение субсидий на удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье крупного рогатого скота)",
    "Заявка на получение субсидий на удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье лошадей)",
    "Заявка на получение субсидий на удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье мелкого рогатого скота)",
    "Заявка на получение субсидий на удешевление стоимости крупного рогатого скота (в том числе племенные мужские особи молочных или молочно-мясных пород), реализованных или перемещенных на откорм в откормочные площадки или мясоперерабатывающие предприятия с убойной мощностью не менее 50 голов крупного рогатого скота в сутки",
    "Заявка на получение субсидий на удешевление стоимости мелкого рогатого скота мужских особей, реализованных или перемещенных на откорм в откормочные площадки или мясоперерабатывающие предприятия с убойной мощностью не менее 300 голов овец в сутки",
    "Заявка на получение субсидий на удешевление стоимости производства молока (верблюжье)",
    "Заявка на получение субсидий на удешевление стоимости производства молока (кобылье)",
    "Заявка на получение субсидий на удешевление стоимости производства молока (коровье) Сельскохозяйственный кооператив",
    "Заявка на получение субсидий на удешевление стоимости производства молока (коровье) с фуражным поголовьем коров от 400 голов",
    "Заявка на получение субсидий на удешевление стоимости производства молока (коровье) с фуражным поголовьем коров от 50 голов",
    "Заявка на получение субсидий на удешевление стоимости производства молока (коровье) с фуражным поголовьем коров от 600 голов",
    "Заявка на получение субсидий на удешевление стоимости производства мяса птицы (мясо курицы) фактическое производство от 10 000 тонн",
    "Заявка на получение субсидий на удешевление стоимости производства мяса птицы (мясо курицы) фактическое производство от 15 000 тонн",
    "Заявка на получение субсидий на удешевление стоимости производства мяса птицы (мясо курицы) фактическое производство от 5 000 тонн",
    "Заявка на получение субсидий на удешевление стоимости производства мяса птицы (мясо курицы) фактическое производство от 500 тонн",
    "Заявка на получение субсидий племенным и дистрибьютерным центрам за услуги по искусственному осеменению маточного поголовья крупного рогатого скота в крестьянских (фермерских) хозяйствах и сельскохозяйственных кооперативах",
    "Заявка на получение субсидий племенным и дистрибьютерным центрам за услуги по искусственному осеменению маточного поголовья овец в хозяйствах и сельскохозяйственных кооперативах",
    "Заявка на удешевление затрат при выращивании племенного молодняка крупного рогатого скота мясного направления",
    "Заявка на удешевление затрат при выращивании племенного молодняка мелкого рогатого скота",
    "Заявка на удешевление стоимости мелкого рогатого скота мужской особи, реализованного на откорм в откормочные площадки или на убой в мясоперерабатывающие предприятия (сезонные поставки)",
    "Заявка на удешевление стоимости производства меда",
    "Заявка на удешевление стоимости свиней, реализованных или перемещенных на убой в мясоперерабатывающие предприятия или на убойные пункты",
    "Заявка на удешевление стоимости тонкой и полутонкой шерсти, реализованной на переработку (шерсть от 60 качества)",
    "Удешевление стоимости затрат на корма маточному поголовью сельскохозяйственных животных (маточное поголовье крупного рогатого скота молочного и молочно-мясного направления)",
]
VALID_DISTRICTS = [
    "Алматинский район", "Шуский район", "Талгарский район", "Карасайский район",
    "Енбекшиказахский район", "Илийский район", "Уйгурский район",
]

FEATURE_NAMES_ORDER = [
    "Дата поступления",
    "Область",
    "Акимат",
    "Направление субсидии",
    "Наименование субсидирования",
    "Норматив",
    "Причитающая сумма",
    "Район хозяйства",
]

FEATURE_LABELS_SHORT = {
    "Дата поступления": "Дата подачи",
    "Область": "Область",
    "Акимат": "Акимат",
    "Направление субсидии": "Направление",
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
    position: relative;
    overflow: visible;
}
.shap-pos { background: #e8f8f0; border-left: 4px solid var(--success); }
.shap-neg { background: #fff0f2; border-left: 4px solid var(--danger); }
.shap-val { font-weight: 700; min-width: 60px; }
.shap-pos .shap-val { color: var(--success); }
.shap-neg .shap-val { color: var(--danger); }

/* ── SHAP группы ── */
.shap-group {
    margin-bottom: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: visible;
    position: relative;
    z-index: 1;
}
.shap-group-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: #f4f6fb;
    border-bottom: 1px solid var(--border);
    font-weight: 700;
    font-size: 13px;
    color: var(--primary);
    cursor: pointer;
    user-select: none;
}
.shap-group-header:hover { background: #e8eef8; }
.shap-group-header .group-emoji { font-size: 16px; }
.shap-group-header .group-count {
    margin-left: auto;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 10px;
}
.shap-group-body { padding: 8px 10px; overflow: visible; }

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

/* ── Citation icons (скрепки) ── */
.citation-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    margin: 0 3px;
    background: var(--accent);
    border-radius: 50%;
    cursor: pointer;
    position: relative;
    vertical-align: middle;
    transition: all 0.2s ease;
    z-index: 1;
}
.citation-icon:hover {
    background: var(--primary);
    transform: scale(1.15);
    z-index: 999999;
}
.citation-icon svg {
    width: 13px;
    height: 13px;
    fill: white;
    pointer-events: none;
}
.citation-popup {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: calc(100% + 12px);
    left: 50%;
    transform: translateX(-50%) translateY(5px);
    background: white;
    border: 2px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    z-index: 999998;
    min-width: 320px;
    max-width: 550px;
    transition: visibility 0s, opacity 0.2s ease, transform 0.2s ease;
    pointer-events: none;
}
.citation-icon:hover .citation-popup {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
    z-index: 999998;
}
.citation-popup::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 9px solid transparent;
    border-top-color: var(--accent);
}
.citation-popup::before {
    content: '';
    position: absolute;
    top: calc(100% + 1px);
    left: 50%;
    transform: translateX(-50%);
    border: 8px solid transparent;
    border-top-color: white;
    z-index: 1;
}
.citation-popup-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.citation-popup-header .icon {
    font-size: 16px;
}
.citation-popup-header .title {
    font-weight: 700;
    font-size: 12px;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.citation-quote {
    background: #f4f6fb;
    border-left: 4px solid var(--accent);
    padding: 12px 14px;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    color: #222;
    margin-bottom: 10px;
    font-style: italic;
    max-height: 200px;
    overflow-y: auto;
}
.citation-line-number {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}
.citation-line-number::before {
    content: '📄';
    font-size: 12px;
}
.citation-explanation {
    font-size: 12px;
    color: var(--text);
    line-height: 1.5;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed var(--border);
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Поиск цитат из PDF для SHAP-факторов ──
_SHAP_KEYWORDS = {
    "debt_load_ratio": ["долг", "ebitda", "кредит", "заем", "долговая нагрузка"],
    "subsidy_dependence_index": ["субсид", "зависимост", "дотаци", "поддержк"],
    "gross_output_growth_yoy": ["рост", "валов", "продукци", "увеличен", "прирост"],
    "pedigree_ratio": ["племен", "пород", "племенно поголов", "класс элит"],
    "historical_survival_rate": ["сохранн", "выживаем", "падеж", "смертн", "гибель"],
    "veterinary_compliance": ["ветеринар", "вет", "вакцин", "болезн", "инфекц", "анализ"],
    "land_to_livestock_ratio": ["га/", "гектар", "пастбищ", "земель", "участок", "аренд"],
    "years_in_operation": ["основан", "работает с", "стаж", "лет работы", "год создан"],
    "previous_subsidies_count": ["субсид", "ранее получен", "предыдущ"],
    "livestock_count": ["голов", "поголов", "стад", "крс", "овц", "лошад"],
    "log_amount": ["сумм", "тенге", "стоимост", "цена"],
    "grazing_norm_deviation": ["нагрузк", "пастбищ", "норм", "отклонен"],
    "natural_loss_risk_score": ["риск", "падеж", "смертн", "естественн убыл"],
}

def _find_pdf_citations_for_shap(pdf_text: str, feature_name: str, pdf_names: list[str] | None = None) -> list[dict]:
    """Ищет в тексте PDF цитаты, относящиеся к SHAP-фактору."""
    if not pdf_text or not pdf_text.strip():
        return []
    
    keywords = _SHAP_KEYWORDS.get(feature_name, [])
    if not keywords:
        return []
    
    lines = pdf_text.split('\n')
    citations = []
    
    # Определяем к какому документу относится строка
    # Формат: === filename.pdf ===
    current_doc = None
    doc_markers = {}  # line_num -> doc_name
    
    for i, line in enumerate(lines, 1):
        import re
        doc_match = re.match(r'^===\s+(.+\.pdf)\s+===$', line.strip(), re.IGNORECASE)
        if doc_match:
            current_doc = doc_match.group(1)
        doc_markers[i] = current_doc
    
    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        # Пропускаем строки-разделители документов
        if re.match(r'^===\s+.+\.pdf\s+===$', line.strip(), re.IGNORECASE):
            continue
        # Проверяем наличие хотя бы одного ключевого слова
        if any(kw in line_lower for kw in keywords):
            # Берём строку +/- 2 для контекста
            start = max(0, i - 3)
            end = min(len(lines), i + 2)
            context = '\n'.join(lines[start:end]).strip()
            if len(context) > 300:
                context = context[:300] + "..."
            
            # Определяем имя документа
            doc_name = doc_markers.get(i)
            if not doc_name:
                # Ищем ближайший документ выше
                for prev_line in range(i - 1, 0, -1):
                    if prev_line in doc_markers and doc_markers[prev_line]:
                        doc_name = doc_markers[prev_line]
                        break
            
            citations.append({
                "line_number": i,
                "quote": context,
                "doc_name": doc_name or "Неизвестный документ",
                "explanation": f"Найдено по ключевым словам: {', '.join(keywords[:3])}"
            })
            # Ограничиваем 2 цитатами
            if len(citations) >= 2:
                break
    
    return citations

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
        st.error(T("api_error_timeout"))
        return None
    except Exception as e:
        st.error(T("api_error_generic").format(error=e))
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
        st.error(T("api_error_doc_timeout"))
        return None
    except Exception as e:
        st.error(T("api_error_generic").format(error=e))
        return None

def _api_get(endpoint: str) -> dict | list | None:
    try:

        _to = 120 if endpoint.rstrip("/").endswith("applications") else 10
        r = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, timeout=_to)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(T("api_error_generic").format(error=e))
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
    "grazing_norm_deviation": "Нагрузка пастбищ",
    "natural_loss_risk_score": "Риск падежа",
    "livestock_count": "Поголовье (расч.)",
    "log_amount": "Масштаб заявки",
    "direction_code": "Направление",
    "region_encoded": "Регион",
    "is_pedigree": "Племенная субсидия",
    "is_producer": "Производители",
    "hour_submitted": "Час подачи",
    "month_submitted": "Месяц подачи",
}

# ── Группировка факторов по категориям ──
SHAP_GROUPS = [
    {
        "key": "finance",
        "label": "Финансы",
        "emoji": "💰",
        "features": ["debt_load_ratio", "subsidy_dependence_index", "log_amount"],
    },
    {
        "key": "production",
        "label": "Производство",
        "emoji": "🌾",
        "features": [
            "gross_output_growth_yoy", "pedigree_ratio", "historical_survival_rate",
            "land_to_livestock_ratio", "livestock_count",
        ],
    },
    {
        "key": "vet_risks",
        "label": "Ветеринария и риски",
        "emoji": "🐄",
        "features": ["veterinary_compliance", "natural_loss_risk_score", "grazing_norm_deviation"],
    },
    {
        "key": "experience",
        "label": "Опыт и история",
        "emoji": "📋",
        "features": ["years_in_operation", "previous_subsidies_count"],
    },
    {
        "key": "context",
        "label": "Контекст заявки",
        "emoji": "📌",
        "features": [
            "direction_code", "region_encoded", "is_pedigree", "is_producer",
            "hour_submitted", "month_submitted",
        ],
    },
]

def _build_feature_to_group():
    mapping = {}
    for group in SHAP_GROUPS:
        for feat in group["features"]:
            mapping[feat] = group["key"]
    return mapping

FEATURE_TO_GROUP = _build_feature_to_group()

def _shap_max_abs(all_shap: dict) -> float:
    if not all_shap:
        return 1e-9
    return max(abs(float(v)) for v in all_shap.values()) or 1e-9

def _shap_to_display_points(shap_val: float, max_abs: float, scale: float = 20.0) -> int:
    if max_abs < 1e-9:
        return 0
    return int(round(scale * float(shap_val) / max_abs))

_PROFILE_SENTINEL = float("nan")   # marks "data not available" in the profile dict

def _normalized_profile_from_raw(raw: dict) -> dict[str, float]:
    """Return a 0-100 normalised profile dict.

    Fields that are None in raw_features_used are set to _PROFILE_SENTINEL so
    callers can distinguish "truly zero" from "data missing".
    """
    if not raw:
        return {}

    def _get(key: str):
        v = raw.get(key)
        return None if v is None else float(v)

    g    = _get("gross_output_growth_yoy")
    pr   = _get("pedigree_ratio")
    land = _get("land_to_livestock_ratio")
    surv = _get("historical_survival_rate")
    debt = _get("debt_load_ratio")
    sub  = _get("subsidy_dependence_index")
    years = _get("years_in_operation")
    vet  = _get("veterinary_compliance")
    graz = _get("grazing_norm_deviation")
    risk = _get("natural_loss_risk_score")

    import math
    S = _PROFILE_SENTINEL

    def _norm_growth(v):
        if v is None:
            return S
        return float(np.clip((v + 0.35) / 1.0 * 100, 0, 100))

    return {
        "Рост продукции":      _norm_growth(g),
        "Племенное поголовье": S if pr   is None else min(pr   * 100, 100),
        "Земля":               S if land is None else min(land / 8.0 * 100, 100),
        "Выживаемость":        S if surv is None else min(surv * 100, 100),
        "Долг (инверт.)":      S if debt is None else max(0.0, (1.0 - debt / 5.0)) * 100,
        "Независимость":       S if sub  is None else max(0.0, (1.0 - sub) * 100),
        "Стаж":                S if years is None else min(years / 20.0 * 100, 100),
        "Ветеринария":         S if vet  is None else min(vet  * 100, 100),
        "Пастбища (норма)":    S if graz is None else float(np.clip((graz + 2.0) / 4.0 * 100, 0, 100)),
        "Риск падежа (инв.)":  S if risk is None else max(0.0, (1.0 - risk / 3.0)) * 100,
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
        "Пастбища (норма)",
        "Риск падежа (инв.)",
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
    # ── ТУМБЛЕР ЯЗЫКА ──
    lang_options = {"🇷🇺 Русский": "ru", "🇰🇿 Қазақша": "kz"}
    lang_labels = list(lang_options.keys())
    current_lang_label = "🇷🇺 Русский" if lang() == "ru" else "🇰🇿 Қазақша"
    selected_lang_label = st.selectbox(
        T("lang_toggle"),
        options=lang_labels,
        index=lang_labels.index(current_lang_label),
        label_visibility="collapsed",
    )
    st.session_state.lang = lang_options[selected_lang_label]

    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:40px;">🌾</div>
        <div style="font-family:'Montserrat',sans-serif; font-size:18px; font-weight:800; color:#fff; margin-top:6px;">
            SmartAgro Score
        </div>
        <div style="font-size:11px; color:#8ab4e8; margin-top:4px; letter-spacing:0.5px;">
            {T("sidebar_subtitle")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"**{T('sidebar_user')}**")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.08); padding:10px 14px; border-radius:8px; font-size:13px;">
        👤 {T('sidebar_user_name')}<br>
        <span style="color:#8ab4e8; font-size:12px;">{T('sidebar_user_dept')}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    total = len(st.session_state.applications)
    green = sum(1 for a in st.session_state.applications if a.get("zone") == "green")
    yellow = sum(1 for a in st.session_state.applications if a.get("zone") == "yellow")
    red = sum(1 for a in st.session_state.applications if a.get("zone") == "red")

    st.markdown(f"**{T('sidebar_stats')}**")
    st.markdown(f"""
    <div style="font-size:13px; line-height:2;">
        {T('sidebar_total')}: <b>{total}</b><br>
        {T('sidebar_green')}: <b>{green}</b><br>
        {T('sidebar_yellow')}: <b>{yellow}</b><br>
        {T('sidebar_red')}: <b>{red}</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<span style='font-size:11px; color:#8ab4e8;'>{T('sidebar_updated')}: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)

st.markdown(f"""
<div class="gov-header">
    <div>
        <div class="logo-text">{T('header_title')}</div>
        <div class="logo-sub">{T('header_subtitle')}</div>
    </div>
    <div class="badge">{T('header_badge')}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    T("tab_applications"),
    T("tab_shortlist"),
    T("tab_profile"),
    T("tab_api"),
])

with tab1:
    st.markdown(f"### {T('tab_applications')} — {T('applications_subtitle')}")

    col_sync, col_status = st.columns([1, 2])

    with col_sync:
        if st.button(T("form_test_apps"), type="primary", use_container_width=True):
            with st.spinner(T("form_loading_test")):
                time.sleep(1.2)
                data = _api_post(
                    "/api/v1/giss/sync",
                    {},
                    timeout=(10, 120),
                )
                if data:
                    st.success(T("sync_success").format(count=data['synced_count']))
                    _refresh_apps()
                    st.rerun()

    with col_status:
        st.markdown(f"""
        <div class="giss-status">
            <span class="giss-dot"></span>
            {T('form_api_online')}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.expander(T("form_expand_submit"), expanded=True):

        st.markdown(f"### {T('form_section_a')}")
        st.caption(T("form_section_a_desc"))

        a1, a2, a3 = st.columns(3)
        with a1:
            man_bin = st.text_input(T("form_field_bin"), placeholder="123456789012")
            man_company = st.text_input(T("form_field_company"), placeholder="ТОО «Агро-Нур»")
        with a2:
            man_region = st.selectbox(T("form_field_region"), VALID_REGIONS)
            man_direction = st.selectbox(T("form_field_direction"), VALID_DIRECTIONS)
        with a3:
            man_subsidy = st.selectbox(
                T("form_field_subsidy"),
                [T("form_field_subsidy_other")] + VALID_SUBSIDY_NAMES,
            )
            if man_subsidy == "Другое (ввести вручную)":
                man_subsidy = st.text_input(
                    T("form_field_subsidy_input"),
                    placeholder="Например: Субсидия на кормовые добавки",
                )
            man_amount = st.number_input(
                T("form_field_amount"),
                min_value=100_000, max_value=500_000_000,
                value=10_000_000, step=500_000,
            )

        st.divider()

        st.markdown(f"### {T('form_section_b')}")
        st.caption(T("form_section_b_desc"))

        b1, b2, b3 = st.columns(3)
        with b1:
            farm_size = st.selectbox(
                T("form_field_farm_size"),
                [T("opt_not_specified"), T("opt_farm_small"), T("opt_farm_medium"), T("opt_farm_large")],
            )
            debt_level = st.selectbox(
                T("form_field_debt"),
                [T("opt_unknown"), T("opt_debt_low"), T("opt_debt_medium"), T("opt_debt_high")],
            )
        with b2:
            subsidy_exp = st.selectbox(
                T("form_field_subsidy_exp"),
                [T("opt_unknown"), T("opt_first_time"), T("opt_prev_1_2"), T("opt_prev_3plus")],
            )
            vet_status = st.selectbox(
                T("form_field_vet"),
                [T("opt_unknown"), T("opt_vet_ok"), T("opt_vet_minor"), T("opt_vet_serious")],
            )
        with b3:
            growth_choice = st.selectbox(
                T("form_field_growth"),
                [T("opt_unknown"), T("opt_growth_decline"), T("opt_growth_flat"), T("opt_growth_moderate"), T("opt_growth_high")],
            )
            pedigree_choice = st.selectbox(
                T("form_field_pedigree"),
                [T("opt_unknown"), T("opt_pedigree_none"), T("opt_pedigree_low"), T("opt_pedigree_mid"), T("opt_pedigree_high")],
            )

        st.divider()

        st.markdown(f"### {T('form_section_docs')}")
        st.caption(T("form_section_docs_desc"))

        uploaded_docs = st.file_uploader(
            T("form_section_docs_hint"),
            type=["pdf"],
            accept_multiple_files=True,
            key="bulk_docs",
        )

        if uploaded_docs:
            total_mb = sum(len(d.getvalue()) for d in uploaded_docs) / 1_048_576
            if total_mb > 200:
                st.error(T("form_filesize_error").format(mb=total_mb))
                uploaded_docs = []
            else:
                st.success(T("form_filesize_ok").format(count=len(uploaded_docs), mb=total_mb))
                with st.expander(T("form_file_list"), expanded=False):
                    for d in uploaded_docs:
                        sz_kb = len(d.getvalue()) / 1024
                        st.caption(f"📄 {d.name} — {sz_kb:.1f} КБ")

        if st.button(T("form_btn_submit"), type="primary", use_container_width=True):
            if not man_bin.strip() or not man_company.strip():
                st.warning(T("form_warn_required"))
            else:

                _debt_map = {
                    T("opt_unknown"):                    None,
                    T("opt_debt_low"):                   0.8,
                    T("opt_debt_medium"):                2.2,
                    T("opt_debt_high"):                  3.8,
                }
                _vet_map = {
                    T("opt_unknown"):                                None,
                    T("opt_vet_ok"):                                 0.97,
                    T("opt_vet_minor"):                              0.72,
                    T("opt_vet_serious"):                            0.45,
                }
                _growth_map = {
                    T("opt_unknown"):          None,
                    T("opt_growth_decline"):   -0.12,
                    T("opt_growth_flat"):      0.03,
                    T("opt_growth_moderate"):  0.12,
                    T("opt_growth_high"):      0.28,
                }
                _pedigree_map = {
                    T("opt_unknown"):    None,
                    T("opt_pedigree_none"): 0.10,
                    T("opt_pedigree_low"):  0.40,
                    T("opt_pedigree_mid"):  0.75,
                    T("opt_pedigree_high"): 0.95,
                }
                _subsidy_exp_map = {
                    T("opt_unknown"):    None,
                    T("opt_first_time"): 0,
                    T("opt_prev_1_2"):   1,
                    T("opt_prev_3plus"): 5,
                }
                _farm_years_map = {
                    T("opt_not_specified"):  None,
                    T("opt_farm_small"):     3,
                    T("opt_farm_medium"):    7,
                    T("opt_farm_large"):     15,
                }

                payload = {
                    "bin_iin":                  man_bin.strip(),
                    "company_name":             man_company.strip(),
                    "region":                   man_region,
                    "subsidy_type":             man_subsidy,
                    "direction":                man_direction,
                    "requested_amount":         man_amount,
                    "source_system":            "manual",
                    "debt_load_ratio":          _debt_map.get(debt_level, None),
                    "veterinary_compliance":    _vet_map.get(vet_status, None),
                    "gross_output_growth_yoy":  _growth_map.get(growth_choice, None),
                    "pedigree_ratio":           _pedigree_map.get(pedigree_choice, None),
                    "previous_subsidies_count": _subsidy_exp_map.get(subsidy_exp, None),
                    "years_in_operation":       _farm_years_map.get(farm_size, None),
                    "normative":                15_000.0,
                }

                if uploaded_docs:
                    total_mb_chk = sum(len(d.getvalue()) for d in uploaded_docs) / 1_048_576
                    if total_mb_chk > 200:
                        st.error(T("form_error_filesize"))
                    else:
                        files = [
                            ("documents", (d.name, d.getvalue(), "application/pdf"))
                            for d in uploaded_docs
                        ]
                        with st.spinner(T("form_spinner_docs").format(count=len(uploaded_docs))):
                            result = _api_post_multipart(
                                "/api/v1/score-with-documents",
                                data={"features_json": json.dumps(payload, ensure_ascii=False)},
                                files=files,
                            )
                else:
                    with st.spinner(T("form_spinner_score")):
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

                    st.success(T("form_success_submitted").format(icon=_ic, app_id=result['application_id'], score_detail=score_detail))
                    _chars = result.get("documents_text_chars") or 0
                    _ext_note = result.get("documents_extraction_note")
                    if uploaded_docs and _chars > 0:
                        st.caption(T("form_caption_pdf_chars").format(chars=_chars))
                    elif uploaded_docs:
                        st.warning(T("form_warn_pdf_missing") + (" " + _ext_note if _ext_note else ""))
                    st.session_state.selected_app_id = result["application_id"]
                    if _manual:
                        st.warning(T("form_warn_manual_review"))
                    _refresh_apps()
                    st.rerun()

    st.divider()
    st.markdown(f"#### {T('all_apps_title')}")

    _refresh_apps()

    if not st.session_state.applications:
        st.info(T("all_apps_empty"))
    else:
        apps_sorted = sorted(st.session_state.applications, key=lambda x: x.get("score", 0), reverse=True)

        # ── Фильтры ──
        all_regions = sorted(set(a.get("region", "") for a in apps_sorted if a.get("region")))
        all_subsidies = sorted(set(a.get("subsidy_type", "") for a in apps_sorted if a.get("subsidy_type")))

        filt_cols = st.columns([2, 2, 1, 1, 1])
        with filt_cols[0]:
            filt_region = st.selectbox(T("all_apps_filter_region"), [T("all_apps_zone_all")] + all_regions, index=0, key="filt_region", label_visibility="collapsed")
        with filt_cols[1]:
            filt_subsidy = st.selectbox(T("all_apps_filter_subsidy"), [T("all_apps_zone_all")] + all_subsidies, index=0, key="filt_subsidy", label_visibility="collapsed")
        with filt_cols[2]:
            filt_zone_label = st.selectbox(T("all_apps_filter_zone"), [T("all_apps_zone_all"), T("all_apps_zone_green"), T("all_apps_zone_yellow"), T("all_apps_zone_red")], index=0, key="filt_zone", label_visibility="collapsed")
            filt_zone = {"🟢 Green": "green", "🟡 Yellow": "yellow", "🔴 Red": "red"}.get(filt_zone_label)
        with filt_cols[3]:
            filt_decision_label = st.selectbox(T("all_apps_filter_decision"), [T("all_apps_decision_all"), T("all_apps_decision_approved"), T("all_apps_decision_rejected"), T("all_apps_decision_pending")], index=0, key="filt_decision", label_visibility="collapsed")
            filt_decision_map = {T("all_apps_decision_approved"): "approved", T("all_apps_decision_rejected"): "rejected", T("all_apps_decision_pending"): "pending"}
            filt_decision = filt_decision_map.get(filt_decision_label)
        with filt_cols[4]:
            filt_min_score = st.number_input(T("all_apps_filter_min_score"), min_value=0, max_value=100, value=0, step=5, key="filt_min_score", label_visibility="collapsed")

        # Применяем фильтры
        filtered = apps_sorted
        if filt_region != T("all_apps_zone_all"):
            filtered = [a for a in filtered if a.get("region") == filt_region]
        if filt_subsidy != T("all_apps_zone_all"):
            filtered = [a for a in filtered if a.get("subsidy_type") == filt_subsidy]
        if filt_zone is not None:
            filtered = [a for a in filtered if a.get("zone") == filt_zone]
        if filt_decision is not None:
            filtered = [a for a in filtered if st.session_state.decisions.get(a["application_id"], "") == filt_decision]
        filtered = [a for a in filtered if a.get("score", 0) >= filt_min_score]

        # Счётчик
        if len(filtered) != len(apps_sorted):
            st.caption(T("all_apps_found").format(found=len(filtered), total=len(apps_sorted)))

        for app in filtered:
            cat = app.get("zone", "yellow")
            icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(cat, "⚪")
            score = app.get("score", 0)
            decision_status = st.session_state.decisions.get(app["application_id"], "")
            dec_badge = ""
            if decision_status == "approved":
                dec_badge = T("all_apps_decision_approved")
            elif decision_status == "rejected":
                dec_badge = T("all_apps_decision_rejected")

            with st.container():
                c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
                with c1:
                    _demo_badge = T("all_apps_demo_tag") if app.get("is_demo") else ""
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
                    elif st.button(T("all_apps_btn_open"), key=f"open_{app['application_id']}"):
                        st.session_state.selected_app_id = app["application_id"]
                        st.info(T("all_apps_info_profile"))

with tab2:
    st.markdown(f"### {T('shortlist_title')}")

    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        budget = st.number_input(
            T("shortlist_budget_label"),
            min_value=1_000_000,
            max_value=10_000_000_000,
            value=100_000_000,
            step=5_000_000,
            help=T("shortlist_budget_help"),
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
                    <div class="m-label">{T('shortlist_metric_total')}</div>
                    <div class="m-value">{_fmt_tenge(total_sum)}</div>
                    <div class="m-delta">{len(apps)} {T('date')}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="m-label">Покрытие бюджетом</div>
                    <div class="m-value">{coverage:.0f}%</div>
                    <div class="m-delta">от общего объёма</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="m-label">{T('shortlist_metric_avg_score')}</div>
                    <div class="m-value">{np.mean([a['score'] for a in apps]):.0f}</div>
                    <div class="m-delta">{T('shortlist_metric_avg_desc')}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    apps_sorted = sorted(st.session_state.applications, key=lambda x: x.get("score", 0), reverse=True)

    if not apps_sorted:
        st.info(T("shortlist_empty"))
    else:
        # ── Фильтры ──
        all_regions = sorted(set(a.get("region", "") for a in apps_sorted if a.get("region")))
        all_subsidies = sorted(set(a.get("subsidy_type", "") for a in apps_sorted if a.get("subsidy_type")))

        filt_cols = st.columns([2, 2, 1, 1, 1])
        with filt_cols[0]:
            filt_region = st.selectbox(T("all_apps_filter_region"), [T("all_apps_zone_all")] + all_regions, index=0, key="filt2_region", label_visibility="collapsed")
        with filt_cols[1]:
            filt_subsidy = st.selectbox(T("all_apps_filter_subsidy"), [T("all_apps_zone_all")] + all_subsidies, index=0, key="filt2_subsidy", label_visibility="collapsed")
        with filt_cols[2]:
            filt_zone_label = st.selectbox(T("all_apps_filter_zone"), [T("all_apps_zone_all"), T("all_apps_zone_green"), T("all_apps_zone_yellow"), T("all_apps_zone_red")], index=0, key="filt2_zone", label_visibility="collapsed")
            filt_zone = {"🟢 Green": "green", "🟡 Yellow": "yellow", "🔴 Red": "red"}.get(filt_zone_label)
        with filt_cols[3]:
            filt_decision_label = st.selectbox(T("all_apps_filter_decision"), [T("all_apps_decision_all"), T("all_apps_decision_approved"), T("all_apps_decision_rejected"), T("all_apps_decision_pending")], index=0, key="filt2_decision", label_visibility="collapsed")
            filt_decision_map = {"✅ Одобрено": "approved", "❌ Отказано": "rejected", "⏳ Ожидание": "pending"}
            filt_decision = filt_decision_map.get(filt_decision_label)
        with filt_cols[4]:
            filt_min_score = st.number_input(T("all_apps_filter_min_score"), min_value=0, max_value=100, value=0, step=5, key="filt2_min_score", label_visibility="collapsed")

        # Применяем фильтры
        filtered = apps_sorted
        if filt_region != "Все":
            filtered = [a for a in filtered if a.get("region") == filt_region]
        if filt_subsidy != "Все":
            filtered = [a for a in filtered if a.get("subsidy_type") == filt_subsidy]
        if filt_zone is not None:
            filtered = [a for a in filtered if a.get("zone") == filt_zone]
        if filt_decision is not None:
            filtered = [a for a in filtered if st.session_state.decisions.get(a["application_id"], "") == filt_decision]
        filtered = [a for a in filtered if a.get("score", 0) >= filt_min_score]

        # Счётчик
        if len(filtered) != len(apps_sorted):
            st.caption(T("all_apps_found").format(found=len(filtered), total=len(apps_sorted)))

        rows_html = ""
        cumulative = 0.0
        cutoff_drawn = False

        for app in filtered:
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
                        {T('shortlist_budget_exhausted').format(budget=_fmt_tenge(budget), remaining=_fmt_tenge(remaining))}
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
                dec_cell = f"<span style='color:green; font-weight:700;'>{T('shortlist_table_approved')}</span>"
            elif decision == "rejected":
                dec_cell = f"<span style='color:red; font-weight:700;'>{T('shortlist_table_rejected')}</span>"
            else:
                dec_cell = f"<span style='color:#aaa;'>{T('all_apps_status_pending')}</span>"

            date_str = app.get("application_date", app.get("calculated_at", "")[:10])
            _demo_tag = f' <small style="color:#888;">{T("all_apps_demo_tag").strip()}</small>' if app.get("is_demo") else ""

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
                    <th>{T('shortlist_table_date')}</th>
                    <th>{T('shortlist_table_company')}</th>
                    <th>{T('shortlist_table_region')}</th>
                    <th>{T('shortlist_table_subsidy')}</th>
                    <th>{T('shortlist_table_amount')}</th>
                    <th>{T('shortlist_table_score')}</th>
                    <th>{T('shortlist_table_decision')}</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div style="margin-top:10px; font-size:12px; color:#888;">
            {T('shortlist_legend')}
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.divider()
        scores = [a["score"] for a in filtered]
        fig = go.Figure()
        colors = ["#1a7a4a" if s >= 80 else ("#e8a800" if s >= 50 else "#b5001f") for s in scores]
        fig.add_trace(go.Bar(
            x=[a["company_name"][:20] for a in filtered],
            y=scores,
            marker_color=colors,
            text=[f"{s:.0f}" for s in scores],
            textposition="outside",
        ))
        fig.add_hline(y=80, line_dash="dot", line_color="#1a7a4a",
                      annotation_text=T("shortlist_chart_hline_80"), annotation_position="right")
        fig.add_hline(y=50, line_dash="dot", line_color="#e8a800",
                      annotation_text=T("shortlist_chart_hline_50"), annotation_position="right")
        fig.update_layout(
            title=T("shortlist_chart_title"),
            xaxis_title=T("shortlist_chart_x"),
            yaxis_title=T("shortlist_chart_y"),
            height=350,
            plot_bgcolor="#f8faff",
            paper_bgcolor="#ffffff",
            font=dict(family="Golos Text", size=12),
            margin=dict(t=50, b=60),
            yaxis=dict(range=[0, 110]),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown(f"### {T('profile_title')}")

    apps = st.session_state.applications

    if not apps:
        st.info(T("profile_no_apps"))
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
            T("profile_select_label"),
            options=list(app_names.keys()),
            format_func=lambda x: app_names[x],
            index=list(app_names.keys()).index(default_id),
        )
        st.session_state.selected_app_id = selected_id

        app = next((a for a in apps if a["application_id"] == selected_id), None)

        if app is None:
            st.warning(T("profile_not_found"))
        else:
            cat = app.get("zone", "yellow")
            score = app.get("score", 0)
            score_ml = app.get("score_ml", score)
            compliance_bonus = app.get("compliance_bonus", 0)
            cat_colors = {"green": "#1a7a4a", "yellow": "#b36200", "red": "#b5001f"}
            cat_bg = {"green": "#e8f8f0", "yellow": "#fffbea", "red": "#fff0f2"}
            cat_labels = {"green": T("profile_verdict_green"), "yellow": T("profile_verdict_yellow"), "red": T("profile_verdict_red")}

            bonus_str = ""
            if compliance_bonus != 0:
                sign = "+" if compliance_bonus > 0 else ""
                bonus_color = "#1a7a4a" if compliance_bonus > 0 else "#b5001f"
                bonus_label = T("profile_score_bonus").format(sign=sign, bonus=f"{compliance_bonus:.1f}")
                bonus_str = f'<span style="font-size:13px; color:{bonus_color}; font-weight:700; margin-left:10px;">{bonus_label}</span>'

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
                        <div style="font-size:11px; color:#888; letter-spacing:0.5px;">{T('profile_score_label')} {bonus_str}</div>
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
                st.success(T("profile_pdf_text_saved").format(chars=len(_det)))
            else:
                if _note:
                    st.warning(T("profile_pdf_text_missing_note").format(note=_note))
                else:
                    st.info(T("profile_pdf_text_missing"))

            col_btn, col_status_g = st.columns([1, 3])
            with col_btn:
                if st.button(T("profile_gemini_btn"), use_container_width=True, key=f"gemini_btn_{selected_id}"):
                    if gemini_key:
                        with st.spinner(T("profile_gemini_spinner")):
                            opinion = generate_gemini_expert_opinion(app, gemini_key)
                            st.session_state[opinion_key] = opinion
                    else:
                        st.warning(T("profile_gemini_warn_key"))

            opinion_text = st.session_state.get(opinion_key)
            if opinion_text:
                verdict_color = {"green": "#1a7a4a", "yellow": "#b36200", "red": "#b5001f"}.get(cat, "#333")
                
                # Пытаемся распарсить JSON с цитатами
                citations_data = None
                conclusion_text = opinion_text
                
                try:
                    parsed = json.loads(opinion_text)
                    if isinstance(parsed, dict) and "conclusion" in parsed:
                        conclusion_text = parsed.get("enriched_conclusion", parsed.get("conclusion", ""))
                        citations_data = parsed.get("citations", parsed.get("citations_list", []))
                except (json.JSONDecodeError, TypeError):
                    # Это обычный текст (старый формат)
                    pass
                
                # Заменяем маркеры [CITATION:N] на иконки скрепок
                def render_citation_icon(point_num):
                    """Создаёт HTML для иконки скрепки с popup."""
                    if not citations_data:
                        return ""
                    
                    # Находим цитаты для этого пункта
                    point_citations = [c for c in citations_data if c.get("point_number") == point_num]
                    if not point_citations:
                        return ""
                    
                    # Берём первую цитату для отображения
                    citation = point_citations[0]
                    quote = citation.get("quote", "Цитата не найдена")
                    line_num = citation.get("line_number", "N/A")
                    explanation = citation.get("explanation", "")
                    
                    return f'''
                    <span class="citation-icon">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v6c0 .55-.45 1-1 1s-1-.45-1-1V5c0-.28-.22-.5-.5-.5s-.5.22-.5.5v12.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V6c0-.55-.45-1-1-1s-1 .45-1 1z"/>
                        </svg>
                        <div class="citation-popup">
                            <div class="citation-popup-header">
                                <span class="icon">📎</span>
                                <span class="title">Источник (строка {line_num})</span>
                            </div>
                            <div class="quote">"{quote}"</div>
                            <div class="citation-line-number">Строка в документе: {line_num}</div>
                            {f'<div class="citation-explanation">{explanation}</div>' if explanation else ''}
                        </div>
                    </span>
                    '''
                
                # Заменяем все [CITATION:N] на иконки
                import re
                def replace_citation_marker(match):
                    point_num = int(match.group(1))
                    return render_citation_icon(point_num)
                
                conclusion_html = re.sub(r'\[CITATION:(\d+)\]', replace_citation_marker, conclusion_text)
                
                st.markdown(f"""
                <div style="background:#f9faff; border:1.5px solid {verdict_color};
                     border-left: 5px solid {verdict_color};
                     border-radius:10px; padding:20px 24px; margin: 12px 0 20px 0;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                        <span style="font-size:20px;">🤖</span>
                        <span style="font-family:'Montserrat',sans-serif; font-weight:700;
                             font-size:14px; color:{verdict_color};">
                            {T('profile_gemini_label')}
                        </span>
                        <span style="font-size:11px; color:#aaa; margin-left:auto;">
                            {T('profile_gemini_source')}
                        </span>
                    </div>
                    <div style="font-size:14px; line-height:1.8; color:#1a2340; white-space:pre-wrap;">{conclusion_html}</div>
                </div>
                """, unsafe_allow_html=True)

            prof_col1, prof_col2 = st.columns([3, 2])

            with prof_col1:
                raw_features = app.get("raw_features_used") or {}
                pdf_ok = bool((app.get("documents_extracted_text") or "").strip())
                st.markdown(f"#### {T('profile_indicators_title')}")
                st.caption(T("profile_indicators_caption"))
                if pdf_ok:
                    st.caption(T("profile_indicators_pdf_caption"))

                import math as _math
                prof = _normalized_profile_from_raw(raw_features)
                bar_labels = list(prof.keys())
                # Separate known values from missing ones for correct rendering
                bar_vals_plot  = [0.0 if _math.isnan(v) else v for v in [prof[k] for k in bar_labels]]
                bar_text       = [
                    "N/A" if _math.isnan(prof[k]) else f"{prof[k]:.0f}"
                    for k in bar_labels
                ]
                bar_colors     = [
                    "#b0bec5" if _math.isnan(prof[k]) else "#0072CE"
                    for k in bar_labels
                ]
                fig_bars = go.Figure(go.Bar(
                    x=bar_vals_plot,
                    y=bar_labels,
                    orientation="h",
                    marker_color=bar_colors,
                    text=bar_text,
                    textposition="outside",
                    customdata=bar_text,
                    hovertemplate="%{y}: %{customdata}<extra></extra>",
                ))
                fig_bars.update_layout(
                    title=T("profile_bar_title"),
                    height=max(280, len(bar_labels) * 28),
                    plot_bgcolor="#f8faff",
                    paper_bgcolor="#ffffff",
                    margin=dict(t=50, b=30, l=10, r=50),
                    font=dict(family="Golos Text", size=11),
                    xaxis=dict(range=[0, 115], title=T("profile_bar_scale")),
                )
                if any(_math.isnan(prof[k]) for k in bar_labels):
                    st.caption(T("profile_indicators_na_caption"))
                st.plotly_chart(fig_bars, use_container_width=True)

                _yoy_raw = raw_features.get("gross_output_growth_yoy")
                yoy_available = _yoy_raw is not None
                yoy = float(_yoy_raw) if yoy_available else 0.0
                cy = datetime.now().year
                idx_prev, idx_curr = 100.0, 100.0 * (1.0 + yoy)
                fig_yoy = go.Figure()
                fig_yoy.add_trace(go.Scatter(
                    x=[cy - 1, cy],
                    y=[idx_prev, idx_curr],
                    name=T("profile_yoy_index"),
                    line=dict(color="#003580", width=3),
                    mode="lines+markers",
                    marker=dict(size=10),
                ))
                fig_yoy.update_layout(
                    title=(
                        T("profile_yoy_title")
                        if yoy_available
                        else T("profile_yoy_title_na")
                    ),
                    height=240,
                    plot_bgcolor="#f8faff",
                    paper_bgcolor="#ffffff",
                    margin=dict(t=50, b=30, l=40, r=20),
                    font=dict(family="Golos Text", size=11),
                    yaxis_title=T("profile_yoy_index_label"),
                    xaxis_title=T("profile_yoy_year"),
                    showlegend=False,
                )
                if not yoy_available:
                    st.caption(T("profile_yoy_na_caption"))
                st.plotly_chart(fig_yoy, use_container_width=True)

                radar_order = _radar_order_labels()
                radar_vals = [
                    0.0 if _math.isnan(prof.get(lab, _PROFILE_SENTINEL)) else prof.get(lab, 0.0)
                    for lab in radar_order
                ]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_order + [radar_order[0]],
                    fill="toself",
                    fillcolor="rgba(0,114,206,0.18)",
                    line=dict(color="#0072CE", width=2.5),
                ))
                fig_radar.update_layout(
                    title=T("profile_radar_title"),
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=340, paper_bgcolor="#ffffff",
                    font=dict(family="Golos Text", size=11),
                    margin=dict(t=50),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with prof_col2:

                st.markdown(f"#### {T('profile_shap_title')}")
                st.markdown(f"""
                <div style="background:#f0f4ff; border:1px solid #dde3ef; border-radius:8px;
                     padding:10px 14px; font-size:12px; color:#6b7a99; margin-bottom:12px;">
                    {T('profile_shap_info')}
                </div>
                """, unsafe_allow_html=True)
                _dc = int(app.get("documents_text_chars") or 0)
                if _dc > 0:
                    st.caption(T("profile_shap_doc_chars").format(chars=_dc))
                elif app.get("documents_pdf_count", 0):
                    st.caption(T("profile_shap_doc_count"))
                else:
                    st.caption(T("profile_shap_doc_none"))

                top_pos = app.get("top_positive_factors", [])
                top_neg = app.get("top_negative_factors", [])
                expl_by_feature = {item["feature"]: item for item in (top_pos + top_neg) if item.get("feature")}

                all_shap = app.get("all_shap_values") or {}
                max_abs = _shap_max_abs(all_shap)
                raw_for_shap = app.get("raw_features_used") or {}
                
                # Получаем текст PDF для поиска цитат
                pdf_text = app.get("documents_extracted_text") or ""
                pdf_names = app.get("documents_pdf_names") or []

                shap_html = ""
                if all_shap:
                    # Собираем факторы по группам
                    grouped = {g["key"]: [] for g in SHAP_GROUPS}
                    for fname, shap_raw in all_shap.items():
                        gkey = FEATURE_TO_GROUP.get(fname, "context")
                        grouped[gkey].append((fname, float(shap_raw)))

                    # Сортируем внутри группы по абсолютному вкладу
                    for gkey in grouped:
                        grouped[gkey].sort(key=lambda x: abs(x[1]), reverse=True)

                    # Рендерим только группы, в которых есть факторы
                    for group in SHAP_GROUPS:
                        gkey = group["key"]
                        items = grouped.get(gkey, [])
                        if not items:
                            continue

                        group_total = sum(v for _, v in items)
                        group_sign = "+" if group_total > 0 else ""

                        shap_html += f"""
                        <div class="shap-group">
                        <details open>
                        <summary class="shap-group-header">
                            <span class="group-emoji">{group['emoji']}</span>
                            <span>{group['label']}</span>
                            <span class="group-count">{T('profile_shap_group_count').format(count=len(items), sign=group_sign, pts=_shap_to_display_points(group_total, max_abs, 20.0))}</span>
                        </summary>
                        <div class="shap-group-body">"""

                        for fname, shap_val in items:
                            pts = _shap_to_display_points(shap_val, max_abs, 20.0)
                            direction = "positive" if shap_val > 0 else "negative"
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
                                    f"Вклад в итоговый балл (SHAP): {shap_val:+.3f} в шкале модели."
                                )
                                raw_hint = raw_for_shap.get(fname)
                            
                            # Ищем цитаты из PDF для этого фактора
                            pdf_citations = _find_pdf_citations_for_shap(pdf_text, fname, pdf_names)

                            # Формируем иконки скрепок для цитат (компактный HTML без переносов)
                            citation_icons_html = ""
                            if pdf_citations:
                                for cit in pdf_citations:
                                    cit_line = cit["line_number"]
                                    cit_doc = cit.get("doc_name", "Неизвестный документ")
                                    cit_quote = cit["quote"].replace('"', '&quot;').replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;')
                                    cit_expl = cit["explanation"].replace('<', '&lt;').replace('>', '&gt;')
                                    
                                    # Ссылка на документ (если есть в списке)
                                    doc_link = ""
                                    if pdf_names and len(pdf_names) == 1:
                                        # Один документ — просто показываем имя
                                        doc_link = f'<div style="margin-top:6px; font-size:11px; color:#0072CE;">📎 {cit_doc}</div>'
                                    elif pdf_names:
                                        # Несколько документов — ищем индекс
                                        doc_idx = None
                                        for idx, pn in enumerate(pdf_names):
                                            if pn in cit_doc or cit_doc in pn:
                                                doc_idx = idx
                                                break
                                        if doc_idx is not None:
                                            doc_link = f'<div style="margin-top:6px; font-size:11px; color:#0072CE;">📎 {cit_doc}</div>'
                                        else:
                                            doc_link = f'<div style="margin-top:6px; font-size:11px; color:#0072CE;">📎 {cit_doc}</div>'
                                    else:
                                        doc_link = f'<div style="margin-top:6px; font-size:11px; color:#0072CE;">📎 {cit_doc}</div>'
                                    
                                    citation_icons_html += (
                                        f'<span class="citation-icon">'
                                        f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v6c0 .55-.45 1-1 1s-1-.45-1-1V5c0-.28-.22-.5-.5-.5s-.5.22-.5.5v12.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V6c0-.55-.45-1-1-1s-1 .45-1 1z"/></svg>'
                                        f'<div class="citation-popup">'
                                        f'<div class="citation-popup-header"><span class="icon">📄</span><span class="title">{cit_doc} (строка {cit_line})</span></div>'
                                        f'<div class="quote">"{cit_quote}"</div>'
                                        f'<div class="citation-line-number">Строка в документе: {cit_line}</div>'
                                        f'{doc_link}'
                                        f'<div class="citation-explanation">{cit_expl}</div>'
                                        f'</div></span>'
                                    )

                            # Если нет цитат — добавляем пояснение
                            no_citation_note = ""
                            if not pdf_citations and pdf_text.strip():
                                no_citation_note = (
                                    '<span style="font-size:10px; color:#bbb; margin-left:6px;" '
                                    'title="В PDF не найдено прямых упоминаний этого показателя. '
                                    'Значение взято из анкеты заявки или рассчитано моделью.">'
                                    '⚠️ нет в PDF</span>'
                                )
                            elif not pdf_text.strip():
                                no_citation_note = (
                                    '<span style="font-size:10px; color:#bbb; margin-left:6px;" '
                                    'title="PDF документы не загружены. Значение взято из анкеты.">'
                                    '📄 нет PDF</span>'
                                )

                            shap_html += (
                                f'<div class="shap-item {cls}">'
                                f'<span class="shap-val">{sign}{pts} б.</span>'
                                f'<div>'
                                f'<div style="font-weight:600;">{label}{citation_icons_html}{no_citation_note}</div>'
                                f'<div style="font-size:12px; color:#888;">{explanation}</div>'
                                f'<div style="font-size:11px; color:#aaa;">исходное значение: {raw_hint}</div>'
                                f'</div></div>'
                            )

                        shap_html += "</div></details></div>"

                if shap_html:
                    st.markdown(shap_html, unsafe_allow_html=True)
                else:
                    st.caption(T("profile_shap_unavailable"))

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

                st.markdown(f"#### {T('profile_financial_summary')}")

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
                        <td style="padding:6px 4px; color:#888;">{T('profile_fin_requested')}</td>
                        <td style="text-align:right; font-weight:700;">{_fmt_tenge(app.get('requested_amount',0))}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:6px 4px; color:#888;">{T('profile_fin_region')}</td>
                        <td style="text-align:right;">{app.get('region','—')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:6px 4px; color:#888;">{T('profile_fin_source')}</td>
                        <td style="text-align:right;">{app.get('source_system','manual').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 4px; color:#888;">{T('profile_fin_date')}</td>
                        <td style="text-align:right;">{app.get('application_date', app.get('calculated_at','')[:10])}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

            compliance = app.get("compliance")
            if compliance:
                st.markdown("---")
                st.markdown(f"#### {T('profile_compliance_title')}")

                c_status = compliance.get("overall_status", "")
                c_score_pct = compliance.get("overall_score_pct", 0)
                c_bonus = compliance.get("compliance_bonus", 0)
                c_checks = compliance.get("checks", [])
                c_disq = compliance.get("disqualifiers_found", [])
                c_name = compliance.get("subsidy_name", "")

                badge_cls = {
                    T("profile_compliance_status_match"): "badge-ok",
                    T("profile_compliance_status_partial"): "badge-warn",
                    T("profile_compliance_status_fail"): "badge-fail",
                    T("profile_compliance_status_disq"): "badge-disq",
                }.get(c_status, "badge-warn")

                bonus_sign = "+" if c_bonus >= 0 else ""
                bonus_color = "#1a7a4a" if c_bonus >= 0 else "#b5001f"

                bonus_label = T("profile_compliance_bonus").format(sign=bonus_sign, bonus=f"{c_bonus:.1f}")
                requirements_label = T("profile_compliance_requirements").format(pct=c_score_pct)

                st.markdown(f"""
                <div class="compliance-block">
                    <div class="compliance-header">
                        <div>
                            <span class="compliance-title">📑 {c_name}</span>
                            <span style="font-size:12px; color:#888; margin-left:10px;">
                                {requirements_label}
                            </span>
                        </div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:13px; font-weight:700; color:{bonus_color};">
                                {bonus_label}
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
                            {T('profile_compliance_disq_title')}
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

                            if status == T("profile_compliance_check_done"):
                                item_cls = "check-ok"
                            elif status in (T("profile_compliance_status_partial"), T("profile_compliance_check_warn")):
                                item_cls = "check-warn"
                            else:
                                item_cls = "check-fail"

                            critical_badge = f'<span class="check-critical">{T("profile_compliance_critical")}</span>' if is_critical else ""

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
                st.markdown(f"""
                <div style="background:#f8f8f8; border:1px dashed #ccc; border-radius:8px;
                     padding:14px 18px; margin-top:16px; font-size:13px; color:#888; text-align:center;">
                    {T('profile_no_docs_compliance')}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="hitl-block">
                <div class="hitl-title">{T('profile_hitl_title')}</div>
                <div class="hitl-desc">
                    {T('profile_hitl_desc')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            hitl_col1, hitl_col2, hitl_col3 = st.columns([3, 1, 1])
            with hitl_col1:
                comment = st.text_area(T("profile_hitl_comment"), height=60,
                                       key=f"comment_{selected_id}")
            with hitl_col2:
                if st.button(T("profile_hitl_approve"), type="primary",
                             key=f"approve_{selected_id}", use_container_width=True):
                    payload = {
                        "application_id": selected_id,
                        "decision": "approved",
                        "comment": comment,
                    }
                    result = _api_post("/api/v1/decision", payload)
                    if result:
                        st.session_state.decisions[selected_id] = "approved"
                        st.success(T("profile_hitl_approved"))
                        _refresh_apps()
                        st.rerun()
            with hitl_col3:
                if st.button(T("profile_hitl_reject"), type="secondary",
                             key=f"reject_{selected_id}", use_container_width=True):
                    payload = {
                        "application_id": selected_id,
                        "decision": "rejected",
                        "comment": comment,
                    }
                    result = _api_post("/api/v1/decision", payload)
                    if result:
                        st.session_state.decisions[selected_id] = "rejected"
                        st.warning(T("profile_hitl_rejected"))
                        _refresh_apps()
                        st.rerun()

            current_decision = st.session_state.decisions.get(selected_id)
            if current_decision:
                d_label = T("all_apps_decision_approved") if current_decision == "approved" else T("all_apps_decision_rejected")
                d_color = "#1a7a4a" if current_decision == "approved" else "#b5001f"
                st.markdown(f"""
                <div style="margin-top:12px; background:#f8f8f8; border:1px solid #ddd;
                     border-radius:8px; padding:10px 16px; font-size:13px; color:{d_color}; font-weight:700;">
                    {T('profile_hitl_decision_fixed').format(label=d_label, datetime=datetime.now().strftime('%d.%m.%Y %H:%M'))}
                </div>
                """, unsafe_allow_html=True)

with tab4:
    st.markdown(f"### {T('api_title')} — {T('api_subtitle')}")

    st.markdown(f"""
    <div class="info-box">
        <h4>{T('api_info_title')}</h4>
        <p style="font-size:14px; color:#444; line-height:1.7;">
            <b>{T('api_info_base')}</b> <code>http://localhost:8003</code><br>
            <b>{T('api_info_auth')}</b> API Key в заголовке <code>X-API-Key</code><br>
            <b>{T('api_info_format')}</b> JSON<br>
            <b>{T('api_info_rate')}</b> 1000 запросов/час
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # POST /api/v1/score
    st.markdown(f"#### {T('api_ep_score')}")
    st.caption(T("api_ep_score_desc"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("api_request_label"))
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
        st.markdown(T("api_response_label"))
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
    st.markdown(f"#### {T('api_ep_score_docs')}")
    st.caption(T("api_ep_score_docs_desc"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("api_request_label"))
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
        st.markdown(T("api_response_label"))
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
    st.markdown(f"#### {T('api_ep_apps')}")
    st.caption(T("api_ep_apps_desc"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("api_request_label"))
        st.code("""curl "http://localhost:8003/api/v1/applications?zone=green&min_score=80" \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" """, language="bash")

    with col2:
        st.markdown(T("api_response_label"))
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
    st.markdown(f"#### {T('api_ep_decision')}")
    st.caption(T("api_ep_decision_desc"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("api_request_label"))
        st.code("""curl -X POST http://localhost:8003/api/v1/decision \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \\
  -d '{
    "application_id": "A7F2B1C0",
    "decision": "approved",
    "comment": "Соответствует критериям, рекомендовано к одобрению"
  }'""", language="bash")

    with col2:
        st.markdown(T("api_response_label"))
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
    st.markdown(f"#### {T('api_ep_sync')}")
    st.caption(T("api_ep_sync_desc"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("api_request_label"))
        st.code("""curl -X POST http://localhost:8003/api/v1/giss/sync \\
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" """, language="bash")

    with col2:
        st.markdown(T("api_response_label"))
        st.code("""{
  "status": "success",
  "synced_count": 15,
  "message": "Тестовые заявки добавлены"
}""", language="json")

    st.divider()

    # Таблица зон скоринга
    st.markdown(f"#### {T('api_zones_title')}")

    zones_data = {
        T("api_zones_zone"): ["🟢 green", "🟡 yellow", "🔴 red"],
        T("api_zones_range"): ["80–100", "50–79", "0–49"],
        T("api_zones_rec"): [
            T("api_zones_green_rec"),
            T("api_zones_yellow_rec"),
            T("api_zones_red_rec")
        ],
        T("api_zones_prob"): [
            T("api_zones_green_prob"),
            T("api_zones_yellow_prob"),
            T("api_zones_red_prob")
        ],
    }
    st.dataframe(pd.DataFrame(zones_data), hide_index=True, use_container_width=True)

st.divider()
st.markdown(f"""
<div style="text-align:center; font-size:12px; color:#9aacce; padding:10px 0;">
    {T('footer')}
</div>
""", unsafe_allow_html=True)
