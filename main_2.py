"""
Генератор PDF-документов для субсидий МСХ РК
Генерирует документы от "отлично" до "плохо" с реалистичными данными,
которые приводят к соответствующему скору.
"""
import os
import sys
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# ── Регистрация шрифта ──
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    FONT_PATHS = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    FONT_NAME = None
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            pdfmetrics.registerFont(TTFont("DejaVu", fp))
            FONT_NAME = "DejaVu"
            break
    if FONT_NAME is None:
        print("⚠️ DejaVuSans.ttf не найден. PDF будет содержать 'кракозябры'.")
        FONT_NAME = "Helvetica"
except ImportError:
    print("❌ reportlab не установлен: pip install reportlab")
    sys.exit(1)

# ── Справочники ──
REGIONS = [
    "Акмолинская область", "Алматинская область", "Атырауская область",
    "Восточно-Казахстанская область", "Жамбылская область",
    "Карагандинская область", "Костанайская область",
    "Кызылординская область", "Мангистауская область",
    "Павлодарская область", "Северо-Казахстанская область",
    "Туркестанская область", "Западно-Казахстанская область",
    "Актюбинская область", "область Ұлытау", "область Жетісу", "г.Шымкент",
]

DIRECTIONS = [
    "Субсидирование в скотоводстве",
    "Субсидирование в овцеводстве",
    "Субсидирование в коневодстве",
    "Субсидирование в птицеводстве",
    "Субсидирование в верблюдоводстве",
]

DIRECTION_CODES = {
    "Субсидирование в скотоводстве": 0,
    "Субсидирование в овцеводстве": 1,
    "Субсидирование в коневодстве": 2,
    "Субсидирование в птицеводстве": 3,
    "Субсидирование в верблюдоводстве": 4,
}

SUBSIDY_NAMES = {
    "Субсидирование в скотоводстве": [
        "Приобретение племенного маточного поголовья КРС",
        "Приобретение племенных быков-производителей КРС",
        "Удешевление стоимости производства молока",
    ],
    "Субсидирование в овцеводстве": [
        "Приобретение племенных баранов-производителей",
        "Субсидирование затрат по реализации продукции овцеводства",
    ],
    "Субсидирование в коневодстве": [
        "Приобретение племенных жеребцов-производителей",
        "Субсидирование затрат по реализации продукции коневодства",
    ],
    "Субсидирование в птицеводстве": [
        "Субсидирование затрат по производству мяса птицы",
        "Субсидирование затрат по производству яиц",
    ],
    "Субсидирование в верблюдоводстве": [
        "Приобретение племенных верблюдов-производителей",
        "Субсидирование затрат по реализации продукции верблюдоводства",
    ],
}

NORMATIVE_MAP = {
    "Приобретение племенного маточного поголовья КРС": 260_000,
    "Приобретение племенных быков-производителей КРС": 260_000,
    "Удешевление стоимости производства молока": 45,
    "Приобретение племенных баранов-производителей": 100_000,
    "Приобретение племенных жеребцов-производителей": 150_000,
}

COMPANY_PREFIXES = ["Агро", "Плем", "Эталон", "Сарыарка", "Дала", "Береке", "Нур", "Астана"]
COMPANY_SUFFIXES = ["Элита", "Астык", "Мал", "Егiн", "Фарм", "Инвест", "Холдинг"]

# ── Сценарии генерации ──
SCENARIOS = {
    "excellent": {
        "label": "ОТЛИЧНО (85-100 баллов)",
        "heads_range": (120, 250),
        "price_per_head_range": (850_000, 1_100_000),
        "pasture_per_head_range": (12.0, 20.0),
        "debt_to_ebitda_range": (0.0, 0.8),
        "vet_health_pct_range": (96, 100),
        "payment_status": "ИСПОЛНЕНО. Оплата произведена в полном объеме в установленный срок.",
        "payment_pct": 100,
        "gross_growth_range": (0.15, 0.45),
        "survival_rate_range": (0.92, 0.98),
        "vet_compliance_range": (0.92, 0.99),
        "subsidy_dependence_range": (0.05, 0.20),
        "pedigree_ratio_range": (0.70, 0.95),
        "years_in_op_range": (10, 22),
        "prev_subsidies_range": (5, 12),
        "grazing_dev_range": (-0.3, 0.3),
        "mortality_risk_range": (0.5, 0.9),
        "doc_completeness": "full",
        "has_vet_certificate": True,
        "has_breeding_cert": True,
        "has_land_cadastre": True,
        "has_iszh_registration": True,
        "has_obligation_clause": True,
        "has_bank_details": True,
        "has_bin_iin": True,
    },
    "good": {
        "label": "ХОРОШО (65-84 балла)",
        "heads_range": (80, 150),
        "price_per_head_range": (700_000, 950_000),
        "pasture_per_head_range": (8.0, 14.0),
        "debt_to_ebitda_range": (0.5, 2.0),
        "vet_health_pct_range": (88, 96),
        "payment_status": "ИСПОЛНЕНО. Оплата произведена.",
        "payment_pct": 100,
        "gross_growth_range": (0.05, 0.20),
        "survival_rate_range": (0.85, 0.93),
        "vet_compliance_range": (0.80, 0.93),
        "subsidy_dependence_range": (0.15, 0.35),
        "pedigree_ratio_range": (0.45, 0.75),
        "years_in_op_range": (5, 15),
        "prev_subsidies_range": (3, 8),
        "grazing_dev_range": (-0.5, 0.5),
        "mortality_risk_range": (0.7, 1.2),
        "doc_completeness": "mostly_full",
        "has_vet_certificate": True,
        "has_breeding_cert": True,
        "has_land_cadastre": True,
        "has_iszh_registration": True,
        "has_obligation_clause": True,
        "has_bank_details": True,
        "has_bin_iin": True,
    },
    "average": {
        "label": "СРЕДНЕ (45-64 балла)",
        "heads_range": (40, 90),
        "price_per_head_range": (500_000, 750_000),
        "pasture_per_head_range": (5.0, 9.0),
        "debt_to_ebitda_range": (1.5, 3.5),
        "vet_health_pct_range": (75, 89),
        "payment_status": "ЧАСТИЧНО. Оплата произведена на 60%.",
        "payment_pct": 60,
        "gross_growth_range": (-0.05, 0.10),
        "survival_rate_range": (0.75, 0.86),
        "vet_compliance_range": (0.60, 0.82),
        "subsidy_dependence_range": (0.30, 0.55),
        "pedigree_ratio_range": (0.20, 0.50),
        "years_in_op_range": (2, 8),
        "prev_subsidies_range": (1, 5),
        "grazing_dev_range": (-0.8, 1.0),
        "mortality_risk_range": (1.0, 1.8),
        "doc_completeness": "partial",
        "has_vet_certificate": True,
        "has_breeding_cert": False,
        "has_land_cadastre": True,
        "has_iszh_registration": True,
        "has_obligation_clause": False,
        "has_bank_details": True,
        "has_bin_iin": True,
    },
    "poor": {
        "label": "ПЛОХО (15-44 балла)",
        "heads_range": (10, 45),
        "price_per_head_range": (300_000, 550_000),
        "pasture_per_head_range": (2.0, 5.5),
        "debt_to_ebitda_range": (3.0, 5.0),
        "vet_health_pct_range": (50, 76),
        "payment_status": "НЕ ИСПОЛНЕНО. Оплата не произведена. Задолженность.",
        "payment_pct": 0,
        "gross_growth_range": (-0.25, 0.0),
        "survival_rate_range": (0.55, 0.76),
        "vet_compliance_range": (0.30, 0.62),
        "subsidy_dependence_range": (0.50, 0.85),
        "pedigree_ratio_range": (0.05, 0.25),
        "years_in_op_range": (1, 4),
        "prev_subsidies_range": (0, 2),
        "grazing_dev_range": (0.5, 2.0),
        "mortality_risk_range": (1.5, 3.0),
        "doc_completeness": "minimal",
        "has_vet_certificate": False,
        "has_breeding_cert": False,
        "has_land_cadastre": False,
        "has_iszh_registration": False,
        "has_obligation_clause": False,
        "has_bank_details": True,
        "has_bin_iin": True,
    },
}


def _rand(rng):
    """Случайное число из диапазона (int или float)."""
    a, b = rng
    if isinstance(a, int) and isinstance(b, int):
        return random.randint(a, b)
    return random.uniform(a, b)


def generate_scenario_data(scenario_key: str, seed: int | None = None):
    """Генерирует полный набор данных для сценария."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sc = SCENARIOS[scenario_key]

    region = random.choice(REGIONS)
    direction = random.choice(DIRECTIONS)
    subsidy_name = random.choice(SUBSIDY_NAMES[direction])
    normative = NORMATIVE_MAP.get(subsidy_name, 150_000)

    heads = _rand(sc["heads_range"])
    price_per_head = _rand(sc["price_per_head_range"])
    total_sum = heads * price_per_head

    company = f"ТОО «{random.choice(COMPANY_PREFIXES)}-{random.choice(COMPANY_SUFFIXES)}»"
    bin_seller = f"{random.randint(100000000000, 999999999999)}"
    bin_buyer = f"{random.randint(100000000000, 999999999999)}"

    contract_num = f"{random.randint(1, 99)}/26"
    contract_date = datetime(2026, random.randint(1, 4), random.randint(1, 28))

    payment_num = f"{random.randint(10, 999)}"
    payment_date = contract_date + timedelta(days=random.randint(1, 5))

    esf_num = f"ЭСФ-2026-{contract_date.month:02d}-{random.randint(1000, 9999)}"
    esf_date = contract_date + timedelta(days=random.randint(1, 3))

    pasture = round(_rand(sc["pasture_per_head_range"]), 1)
    debt_ebitda = round(_rand(sc["debt_to_ebitda_range"]), 2)
    vet_pct = _rand(sc["vet_health_pct_range"])
    gross_growth = round(_rand(sc["gross_growth_range"]), 3)
    survival = round(_rand(sc["survival_rate_range"]), 3)
    vet_compliance = round(_rand(sc["vet_compliance_range"]), 3)
    subsidy_dep = round(_rand(sc["subsidy_dependence_range"]), 3)
    pedigree_ratio = round(_rand(sc["pedigree_ratio_range"]), 3)
    years_op = _rand(sc["years_in_op_range"])
    prev_subs = _rand(sc["prev_subsidies_range"])
    grazing_dev = round(_rand(sc["grazing_dev_range"]), 3)
    mortality_risk = round(_rand(sc["mortality_risk_range"]), 3)

    # Ветеринарный статус — текстовое описание
    if scenario_key == "excellent":
        vet_text = "100%. Инфекционных заболеваний не выявлено. Хозяйство благополучно по всем заболеваниям."
    elif scenario_key == "good":
        vet_text = f"{vet_pct:.0f}%. Требуется плановая вакцинация. Карантинных мероприятий нет."
    elif scenario_key == "average":
        issues = random.choice([
            "Выявлены единичные случаи мастита. Требуется лечение.",
            "Плановая вакцинация не завершена. Карантин на 2 участках.",
            f"{vet_pct:.0f}%. Обнаружены случаи респираторных заболеваний."
        ])
        vet_text = issues
    else:  # poor
        issues = random.choice([
            f"{vet_pct:.0f}%. Обнаружены случаи бруцеллеза. Хозяйство на карантине.",
            f"{vet_pct:.0f}%. Массовый падеж. Ветеринарный паспорт не оформлен.",
            f"{vet_pct:.0f}%. Критическая ситуация. Множественные инфекционные заболевания."
        ])
        vet_text = issues

    data = {
        # Основные
        "scenario_key": scenario_key,
        "scenario_label": sc["label"],
        "region": region,
        "direction": direction,
        "subsidy_name": subsidy_name,
        "normative": normative,
        "company": company,
        "bin_seller": bin_seller,
        "bin_buyer": bin_buyer,
        "contract_num": contract_num,
        "contract_date": contract_date,
        "payment_num": payment_num,
        "payment_date": payment_date,
        "esf_num": esf_num,
        "esf_date": esf_date,
        "heads": heads,
        "price_per_head": price_per_head,
        "total_sum": total_sum,
        "pasture": pasture,
        "debt_ebitda": debt_ebitda,
        "vet_pct": vet_pct,
        "vet_text": vet_text,
        "payment_status": sc["payment_status"],
        "payment_pct": sc["payment_pct"],
        # ML-фичи
        "gross_output_growth_yoy": gross_growth,
        "historical_survival_rate": survival,
        "veterinary_compliance": vet_compliance,
        "subsidy_dependence_index": subsidy_dep,
        "pedigree_ratio": pedigree_ratio,
        "years_in_operation": years_op,
        "previous_subsidies_count": prev_subs,
        "debt_load_ratio": debt_ebitda,
        "land_to_livestock_ratio": pasture,
        "grazing_norm_deviation": grazing_dev,
        "natural_loss_risk_score": mortality_risk,
        "livestock_count": heads,
        "log_amount": float(np.log1p(total_sum)),
        "direction_code": DIRECTION_CODES[direction],
        "is_pedigree": 1 if "племен" in subsidy_name.lower() else 0,
        "is_producer": 1 if "производит" in subsidy_name.lower() else 0,
        # Документы
        "has_vet_certificate": sc["has_vet_certificate"],
        "has_breeding_cert": sc["has_breeding_cert"],
        "has_land_cadastre": sc["has_land_cadastre"],
        "has_iszh_registration": sc["has_iszh_registration"],
        "has_obligation_clause": sc["has_obligation_clause"],
        "has_bank_details": sc["has_bank_details"],
        "has_bin_iin": sc["has_bin_iin"],
        "doc_completeness": sc["doc_completeness"],
    }
    return data


# ── Глобальный кэш модели (загружается один раз) ──
_engine_cache = {"engine": None}


def _get_scoring_engine():
    """Ленивая загрузка ScoringEngine — модель грузится один раз."""
    if _engine_cache["engine"] is not None:
        return _engine_cache["engine"]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ml.shap_integration import ScoringEngine
        engine = ScoringEngine(Path("models"))
        _engine_cache["engine"] = engine
        print("✅ XGBoost модель загружена для предсказаний")
        return engine
    except Exception as e:
        print(f"⚠️ Не удалось загрузить XGBoost модель ({e}), использую упрощённую формулу")
        return None


def predict_score(data: dict) -> dict:
    """
    Предсказание скора через реальную XGBoost модель.
    Fallback: упрощённая формула если модель не загружена.
    """
    engine = _get_scoring_engine()

    if engine is not None:
        # Реальный скоринг через XGBoost
        feature_dict = {
            "gross_output_growth_yoy": data["gross_output_growth_yoy"],
            "land_to_livestock_ratio": data["land_to_livestock_ratio"],
            "historical_survival_rate": data["historical_survival_rate"],
            "subsidy_dependence_index": data["subsidy_dependence_index"],
            "veterinary_compliance": data["veterinary_compliance"],
            "years_in_operation": data["years_in_operation"],
            "pedigree_ratio": data["pedigree_ratio"],
            "previous_subsidies_count": data["previous_subsidies_count"],
            "debt_load_ratio": data["debt_load_ratio"],
            "grazing_norm_deviation": data["grazing_norm_deviation"],
            "natural_loss_risk_score": data["natural_loss_risk_score"],
            "log_amount": data["log_amount"],
            "livestock_count": data["livestock_count"],
            "direction_code": data["direction_code"],
            "is_pedigree": data["is_pedigree"],
            "is_producer": data["is_producer"],
            "hour_submitted": 12.0,
            "month_submitted": 4.0,
            "region_encoded": 0.0,
        }
        result = engine.score_farmer(feature_dict, include_shap=True)

        # Разбивка по компонентам из SHAP
        components = {}
        shap_all = result.get("all_shap_values", {})
        label_map = {
            "gross_output_growth_yoy": "Рост валовой продукции",
            "pedigree_ratio": "Доля племенного поголовья",
            "historical_survival_rate": "Сохранность поголовья",
            "veterinary_compliance": "Ветеринарное соответствие",
            "subsidy_dependence_index": "Независимость от субсидий",
            "debt_load_ratio": "Долговая нагрузка (инверт.)",
            "grazing_norm_deviation": "Отклонение пастбищ от нормы",
            "natural_loss_risk_score": "Риск аномальной смертности",
            "land_to_livestock_ratio": "Обеспеченность пастбищами",
            "years_in_operation": "Стаж работы",
        }
        for feat, label in label_map.items():
            val = shap_all.get(feat, 0.0)
            components[label] = round(val, 1)

        return {
            "predicted_score": result["score"],
            "zone": result["zone"],
            "zone_emoji": "🟢" if result["zone"] == "green" else "🟡" if result["zone"] == "yellow" else "🔴",
            "verdict": result["recommendation"],
            "components": components,
            "method": "XGBoost",
        }

    # Fallback: упрощённая формула
    return _predict_score_fallback(data)


def _predict_score_fallback(data: dict) -> dict:
    """Упрощённое предсказание если XGBoost не доступен."""
    def norm_val(val, low, high):
        rng = high - low
        if rng <= 0:
            return 0.5
        return max(0.0, min(1.0, (val - low) / rng))

    gross_norm = norm_val(data["gross_output_growth_yoy"], -0.25, 0.45)
    pedigree_norm = norm_val(data["pedigree_ratio"], 0.05, 0.95)
    survival_norm = norm_val(data["historical_survival_rate"], 0.55, 0.98)
    vet_norm = norm_val(data["veterinary_compliance"], 0.30, 0.99)
    subsidy_indep = 1.0 - norm_val(data["subsidy_dependence_index"], 0.05, 0.85)
    debt_indep = 1.0 - norm_val(data["debt_load_ratio"], 0.0, 5.0)
    grazing_norm = norm_val(data["grazing_norm_deviation"] + 2.0, 0.0, 4.0)
    risk_indep = 1.0 - norm_val(data["natural_loss_risk_score"], 0.5, 3.0)
    land_norm = norm_val(data["land_to_livestock_ratio"], 2.0, 20.0)
    years_norm = norm_val(data["years_in_operation"], 1, 22)

    raw_score = (
        gross_norm    * 22.0 +
        pedigree_norm * 18.0 +
        survival_norm * 13.0 +
        vet_norm      * 11.0 +
        subsidy_indep * 10.0 +
        debt_indep    *  9.0 +
        grazing_norm  *  5.0 +
        risk_indep    *  4.0 +
        land_norm     *  4.0 +
        years_norm    *  4.0
    )

    score = max(1.0, min(100.0, raw_score + random.gauss(0, 2)))
    score = round(score, 1)

    if score >= 80:
        zone, zone_emoji, verdict = "GREEN", "🟢", "Строго рекомендовано"
    elif score >= 50:
        zone, zone_emoji, verdict = "YELLOW", "🟡", "Требует рассмотрения"
    else:
        zone, zone_emoji, verdict = "RED", "🔴", "Не рекомендовано"

    return {
        "predicted_score": score,
        "zone": zone,
        "zone_emoji": zone_emoji,
        "verdict": verdict,
        "components": {
            "Рост валовой продукции": round(gross_norm * 22, 1),
            "Доля племенного поголовья": round(pedigree_norm * 18, 1),
            "Сохранность поголовья": round(survival_norm * 13, 1),
            "Ветеринарное соответствие": round(vet_norm * 11, 1),
            "Независимость от субсидий": round(subsidy_indep * 10, 1),
            "Долговая нагрузка (инверт.)": round(debt_indep * 9, 1),
            "Отклонение пастбищ от нормы": round(grazing_norm * 5, 1),
            "Риск аномальной смертности": round(risk_indep * 4, 1),
            "Обеспеченность пастбищами": round(land_norm * 4, 1),
            "Стаж работы": round(years_norm * 4, 1),
        },
        "method": "fallback",
    }


def create_pdf(filename: str, content: str):
    """Создаёт PDF с заданным текстом."""
    c = canvas.Canvas(filename)
    c.setFont(FONT_NAME, 10)
    y = 800
    for line in content.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 15
        if y < 50:
            c.showPage()
            c.setFont(FONT_NAME, 10)
            y = 800
    c.save()


def generate_documents_for_scenario(data: dict) -> list[str]:
    """Генерирует 5 PDF-документов для сценария."""
    d = data
    docs = {}

    contract_day = f"{d['contract_date'].day:02d}"
    contract_month = d['contract_date'].strftime('%B').capitalize()
    contract_date_str = f"{d['contract_date'].strftime('%d.%m.%Y')}"
    payment_date_str = d['payment_date'].strftime('%d.%m.%Y')
    esf_date_str = d['esf_date'].strftime('%d.%m.%Y')

    completeness = d["doc_completeness"]

    # ═══════════════════════════════════════════════════
    # 1. ДОГОВОР КУПЛИ-ПРОДАЖИ
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
ДОГОВОР КУПЛИ-ПРОДАЖИ № {d['contract_num']}
г. Кокшетау, «{contract_day}» {contract_month} 2026 г.

Продавец: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
Покупатель: {d['company']} (БИН {d['bin_buyer']})

1. ПРЕДМЕТ ДОГОВОРА
1.1. Продавец передает племенное маточное поголовье КРС.
1.2. Количество: {int(d['heads'])} голов. Половозрастная группа: телки.
1.3. Порода: казахская белоголовая.
1.4. Возраст: 8-14 месяцев.

2. СУММА И РАСЧЕТЫ
2.1. Стоимость головы: {int(d['price_per_head']):,} тенге.
2.2. Общая сумма: {int(d['total_sum']):,} тенге.
2.3. Кредитная нагрузка (Долг/EBITDA): {d['debt_ebitda']}
2.4. Рост валовой продукции: {d['gross_output_growth_yoy']*100:+.1f}%

3. ОБЯЗАТЕЛЬСТВА СТОРОН
3.1. Покупатель обязуется использовать поголовье по целевому назначению
    для воспроизводства стада в течение не менее 2 (двух) лет.
3.2. Покупатель обязуется обеспечить сохранность поголовья.
"""
    elif completeness == "mostly_full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
ДОГОВОР КУПЛИ-ПРОДАЖИ № {d['contract_num']}
г. Кокшетау, «{contract_day}» {contract_month} 2026 г.

Продавец: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
Покупатель: {d['company']} (БИН {d['bin_buyer']})

1. ПРЕДМЕТ ДОГОВОРА
1.1. Продавец передает поголовье КРС.
1.2. Количество: {int(d['heads'])} голов.
1.3. Порода: смешанная.

2. СУММА И РАСЧЕТЫ
2.1. Стоимость головы: {int(d['price_per_head']):,} тенге.
2.2. Общая сумма: {int(d['total_sum']):,} тенге.
2.3. Кредитная нагрузка (Долг/EBITDA): {d['debt_ebitda']}

3. ОБЯЗАТЕЛЬСТВА СТОРОН
3.1. Покупатель обязуется использовать поголовье по целевому назначению.
"""
    elif completeness == "partial":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
ДОГОВОР КУПЛИ-ПРОДАЖИ № {d['contract_num']}
г. Кокшетау, «{contract_day}» {contract_month} 2026 г.

Продавец: ТОО «Племзавод-Элита»
Покупатель: {d['company']}

1. ПРЕДМЕТ ДОГОВОРА
1.1. Продавец передает поголовье КРС.
1.2. Количество: {int(d['heads'])} голов.

2. СУММА
2.1. Общая сумма: {int(d['total_sum']):,} тенге.
2.2. Кредитная нагрузка (Долг/EBITDA): {d['debt_ebitda']}
"""
    else:  # minimal — ПЛОХОЙ сценарий: ВПИСЫВАЕМ плохие числа
        # Чтобы LLM/regex извлекли ПЛОХИЕ значения вместо оптимистичных дефолтов
        survival_pct = int(d['historical_survival_rate'] * 100)
        vet_pct_int = int(d['veterinary_compliance'] * 100)
        pedigree_pct = int(d['pedigree_ratio'] * 100)
        growth_pct = round(d['gross_output_growth_yoy'] * 100, 1)
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
ДОГОВОР № {d['contract_num']}
г. Кокшетау, {contract_date_str}

Сторона 1: ТОО «Племзавод-Элита»
Сторона 2: {d['company']}

1. Сторона 1 передает поголовье.
2. Количество: {int(d['heads'])} голов.
3. Стоимость: {int(d['total_sum']):,} тенге.
4. Оплата: не произведена. Задолженность.
5. Долг/EBITDA = {d['debt_ebitda']}
6. Сохранность стада: {survival_pct}%
7. Ветеринарное соответствие: {vet_pct_int}%
8. Доля племенного поголовья: {pedigree_pct}%
9. Рост валовой продукции: {growth_pct}%
10. Стаж работы: {int(d['years_in_operation'])} лет
11. Зависимость от субсидий: {round(d['subsidy_dependence_index']*100)}%
12. Ранее получено субсидий: {int(d['previous_subsidies_count'])}
"""

    # ═══════════════════════════════════════════════════
    # 2. КОПИЯ ДОГОВОРА
    # ═══════════════════════════════════════════════════
    docs["2_Dogovor_Kuplyu_Prodazhy_Copy.pdf"] = (
        f"Копия документа № {d['contract_num']} от {contract_date_str}.\n"
        f"Параметры: {int(d['heads'])} голов на сумму {int(d['total_sum']):,} KZT."
    )

    # ═══════════════════════════════════════════════════
    # 3. ПЛАТЁЖНОЕ ПОРУЧЕНИЕ
    # ═══════════════════════════════════════════════════
    if completeness in ("full", "mostly_full", "partial"):
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
ПЛАТЕЖНОЕ ПОРУЧЕНИЕ № {d['payment_num']} от {payment_date_str}
Отправитель: {d['company']}
Получатель: ТОО «Племзавод-Элита»
Сумма: {int(d['total_sum'] * d['payment_pct'] / 100):,} KZT ({d['payment_pct']}% от общей суммы)
Назначение: Оплата за КРС ({int(d['heads'])} голов) по дог. № {d['contract_num']}.
Статус: {d['payment_status']}
БИК: KZKAKZKX
ИИК: KZ123456789012345678
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа для LLM/regex
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
ПЛАТЕЖНОЕ ПОРУЧЕНИЕ № {d['payment_num']} от {payment_date_str}
Отправитель: {d['company']}
Получатель: ТОО «Племзавод-Элита»
Сумма: 0 KZT — оплата не произведена.
Назначение: Оплата по дог. № {d['contract_num']}.
Статус: НЕ ИСПОЛНЕНО. Задолженность.
Долг/EBITDA = {d['debt_ebitda']}
"""

    # ═══════════════════════════════════════════════════
    # 4. СПРАВКА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["4_Spravka_ISZH.pdf"] = f"""
СПРАВКА-ПОДТВЕРЖДЕНИЕ ИЗ ИНФОРМАЦИОННЫХ СИСТЕМ
Выдана: {d['company']}
Подтверждено голов: {int(d['heads'])} ед.
Регистрация в ИСЖ и ИБСПР: ЗАРЕГИСТРИРОВАНО
Ветеринарный паспорт: ОФОРМЛЕН
Ветеринарное благополучие: {d['vet_text']}
Земельный кадастр: {d['pasture']} Га/голову
Обеспеченность пастбищами: {d['pasture']} Га/голову.
Отклонение от нормы нагрузки: {d['grazing_norm_deviation']:+.2f}
Риск аномальной смертности: {d['natural_loss_risk_score']:.2f}
"""
    elif completeness == "mostly_full":
        docs["4_Spravka_ISZH.pdf"] = f"""
СПРАВКА-ПОДТВЕРЖДЕНИЕ
Выдана: {d['company']}
Подтверждено голов: {int(d['heads'])} ед.
Регистрация в ИСЖ и ИБСПР: ЗАРЕГИСТРИРОВАНО
Ветеринарный паспорт: ОФОРМЛЕН
Ветеринарное благополучие: {d['vet_text']}
Земельный кадастр: имеется.
"""
    elif completeness == "partial":
        # Без "ветеринар", "благополучи", "кадастр", "пастбищ"
        docs["4_Spravka_ISZH.pdf"] = f"""
СПРАВКА
Выдана: {d['company']}
Подтверждено голов: {int(d['heads'])} ед.
Регистрация в ИСЖ и ИБСПР: ЗАРЕГИСТРИРОВАНО
Статус хозяйства: действующее.
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа
        docs["4_Spravka_ISZH.pdf"] = f"""
СПРАВКА
Выдана: {d['company']}
Количество голов: {int(d['heads'])} ед.
Сохранность поголовья: {int(d['historical_survival_rate']*100)}%
Ветеринарное соответствие: {int(d['veterinary_compliance']*100)}%
Статус: информация не предоставлена.
"""

    # ═══════════════════════════════════════════════════
    # 5. СЧЕТ-ФАКТУРА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["5_ESF.pdf"] = f"""
ЭЛЕКТРОННАЯ СЧЕТ-ФАКТУРА № {d['esf_num']}
Дата выписки: {esf_date_str}
Поставщик: ТОО «Племзавод-Элита»
Покупатель: {d['company']}
Товар: КРС маточное поголовье.
Количество: {int(d['heads'])}
Цена: {int(d['price_per_head']):,}
Итого: {int(d['total_sum']):,}
Племенное свидетельство: ПРИЛАГАЕТСЯ
Доля племенного поголовья: {d['pedigree_ratio']*100:.1f}%
Сохранность поголовья: {d['historical_survival_rate']*100:.1f}%
Ветеринарное соответствие: {d['veterinary_compliance']*100:.1f}%
"""
    elif completeness == "mostly_full":
        docs["5_ESF.pdf"] = f"""
СЧЕТ-ФАКТУРА № {d['esf_num']}
Дата: {esf_date_str}
Поставщик: ТОО «Племзавод-Элита»
Покупатель: {d['company']}
Товар: КРС.
Количество: {int(d['heads'])}
Цена: {int(d['price_per_head']):,}
Итого: {int(d['total_sum']):,}
Племенное свидетельство: прилагается.
"""
    elif completeness == "partial":
        # Без "акт", "ЭСФ", "счет-фактура", "порода", "племенное свидетельство"
        docs["5_ESF.pdf"] = f"""
ДОКУМЕНТ № {d['esf_num']}
Дата: {esf_date_str}
Поставщик: ТОО «Племзавод-Элита»
Покупатель: {d['company']}
Товар: КРС.
Количество: {int(d['heads'])}
Сумма: {int(d['total_sum']):,}
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа
        docs["5_ESF.pdf"] = f"""
ДОКУМЕНТ № {d['esf_num']}
Дата: {esf_date_str}
Продавец: ТОО «Племзавод-Элита»
Покупатель: {d['company']}
Товар: поголовье.
Кол-во: {int(d['heads'])}
Сумма: {int(d['total_sum']):,}
Доля племенного поголовья: {int(d['pedigree_ratio']*100)}%
Сохранность: {int(d['historical_survival_rate']*100)}%
Ветеринарное соответствие: {int(d['veterinary_compliance']*100)}%
Рост продукции: {round(d['gross_output_growth_yoy']*100, 1)}%
Стаж работы: {int(d['years_in_operation'])} лет
Долг/EBITDA = {d['debt_ebitda']}
Зависимость от субсидий: {round(d['subsidy_dependence_index']*100)}%
Ранее получено субсидий: {int(d['previous_subsidies_count'])}
"""

    # Генерация файлов
    print(f"\n📄 Генерация документов для сценария: {d['scenario_label']}...")
    created_files = []
    for filename, text in docs.items():
        create_pdf(filename, text)
        created_files.append(filename)
        print(f"   ✅ {filename}")

    return created_files


def print_mandatory_info(data: dict):
    """Выводит обязательную информацию заявки (принтом, не в PDF)."""
    d = data
    print("\n" + "=" * 55)
    print("  ДАННЫЕ ЗАЯВКИ")
    print("=" * 55)
    print(f"  БИН / ИИН предприятия:        {d['bin_buyer']}")
    print(f"  Область:                      {d['region']}")
    print(f"  Наименование предприятия:     {d['company']}")
    print(f"  Направление субсидии:         {d['direction']}")
    print(f"  Вид субсидии:                 {d['subsidy_name']}")
    print(f"  Запрашиваемая сумма (тенге):  {int(d['total_sum']):,}")
    print("=" * 55)


def print_score_prediction(prediction: dict):
    """Выводит предсказание скора."""
    p = prediction
    print("\n" + "=" * 60)
    print(f"  ПРЕДСКАЗАНИЕ СКОРА: {p['predicted_score']:.1f}  {p['zone_emoji']} {p['zone']}")
    print(f"  Вердикт: {p['verdict']}")
    print("=" * 60)
    print("  Разбивка по компонентам (максимум = вес компонента):")
    for comp, val in p["components"].items():
        bar_len = int(val / 100 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {comp:35s} {bar} {val:5.1f}")
    print("=" * 60)


def main():
    print("=" * 65)
    print("  SmartAgro Score | Генератор PDF-документов")
    print("=" * 65)

    scenarios_list = ["excellent", "good", "average", "poor"]

    print("\nДоступные сценарии:")
    for i, key in enumerate(scenarios_list, 1):
        print(f"  {i}. {SCENARIOS[key]['label']}")

    print(f"\n  5. Сгенерировать ВСЕ сценарии")
    print(f"  6. Сгенерировать НЕБЛАГОПРИЯТНЫЕ (average + poor)")

    choice = input("\nВыберите сценарий (1-6): ").strip()

    if choice == "5":
        selected = scenarios_list
    elif choice == "6":
        selected = ["average", "poor"]
    elif choice in "1234":
        selected = [scenarios_list[int(choice) - 1]]
    else:
        print("❌ Неверный выбор. По умолчанию: все сценарии.")
        selected = scenarios_list

    all_results = []

    for scenario_key in selected:
        print("\n" + "─" * 65)
        data = generate_scenario_data(scenario_key)
        prediction = predict_score(data)

        print_mandatory_info(data)
        print_score_prediction(prediction)

        files = generate_documents_for_scenario(data)

        all_results.append({
            "scenario": scenario_key,
            "data": data,
            "prediction": prediction,
            "files": files,
        })

    # Итоговая сводка
    print("\n\n" + "=" * 65)
    print("  ИТОГОВАЯ СВОДКА")
    print("=" * 65)
    for res in all_results:
        p = res["prediction"]
        d = res["data"]
        print(f"\n  {p['zone_emoji']} {d['scenario_label']}")
        print(f"     Скор: {p['predicted_score']:.1f} | {d['company']}")
        print(f"     Направление: {d['direction']}")
        print(f"     Голов: {int(d['heads'])} | Сумма: {int(d['total_sum']):,} тенге")
        print(f"     Файлы: {', '.join(res['files'])}")
    print("\n" + "=" * 65)
    print("  ✨ Генерация завершена!")
    print("=" * 65)


if __name__ == "__main__":
    main()
