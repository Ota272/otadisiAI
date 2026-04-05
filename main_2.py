"""
Генератор PDF-документов для субсидий МСХ РК / МСХ РК субсидияларына арналған PDF құжаттар генераторы
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

# ═══════════════════════════════════════════════════════
# ЯЗЫК / ТІЛ
# ═══════════════════════════════════════════════════════
LANGUAGE = "ru"  # "ru" или "kz" — меняется в main()

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

# ═══════════════════════════════════════════════════════
# СПРАВОЧНИКИ / АНЫҚТАМАЛЫҚТАР
# ═══════════════════════════════════════════════════════
REGIONS = {
    "ru": [
        "Акмолинская область", "Алматинская область", "Атырауская область",
        "Восточно-Казахстанская область", "Жамбылская область",
        "Карагандинская область", "Костанайская область",
        "Кызылординская область", "Мангистауская область",
        "Павлодарская область", "Северо-Казахстанская область",
        "Туркестанская область", "Западно-Казахстанская область",
        "Актюбинская область", "область Ұлытау", "область Жетісу", "г.Шымкент",
    ],
    "kz": [
        "Ақмола облысы", "Алматы облысы", "Атырау облысы",
        "Шығыс Қазақстан облысы", "Жамбыл облысы",
        "Қарағанды облысы", "Қостанай облысы",
        "Қызылорда облысы", "Маңғыстау облысы",
        "Павлодар облысы", "Солтүстік Қазақстан облысы",
        "Түркістан облысы", "Батыс Қазақстан облысы",
        "Ақтөбе облысы", "Ұлытау облысы", "Жетісу облысы", "Шымкент қ.",
    ],
}

DIRECTIONS = {
    "ru": [
        "Субсидирование в скотоводстве",
        "Субсидирование в овцеводстве",
        "Субсидирование в коневодстве",
        "Субсидирование в птицеводстве",
        "Субсидирование в верблюдоводстве",
    ],
    "kz": [
        "Мал шаруашылығын субсидиялау",
        "Қой шаруашылығын субсидиялау",
        "Жылқы шаруашылығын субсидиялау",
        "Құс шаруашылығын субсидиялау",
        "Түйе шаруашылығын субсидиялау",
    ],
}

DIRECTION_CODES = {
    "ru": {d: i for i, d in enumerate(DIRECTIONS["ru"])},
    "kz": {d: i for i, d in enumerate(DIRECTIONS["kz"])},
}

SUBSIDY_NAMES = {
    "ru": {
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
    },
    "kz": {
        "Мал шаруашылығын субсидиялау": [
            "Тұқымдық ірі қара мал аналық басын сатып алу",
            "Тұқымдық бұқа-өндірушілерді сатып алу",
            "Сүт өндірісі құнын арзандату",
        ],
        "Қой шаруашылығын субсидиялау": [
            "Тұқымдық қошқар-өндірушілерді сатып алу",
            "Қой шаруашылығы өнімін өткізу шығындарын субсидиялау",
        ],
        "Жылқы шаруашылығын субсидиялау": [
            "Тұқымдық айғыр-өндірушілерді сатып алу",
            "Жылқы шаруашылығы өнімін өткізу шығындарын субсидиялау",
        ],
        "Құс шаруашылығын субсидиялау": [
            "Құс еті өндірісі шығындарын субсидиялау",
            "Жұмыртқа өндірісі шығындарын субсидиялау",
        ],
        "Түйе шаруашылығын субсидиялау": [
            "Тұқымдық түйе-өндірушілерді сатып алу",
            "Түйе шаруашылығы өнімін өткізу шығындарын субсидиялау",
        ],
    },
}

NORMATIVE_MAP = {
    "ru": {
        "Приобретение племенного маточного поголовья КРС": 260_000,
        "Приобретение племенных быков-производителей КРС": 260_000,
        "Удешевление стоимости производства молока": 45,
        "Приобретение племенных баранов-производителей": 100_000,
        "Приобретение племенных жеребцов-производителей": 150_000,
    },
    "kz": {
        "Тұқымдық ірі қара мал аналық басын сатып алу": 260_000,
        "Тұқымдық бұқа-өндірушілерді сатып алу": 260_000,
        "Сүт өндірісі құнын арзандату": 45,
        "Тұқымдық қошқар-өндірушілерді сатып алу": 100_000,
        "Тұқымдық айғыр-өндірушілерді сатып алу": 150_000,
    },
}

COMPANY_PREFIXES = ["Агро", "Плем", "Эталон", "Сарыарка", "Дала", "Береке", "Нур", "Астана"]
COMPANY_SUFFIXES = ["Элита", "Астык", "Мал", "Егiн", "Фарм", "Инвест", "Холдинг"]

# ── Сценарии генерации ──
SCENARIOS = {
    "excellent": {
        "label": {"ru": "ОТЛИЧНО (85-100 баллов)", "kz": "ӨТЕ ЖАҚСЫ (85-100 балл)"},
        "heads_range": (120, 250),
        "price_per_head_range": (850_000, 1_100_000),
        "pasture_per_head_range": (12.0, 20.0),
        "debt_to_ebitda_range": (0.0, 0.8),
        "vet_health_pct_range": (96, 100),
        "payment_status": {
            "ru": "ИСПОЛНЕНО. Оплата произведена в полном объеме в установленный срок.",
            "kz": "ОРЫНДАЛДЫ. Төлем толық көлемде белгіленген мерзімде жүргізілді.",
        },
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
        "label": {"ru": "ХОРОШО (65-84 балла)", "kz": "ЖАҚСЫ (65-84 балл)"},
        "heads_range": (80, 150),
        "price_per_head_range": (700_000, 950_000),
        "pasture_per_head_range": (8.0, 14.0),
        "debt_to_ebitda_range": (0.5, 2.0),
        "vet_health_pct_range": (88, 96),
        "payment_status": {
            "ru": "ИСПОЛНЕНО. Оплата произведена.",
            "kz": "ОРЫНДАЛДЫ. Төлем жүргізілді.",
        },
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
        "label": {"ru": "СРЕДНЕ (45-64 балла)", "kz": "ОРТАША (45-64 балл)"},
        "heads_range": (40, 90),
        "price_per_head_range": (500_000, 750_000),
        "pasture_per_head_range": (5.0, 9.0),
        "debt_to_ebitda_range": (1.5, 3.5),
        "vet_health_pct_range": (75, 89),
        "payment_status": {
            "ru": "ЧАСТИЧНО. Оплата произведена на 60%.",
            "kz": "ІШІНАРА. Төлем 60% көлемінде жүргізілді.",
        },
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
        "label": {"ru": "ПЛОХО (15-44 балла)", "kz": "НАШАР (15-44 балл)"},
        "heads_range": (10, 45),
        "price_per_head_range": (300_000, 550_000),
        "pasture_per_head_range": (2.0, 5.5),
        "debt_to_ebitda_range": (3.0, 5.0),
        "vet_health_pct_range": (50, 76),
        "payment_status": {
            "ru": "НЕ ИСПОЛНЕНО. Оплата не произведена. Задолженность.",
            "kz": "ОРЫНДАЛМАДЫ. Төлем жүргізілмеді. Берешек.",
        },
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
    global LANGUAGE
    lang = LANGUAGE

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sc = SCENARIOS[scenario_key]

    region = random.choice(REGIONS[lang])
    direction = random.choice(DIRECTIONS[lang])
    subsidy_name = random.choice(SUBSIDY_NAMES[lang][direction])
    normative = NORMATIVE_MAP[lang].get(subsidy_name, 150_000)

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
    VET_TEXT = {
        "ru": {
            "excellent": "100%. Инфекционных заболеваний не выявлено. Хозяйство благополучно по всем заболеваниям.",
            "good": "{pct}%. Требуется плановая вакцинация. Карантинных мероприятий нет.",
            "average": [
                "Выявлены единичные случаи мастита. Требуется лечение.",
                "Плановая вакцинация не завершена. Карантин на 2 участках.",
                "{pct}%. Обнаружены случаи респираторных заболеваний."
            ],
            "poor": [
                "{pct}%. Обнаружены случаи бруцеллеза. Хозяйство на карантине.",
                "{pct}%. Массовый падеж. Ветеринарный паспорт не оформлен.",
                "{pct}%. Критическая ситуация. Множественные инфекционные заболевания."
            ],
        },
        "kz": {
            "excellent": "100%. Жұқпалы аурулар анықталмады. Шаруашылық барлық аурулар бойынша благополучный.",
            "good": "{pct}%. Жоспарлы вакцинация қажет. Карантиндік шаралар жоқ.",
            "average": [
                "Маститтің жеке жағдайлары анықталды. Емдеу қажет.",
                "Жоспарлы вакцинация аяқталмады. 2 учаскеде карантин.",
                "{pct}%. Респираторлық аурулар жағдайлары анықталды."
            ],
            "poor": [
                "{pct}%. Бруцеллез жағдайлары анықталды. Шаруашылық карантинде.",
                "{pct}%. Жаппай қырылу. Ветеринарлық паспорт ресімделмеген.",
                "{pct}%. Сыни жағдай. Көптеген жұқпалы аурулар."
            ],
        },
    }
    vt = VET_TEXT[lang]
    if scenario_key == "excellent":
        vet_text = vt["excellent"]
    elif scenario_key == "good":
        vet_text = vt["good"].format(pct=int(vet_pct))
    elif scenario_key == "average":
        vet_text = random.choice(vt["average"]).format(pct=int(vet_pct))
    else:  # poor
        vet_text = random.choice(vt["poor"]).format(pct=int(vet_pct))

    data = {
        # Основные
        "scenario_key": scenario_key,
        "scenario_label": sc["label"][lang],
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
        "payment_status": sc["payment_status"][lang],
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
        "direction_code": DIRECTION_CODES[lang][direction],
        "is_pedigree": 1 if "племен" in subsidy_name.lower() or "тұқымдық" in subsidy_name.lower() else 0,
        "is_producer": 1 if "производит" in subsidy_name.lower() or "өндіруші" in subsidy_name.lower() else 0,
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
    """Предсказание скора через реальную XGBoost модель."""
    global LANGUAGE
    lang = LANGUAGE
    engine = _get_scoring_engine()

    if engine is not None:
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
            "language_code": 1.0 if lang == "kz" else 0.0,
        }
        result = engine.score_farmer(feature_dict, lang=lang, include_shap=True)

        LABEL_MAP = {
            "ru": {
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
            },
            "kz": {
                "gross_output_growth_yoy": "Жалпы өнімнің өсуі",
                "pedigree_ratio": "Тұқымдық мал үлесі",
                "historical_survival_rate": "Мал сақталуы",
                "veterinary_compliance": "Ветеринарлық сәйкестік",
                "subsidy_dependence_index": "Субсидияға тәуелділік",
                "debt_load_ratio": "Борыш жүктемесі (кері)",
                "grazing_norm_deviation": "Жайылым ауытқуы",
                "natural_loss_risk_score": "Аномалды өлім қаупі",
                "land_to_livestock_ratio": "Жайылыммен қамтамасыз ету",
                "years_in_operation": "Жұмыс стажы",
            },
        }
        components = {}
        shap_all = result.get("all_shap_values", {})
        for feat, label in LABEL_MAP[lang].items():
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

    return _predict_score_fallback(data)


def _predict_score_fallback(data: dict) -> dict:
    """Упрощённое предсказание если XGBoost не доступен."""
    global LANGUAGE
    lang = LANGUAGE

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

    VERDICTS = {
        "ru": {
            "green": ("GREEN", "🟢", "Строго рекомендовано"),
            "yellow": ("YELLOW", "🟡", "Требует рассмотрения"),
            "red": ("RED", "🔴", "Не рекомендовано"),
        },
        "kz": {
            "green": ("GREEN", "🟢", "Қатаң түрде ұсынылады"),
            "yellow": ("YELLOW", "🟡", "Қосымша қарау қажет"),
            "red": ("RED", "🔴", "Ұсынылмайды"),
        },
    }
    if score >= 80:
        zone, zone_emoji, verdict = VERDICTS[lang]["green"]
    elif score >= 50:
        zone, zone_emoji, verdict = VERDICTS[lang]["yellow"]
    else:
        zone, zone_emoji, verdict = VERDICTS[lang]["red"]

    COMPONENT_LABELS = {
        "ru": {
            "gross": "Рост валовой продукции",
            "pedigree": "Доля племенного поголовья",
            "survival": "Сохранность поголовья",
            "vet": "Ветеринарное соответствие",
            "subsidy": "Независимость от субсидий",
            "debt": "Долговая нагрузка (инверт.)",
            "grazing": "Отклонение пастбищ от нормы",
            "risk": "Риск аномальной смертности",
            "land": "Обеспеченность пастбищами",
            "years": "Стаж работы",
        },
        "kz": {
            "gross": "Жалпы өнімнің өсуі",
            "pedigree": "Тұқымдық мал үлесі",
            "survival": "Мал сақталуы",
            "vet": "Ветеринарлық сәйкестік",
            "subsidy": "Субсидияға тәуелділік",
            "debt": "Борыш жүктемесі (кері)",
            "grazing": "Жайылым ауытқуы",
            "risk": "Аномалды өлім қаупі",
            "land": "Жайылыммен қамтамасыз ету",
            "years": "Жұмыс стажы",
        },
    }
    lbl = COMPONENT_LABELS[lang]

    return {
        "predicted_score": score,
        "zone": zone,
        "zone_emoji": zone_emoji,
        "verdict": verdict,
        "components": {
            lbl["gross"]: round(gross_norm * 22, 1),
            lbl["pedigree"]: round(pedigree_norm * 18, 1),
            lbl["survival"]: round(survival_norm * 13, 1),
            lbl["vet"]: round(vet_norm * 11, 1),
            lbl["subsidy"]: round(subsidy_indep * 10, 1),
            lbl["debt"]: round(debt_indep * 9, 1),
            lbl["grazing"]: round(grazing_norm * 5, 1),
            lbl["risk"]: round(risk_indep * 4, 1),
            lbl["land"]: round(land_norm * 4, 1),
            lbl["years"]: round(years_norm * 4, 1),
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
    global LANGUAGE
    lang = LANGUAGE
    d = data
    docs = {}

    contract_day = f"{d['contract_date'].day:02d}"
    contract_month = d['contract_date'].strftime('%B').capitalize()
    contract_date_str = f"{d['contract_date'].strftime('%d.%m.%Y')}"
    payment_date_str = d['payment_date'].strftime('%d.%m.%Y')
    esf_date_str = d['esf_date'].strftime('%d.%m.%Y')

    completeness = d["doc_completeness"]

    # ── Тексты для PDF ──
    T = {
        "ru": {
            "contract_title": "ДОГОВОР КУПЛИ-ПРОДАЖИ",
            "city": "г. Кокшетау",
            "seller": "Продавец",
            "buyer": "Покупатель",
            "subject": "ПРЕДМЕТ ДОГОВОРА",
            "subject_text": "Продавец передает поголовье КРС.",
            "subject_text_full": "Продавец передает племенное маточное поголовье КРС.",
            "quantity": "Количество",
            "heads": "голов",
            "gender_age": "Половозрастная группа: телки.",
            "breed": "Порода: казахская белоголовая.",
            "breed_mix": "Порода: смешанная.",
            "age": "Возраст: 8-14 месяцев.",
            "sum_calc": "СУММА И РАСЧЕТЫ",
            "price_head": "Стоимость головы",
            "total_sum": "Общая сумма",
            "debt_load": "Кредитная нагрузка (Долг/EBITDA)",
            "growth": "Рост валовой продукции",
            "obligations": "ОБЯЗАТЕЛЬСТВА СТОРОН",
            "oblig_text": "Покупатель обязуется использовать поголовье по целевому назначению для воспроизводства стада в течение не менее 2 (двух) лет.",
            "oblig_text_short": "Покупатель обязуется использовать поголовье по целевому назначению.",
            "party1": "Сторона 1",
            "party2": "Сторона 2",
            "payment": "Оплата",
            "not_paid": "не произведена. Задолженность.",
            "survival": "Сохранность стада",
            "vet_compl": "Ветеринарное соответствие",
            "pedigree_share": "Доля племенного поголовья",
            "growth_prod": "Рост валовой продукции",
            "work_exp": "Стаж работы",
            "years": "лет",
            "subsidy_dep": "Зависимость от субсидий",
            "prev_subsidies": "Ранее получено субсидий",
            "copy_doc": "Копия документа",
            "params": "Параметры",
            "payment_title": "ПЛАТЕЖНОЕ ПОРУЧЕНИЕ",
            "sender": "Отправитель",
            "receiver": "Получатель",
            "sum_label": "Сумма",
            "purpose": "Назначение",
            "payment_for": "Оплата за КРС",
            "payment_for_contract": "Оплата по дог.",
            "status": "Статус",
            "not_executed": "НЕ ИСПОЛНЕНО. Задолженность.",
            "not_paid_full": "оплата не произведена.",
            "of_total": "от общей суммы",
            "spravka_title": "СПРАВКА-ПОДТВЕРЖДЕНИЕ ИЗ ИНФОРМАЦИОННЫХ СИСТЕМ",
            "spravka_short": "СПРАВКА-ПОДТВЕРЖДЕНИЕ",
            "spravka_min": "СПРАВКА",
            "issued_to": "Выдана",
            "confirmed_heads": "Подтверждено голов",
            "units": "ед.",
            "reg_iszh": "Регистрация в ИСЖ и ИБСПР",
            "registered": "ЗАРЕГИСТРИРОВАНО",
            "vet_passport": "Ветеринарный паспорт",
            "issued_doc": "ОФОРМЛЕН",
            "vet_welfare": "Ветеринарное благополучие",
            "land_cadastre": "Земельный кадастр",
            "available": "имеется",
            "pasture_supply": "Обеспеченность пастбищами",
            "ha_head": "Га/голову",
            "norm_dev": "Отклонение от нормы нагрузки",
            "mortality_risk": "Риск аномальной смертности",
            "farm_status": "Статус хозяйства",
            "active": "действующее",
            "no_info": "информация не предоставлена",
            "esf_title": "ЭЛЕКТРОННАЯ СЧЕТ-ФАКТУРА",
            "esf_short": "СЧЕТ-ФАКТУРА",
            "esf_min": "ДОКУМЕНТ",
            "date_issue": "Дата выписки",
            "date_label": "Дата",
            "supplier": "Поставщик",
            "goods": "Товар",
            "cattle": "КРС маточное поголовье",
            "cattle_short": "КРС",
            "cattle_min": "поголовье",
            "qty": "Количество",
            "qty_short": "Кол-во",
            "price": "Цена",
            "total": "Итого",
            "breed_cert": "Племенное свидетельство",
            "attached": "ПРИЛАГАЕТСЯ",
            "attached_low": "прилагается",
            "survival_label": "Сохранность поголовья",
            "vet_label": "Ветеринарное соответствие",
            "growth_label": "Рост продукции",
            "debt_label": "Долг/EBITDA",
            "subsidy_label": "Зависимость от субсидий",
            "prev_label": "Ранее получено субсидий",
        },
        "kz": {
            "contract_title": "САТЫП АЛУ-САТУ ШАРТЫ",
            "city": "Көкшетау қ.",
            "seller": "Сатушы",
            "buyer": "Сатып алушы",
            "subject": "ШАРТТЫҢ ЗАТЫ",
            "subject_text": "Сатушы ІҚМ мал басын береді.",
            "subject_text_full": "Сатушы тұқымдық ірі қара мал аналық басын береді.",
            "quantity": "Саны",
            "heads": "бас",
            "gender_age": "Жыныс-жас тобы: таналар.",
            "breed": "Тұқымы: қазақ ақбас.",
            "breed_mix": "Тұқымы: аралас.",
            "age": "Жасы: 8-14 ай.",
            "sum_calc": "СОМА ЖӘНЕ ЕСЕПТЕСУЛЕР",
            "price_head": "Бас құны",
            "total_sum": "Жалпы сома",
            "debt_load": "Несие жүктемесі (Борыш/EBITDA)",
            "growth": "Жалпы өнімнің өсуі",
            "obligations": "ТАРАПТАРДЫҢ МІНДЕТТЕМЕЛЕРІ",
            "oblig_text": "Сатып алушы малды мақсатты пайдалануды және 2 (екі) жыл ішінде үйді жаңғыртуды міндеттенеді.",
            "oblig_text_short": "Сатып алушы малды мақсатты пайдалануды міндеттенеді.",
            "party1": "1-тарап",
            "party2": "2-тарап",
            "payment": "Төлем",
            "not_paid": "жүргізілмеді. Берешек.",
            "survival": "Мал сақталуы",
            "vet_compl": "Ветеринарлық сәйкестік",
            "pedigree_share": "Тұқымдық мал үлесі",
            "growth_prod": "Жалпы өнімнің өсуі",
            "work_exp": "Жұмыс стажы",
            "years": "жыл",
            "subsidy_dep": "Субсидияға тәуелділік",
            "prev_subsidies": "Бұрын алынған субсидиялар",
            "copy_doc": "Құжат көшірмесі",
            "params": "Параметрлер",
            "payment_title": "ТӨЛЕМ ТАПСЫРМАСЫ",
            "sender": "Жіберуші",
            "receiver": "Алушы",
            "sum_label": "Сома",
            "purpose": "Тағайындау",
            "payment_for": "ІҚМ үшін төлем",
            "payment_for_contract": "Шарт бойынша төлем",
            "status": "Мәртебе",
            "not_executed": "ОРЫНДАЛМАДЫ. Берешек.",
            "not_paid_full": "төлем жүргізілмеді.",
            "of_total": "жалпы сомадан",
            "spravka_title": "АҚПАРАТТЫҚ ЖҮЙЕЛЕРДЕН АНЫҚТАМА-РАСТАУ",
            "spravka_short": "АНЫҚТАМА-РАСТАУ",
            "spravka_min": "АНЫҚТАМА",
            "issued_to": "Берілді",
            "confirmed_heads": "Расталған бас саны",
            "units": "бас",
            "reg_iszh": "ИСЖ және ИБСПР тіркеу",
            "registered": "ТІРКЕЛГЕН",
            "vet_passport": "Ветеринарлық паспорт",
            "issued_doc": "РӘСІМДЕЛГЕН",
            "vet_welfare": "Ветеринарлық благополучие",
            "land_cadastre": "Жер кадастры",
            "available": "бар",
            "pasture_supply": "Жайылыммен қамтамасыз ету",
            "ha_head": "Га/бас",
            "norm_dev": "Жүктеме нормасынан ауытқу",
            "mortality_risk": "Аномалды өлім қаупі",
            "farm_status": "Шаруашылық мәртебесі",
            "active": "жұмыс істейтін",
            "no_info": "ақпарат берілмеген",
            "esf_title": "ЭЛЕКТРОНДЫ ШОТ-ФАКТУРА",
            "esf_short": "ШОТ-ФАКТУРА",
            "esf_min": "ҚҰЖАТ",
            "date_issue": "Жазылған күні",
            "date_label": "Күні",
            "supplier": "Жеткізуші",
            "goods": "Тауар",
            "cattle": "ІҚМ аналық басы",
            "cattle_short": "ІҚМ",
            "cattle_min": "мал басы",
            "qty": "Саны",
            "qty_short": "Саны",
            "price": "Бағасы",
            "total": "Барлығы",
            "breed_cert": "Тұқымдық куәлік",
            "attached": "ҚОСА БЕРІЛДІ",
            "attached_low": "қоса берілді",
            "survival_label": "Мал сақталуы",
            "vet_label": "Ветеринарлық сәйкестік",
            "growth_label": "Өнімнің өсуі",
            "debt_label": "Борыш/EBITDA",
            "subsidy_label": "Субсидияға тәуелділік",
            "prev_label": "Бұрын алынған субсидиялар",
        },
    }
    t = T[lang]

    # ═══════════════════════════════════════════════════
    # 1. ДОГОВОР / ШАРТ
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{t['contract_title']} № {d['contract_num']}
{t['city']}, «{contract_day}» {contract_month} 2026 {t['years'][:2]}.

{t['seller']}: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
{t['buyer']}: {d['company']} (БИН {d['bin_buyer']})

1. {t['subject']}
1.1. {t['subject_text_full']}
1.2. {t['quantity']}: {int(d['heads'])} {t['heads']}. {t['gender_age']}
1.3. {t['breed']}
1.4. {t['age']}

2. {t['sum_calc']}
2.1. {t['price_head']}: {int(d['price_per_head']):,} тенге.
2.2. {t['total_sum']}: {int(d['total_sum']):,} тенге.
2.3. {t['debt_load']}: {d['debt_ebitda']}
2.4. {t['growth']}: {d['gross_output_growth_yoy']*100:+.1f}%

3. {t['obligations']}
3.1. {t['oblig_text']}
3.2. {t['buyer']} {t['survival'].lower()} қамтамасыз етуді міндеттенеді.
"""
    elif completeness == "mostly_full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{t['contract_title']} № {d['contract_num']}
{t['city']}, «{contract_day}» {contract_month} 2026 {t['years'][:2]}.

{t['seller']}: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
{t['buyer']}: {d['company']} (БИН {d['bin_buyer']})

1. {t['subject']}
1.1. {t['subject_text']}
1.2. {t['quantity']}: {int(d['heads'])} {t['heads']}.
1.3. {t['breed_mix']}

2. {t['sum_calc']}
2.1. {t['price_head']}: {int(d['price_per_head']):,} тенге.
2.2. {t['total_sum']}: {int(d['total_sum']):,} тенге.
2.3. {t['debt_load']}: {d['debt_ebitda']}

3. {t['obligations']}
3.1. {t['oblig_text_short']}
"""
    elif completeness == "partial":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{t['contract_title']} № {d['contract_num']}
{t['city']}, «{contract_day}» {contract_month} 2026 {t['years'][:2]}.

{t['seller']}: ТОО «Племзавод-Элита»
{t['buyer']}: {d['company']}

1. {t['subject']}
1.1. {t['subject_text']}
1.2. {t['quantity']}: {int(d['heads'])} {t['heads']}.

2. {t['sum_calc']}
2.1. {t['total_sum']}: {int(d['total_sum']):,} тенге.
2.2. {t['debt_load']}: {d['debt_ebitda']}
"""
    else:  # minimal
        survival_pct = int(d['historical_survival_rate'] * 100)
        vet_pct_int = int(d['veterinary_compliance'] * 100)
        pedigree_pct = int(d['pedigree_ratio'] * 100)
        growth_pct = round(d['gross_output_growth_yoy'] * 100, 1)
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{t['contract_title'].split()[0]} № {d['contract_num']}
{t['city']}, {contract_date_str}

{t['party1']}: ТОО «Племзавод-Элита»
{t['party2']}: {d['company']}

1. {t['party1']} {t['cattle_min']} береді.
2. {t['quantity']}: {int(d['heads'])} {t['heads']}.
3. {t['total_sum']}: {int(d['total_sum']):,} тенге.
4. {t['payment']}: {t['not_paid']}
5. {t['debt_load']} = {d['debt_ebitda']}
6. {t['survival']}: {survival_pct}%
7. {t['vet_compl']}: {vet_pct_int}%
8. {t['pedigree_share']}: {pedigree_pct}%
9. {t['growth_prod']}: {growth_pct}%
10. {t['work_exp']}: {int(d['years_in_operation'])} {t['years']}
11. {t['subsidy_dep']}: {round(d['subsidy_dependence_index']*100)}%
12. {t['prev_subsidies']}: {int(d['previous_subsidies_count'])}
"""

    # ═══════════════════════════════════════════════════
    # 2. КОПИЯ ДОГОВОРА
    # ═══════════════════════════════════════════════════
    docs["2_Dogovor_Kuplyu_Prodazhy_Copy.pdf"] = (
        f"{t['copy_doc']} № {d['contract_num']} {contract_date_str}.\n"
        f"{t['params']}: {int(d['heads'])} {t['heads']} {t['total_sum'].lower()} {int(d['total_sum']):,} KZT."
    )

    # ═══════════════════════════════════════════════════
    # 3. ПЛАТЁЖНОЕ ПОРУЧЕНИЕ
    # ═══════════════════════════════════════════════════
    if completeness in ("full", "mostly_full", "partial"):
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
{t['payment_title']} № {d['payment_num']} {payment_date_str}
{t['sender']}: {d['company']}
{t['receiver']}: ТОО «Племзавод-Элита»
{t['sum_label']}: {int(d['total_sum'] * d['payment_pct'] / 100):,} KZT ({d['payment_pct']}% {t['of_total']})
{t['purpose']}: {t['payment_for']} ({int(d['heads'])} {t['heads']}) {t['payment_for_contract']} № {d['contract_num']}.
{t['status']}: {d['payment_status']}
БИК: KZKAKZKX
ИИК: KZ123456789012345678
"""
    else:
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
{t['payment_title']} № {d['payment_num']} {payment_date_str}
{t['sender']}: {d['company']}
{t['receiver']}: ТОО «Племзавод-Элита»
{t['sum_label']}: 0 KZT — {t['not_paid_full']}
{t['purpose']}: {t['payment_for_contract']} № {d['contract_num']}.
{t['status']}: {t['not_executed']}
{t['debt_label']} = {d['debt_ebitda']}
"""

    # ═══════════════════════════════════════════════════
    # 4. СПРАВКА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["4_Spravka_ISZH.pdf"] = f"""
{t['spravka_title']}
{t['issued_to']}: {d['company']}
{t['confirmed_heads']}: {int(d['heads'])} {t['units']}.
{t['reg_iszh']}: {t['registered']}
{t['vet_passport']}: {t['issued_doc']}
{t['vet_welfare']}: {d['vet_text']}
{t['land_cadastre']}: {d['pasture']} {t['ha_head']}
{t['pasture_supply']}: {d['pasture']} {t['ha_head']}.
{t['norm_dev']}: {d['grazing_norm_deviation']:+.2f}
{t['mortality_risk']}: {d['natural_loss_risk_score']:.2f}
"""
    elif completeness == "mostly_full":
        docs["4_Spravka_ISZH.pdf"] = f"""
{t['spravka_short']}
{t['issued_to']}: {d['company']}
{t['confirmed_heads']}: {int(d['heads'])} {t['units']}.
{t['reg_iszh']}: {t['registered']}
{t['vet_passport']}: {t['issued_doc']}
{t['vet_welfare']}: {d['vet_text']}
{t['land_cadastre']}: {t['available']}.
"""
    elif completeness == "partial":
        docs["4_Spravka_ISZH.pdf"] = f"""
{t['spravka_min']}
{t['issued_to']}: {d['company']}
{t['confirmed_heads']}: {int(d['heads'])} {t['units']}.
{t['reg_iszh']}: {t['registered']}
{t['farm_status']}: {t['active']}.
"""
    else:
        docs["4_Spravka_ISZH.pdf"] = f"""
{t['spravka_min']}
{t['issued_to']}: {d['company']}
{t['quantity']}: {int(d['heads'])} {t['units']}.
{t['survival_label']}: {int(d['historical_survival_rate']*100)}%
{t['vet_label']}: {int(d['veterinary_compliance']*100)}%
{t['status']}: {t['no_info']}.
"""

    # ═══════════════════════════════════════════════════
    # 5. СЧЕТ-ФАКТУРА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["5_ESF.pdf"] = f"""
{t['esf_title']} № {d['esf_num']}
{t['date_issue']}: {esf_date_str}
{t['supplier']}: ТОО «Племзавод-Элита»
{t['buyer']}: {d['company']}
{t['goods']}: {t['cattle']}.
{t['qty']}: {int(d['heads'])}
{t['price']}: {int(d['price_per_head']):,}
{t['total']}: {int(d['total_sum']):,}
{t['breed_cert']}: {t['attached']}
{t['pedigree_share']}: {d['pedigree_ratio']*100:.1f}%
{t['survival_label']}: {d['historical_survival_rate']*100:.1f}%
{t['vet_label']}: {d['veterinary_compliance']*100:.1f}%
"""
    elif completeness == "mostly_full":
        docs["5_ESF.pdf"] = f"""
{t['esf_short']} № {d['esf_num']}
{t['date_label']}: {esf_date_str}
{t['supplier']}: ТОО «Племзавод-Элита»
{t['buyer']}: {d['company']}
{t['goods']}: {t['cattle_short']}.
{t['qty']}: {int(d['heads'])}
{t['price']}: {int(d['price_per_head']):,}
{t['total']}: {int(d['total_sum']):,}
{t['breed_cert']}: {t['attached_low']}.
"""
    elif completeness == "partial":
        docs["5_ESF.pdf"] = f"""
{t['esf_min']} № {d['esf_num']}
{t['date_label']}: {esf_date_str}
{t['supplier']}: ТОО «Племзавод-Элита»
{t['buyer']}: {d['company']}
{t['goods']}: {t['cattle_short']}.
{t['qty']}: {int(d['heads'])}
{t['sum_label']}: {int(d['total_sum']):,}
"""
    else:
        docs["5_ESF.pdf"] = f"""
{t['esf_min']} № {d['esf_num']}
{t['date_label']}: {esf_date_str}
{t['seller']}: ТОО «Племзавод-Элита»
{t['buyer']}: {d['company']}
{t['goods']}: {t['cattle_min']}.
{t['qty_short']}: {int(d['heads'])}
{t['sum_label']}: {int(d['total_sum']):,}
{t['pedigree_share']}: {int(d['pedigree_ratio']*100)}%
{t['survival_label']}: {int(d['historical_survival_rate']*100)}%
{t['vet_label']}: {int(d['veterinary_compliance']*100)}%
{t['growth_label']}: {round(d['gross_output_growth_yoy']*100, 1)}%
{t['work_exp']}: {int(d['years_in_operation'])} {t['years']}
{t['debt_label']} = {d['debt_ebitda']}
{t['subsidy_label']}: {round(d['subsidy_dependence_index']*100)}%
{t['prev_label']}: {int(d['previous_subsidies_count'])}
"""

    # Генерация файлов
    gen_label = f"📄 Құжаттарды генерациялау: {d['scenario_label']}..." if lang == "kz" else f"📄 Генерация документов для сценария: {d['scenario_label']}..."
    print(f"\n{gen_label}")
    created_files = []
    for filename, text in docs.items():
        create_pdf(filename, text)
        created_files.append(filename)
        print(f"   ✅ {filename}")

    return created_files


def print_mandatory_info(data: dict):
    """Выводит обязательную информацию заявки."""
    global LANGUAGE
    lang = LANGUAGE
    d = data

    TITLE = "ДАННЫЕ ЗАЯВКИ" if lang == "ru" else "ӨТІНІМ ДЕРЕКТЕРІ"
    BIN_LABEL = "БИН / ИИН предприятия:" if lang == "ru" else "Кәсіпорын БСН / ЖСН:"
    REGION_LABEL = "Область:" if lang == "ru" else "Облыс:"
    COMPANY_LABEL = "Наименование предприятия:" if lang == "ru" else "Кәсіпорын атауы:"
    DIRECTION_LABEL = "Направление субсидии:" if lang == "ru" else "Субсидия бағыты:"
    TYPE_LABEL = "Вид субсидии:" if lang == "ru" else "Субсидия түрі:"
    SUM_LABEL = "Запрашиваемая сумма (тенге):" if lang == "ru" else "Сұралған сома (теңге):"

    print("\n" + "=" * 55)
    print(f"  {TITLE}")
    print("=" * 55)
    print(f"  {BIN_LABEL:35s} {d['bin_buyer']}")
    print(f"  {REGION_LABEL:35s} {d['region']}")
    print(f"  {COMPANY_LABEL:35s} {d['company']}")
    print(f"  {DIRECTION_LABEL:35s} {d['direction']}")
    print(f"  {TYPE_LABEL:35s} {d['subsidy_name']}")
    print(f"  {SUM_LABEL:35s} {int(d['total_sum']):,}")
    print("=" * 55)


def print_score_prediction(prediction: dict):
    """Выводит предсказание скора."""
    global LANGUAGE
    lang = LANGUAGE
    p = prediction

    TITLE = "ПРЕДСКАЗАНИЕ СКОРА" if lang == "ru" else "БОЛЖАМ БАЛЛ"
    VERDICT_LABEL = "Вердикт:" if lang == "ru" else "Үкім:"
    COMPONENTS_LABEL = "Разбивка по компонентам (максимум = вес компонента):" if lang == "ru" else "Компоненттер бойынша (максимум = салмақ):"

    print("\n" + "=" * 60)
    print(f"  {TITLE}: {p['predicted_score']:.1f}  {p['zone_emoji']} {p['zone']}")
    print(f"  {VERDICT_LABEL} {p['verdict']}")
    print("=" * 60)
    print(f"  {COMPONENTS_LABEL}")
    for comp, val in p["components"].items():
        bar_len = max(0, min(20, int(val / 100 * 20)))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {comp:35s} {bar} {val:5.1f}")
    print("=" * 60)


def main():
    global LANGUAGE

    print("=" * 65)
    print("  SmartAgro Score | PDF құжаттар генераторы")
    print("=" * 65)

    # Выбор языка / Тілді таңдау
    print("\n🌐 Выберите язык / Тілді таңдаңыз:")
    print("  1. Русский (ru)")
    print("  2. Қазақша (kz)")
    lang_choice = input("\nТіл (1-2) [ru]: ").strip()
    if lang_choice == "2":
        LANGUAGE = "kz"
        print("✅ Таңдалған тіл: Қазақша")
    else:
        LANGUAGE = "ru"
        print("✅ Выбранный язык: Русский")

    lang = LANGUAGE

    scenarios_list = ["excellent", "good", "average", "poor"]

    print(f"\n{'Қолжетімді сценарийлер:' if lang == 'kz' else 'Доступные сценарии:'}")
    for i, key in enumerate(scenarios_list, 1):
        print(f"  {i}. {SCENARIOS[key]['label'][lang]}")

    all_label = "БАРЛЫҚ сценарийлер" if lang == "kz" else "ВСЕ сценарии"
    poor_label = "ҚОЛАЙСЫЗ (average + poor)" if lang == "kz" else "НЕБЛАГОПРИЯТНЫЕ (average + poor)"
    choice_label = "Сценарийді таңдаңыз" if lang == "kz" else "Выберите сценарий"

    print(f"\n  5. {all_label}")
    print(f"  6. {poor_label}")

    choice = input(f"\n{choice_label} (1-6): ").strip()

    if choice == "5":
        selected = scenarios_list
    elif choice == "6":
        selected = ["average", "poor"]
    elif choice in "1234":
        selected = [scenarios_list[int(choice) - 1]]
    else:
        default_msg = "Қате. Барлық сценарийлер." if lang == "kz" else "Неверный выбор. По умолчанию: все сценарии."
        print(f"❌ {default_msg}")
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
    summary_title = "ҚОРЫТЫНДЫ" if lang == "kz" else "ИТОГОВАЯ СВОДКА"
    done_msg = "Генерация аяқталды!" if lang == "kz" else "Генерация завершена!"
    print("\n\n" + "=" * 65)
    print(f"  {summary_title}")
    print("=" * 65)
    for res in all_results:
        p = res["prediction"]
        d = res["data"]
        print(f"\n  {p['zone_emoji']} {d['scenario_label']}")
        print(f"     {'Скор:':<8} {p['predicted_score']:.1f} | {d['company']}")
        print(f"     {'Бағыт:':<8} {d['direction']}" if lang == "kz" else f"     {'Направление:':<14} {d['direction']}")
        heads_label = "Бас саны" if lang == "kz" else "Голов"
        sum_label = "Сома" if lang == "kz" else "Сумма"
        print(f"     {heads_label}: {int(d['heads'])} | {sum_label}: {int(d['total_sum']):,} тенге")
        files_label = "Файлдар" if lang == "kz" else "Файлы"
        print(f"     {files_label}: {', '.join(res['files'])}")
    print("\n" + "=" * 65)
    print(f"  ✨ {done_msg}")
    print("=" * 65)


if __name__ == "__main__":
    main()
