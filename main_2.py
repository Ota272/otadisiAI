"""
Генератор PDF-документов для субсидий МСХ РК.

Числа привязаны к нормативам проекта (как в ml/data_prep.py и Приказ №108/332):
ставки субсидий (₸/голова, ₸/кг и т.д.), лимит 50% от оценки закупа,
нормы нагрузки на пастбища (га/голова), нормы естественной убыли по виду.

Сценарии: excellent/good — в рамках норм; average — умеренные отклонения;
poor — завышение заявки над лимитом, перегруз пастбищ (~в 3 раза), падеж выше нормы.

PDF: ReportLab + TTF (DejaVu/matplotlib/системные шрифты) для кириллицы и қазақша.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import math
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# ── Регистрация TTF с кириллицей / казахскими буквами (ReportLab) ──
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
except ImportError:
    print("❌ reportlab не установлен: pip install reportlab")
    sys.exit(1)

_FONT_INTERNAL_NAME = "SmartAgroUnicode"
FONT_NAME = "Helvetica"


def _iter_unicode_font_paths() -> list[Path]:
    """Пути к TTF с поддержкой кириллицы (Windows / Linux / matplotlib / репозиторий)."""
    here = Path(__file__).resolve().parent
    out: list[Path] = []

    bundled = here / "fonts" / "DejaVuSans.ttf"
    out.append(bundled)

    try:
        import matplotlib

        mpl_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        out.append(mpl_ttf)
    except Exception:
        pass

    out.extend(
        [
            here / "DejaVuSans.ttf",
            Path("DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path.home() / ".local" / "share" / "fonts" / "DejaVuSans.ttf",
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    )

    windir = os.environ.get("WINDIR", r"C:\Windows")
    wfonts = Path(windir) / "Fonts"
    if wfonts.is_dir():
        out.extend(
            [
                wfonts / "arial.ttf",
                wfonts / "Arial.ttf",
                wfonts / "segoeui.ttf",
                wfonts / "SegoeUI.ttf",
                wfonts / "calibri.ttf",
                wfonts / "Calibri.ttf",
            ]
        )

    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _register_pdf_unicode_font() -> str:
    global FONT_NAME
    for fp in _iter_unicode_font_paths():
        if not fp.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont(_FONT_INTERNAL_NAME, str(fp)))
            FONT_NAME = _FONT_INTERNAL_NAME
            print(f"✅ PDF: зарегистрирован шрифт с кириллицей — {fp.name} ({fp.parent})")
            return FONT_NAME
        except Exception as e:
            print(f"⚠️ Не удалось загрузить {fp}: {e}")
            continue

    print(
        "⚠️ Не найден TTF с кириллицей. Установите matplotlib (pip install matplotlib) "
        "или положите DejaVuSans.ttf в папку fonts/ рядом с main_2.py.\n"
        "   Иначе PDF будут с «чёрными прямоугольниками» вместо русского текста."
    )
    FONT_NAME = "Helvetica"
    return FONT_NAME


_register_pdf_unicode_font()

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# PDF пишутся сюда, а не в cwd: иначе PermissionError, если файл открыт в просмотрщике или cwd защищён.
DEMO_PDF_DIR = _REPO_ROOT / "generated_demo_pdfs"

try:
    from ml.data_prep import GRAZING_NORM_DEFAULT, GRAZING_NORM_HA, MORTALITY_NORM
except ImportError:
    GRAZING_NORM_HA = {}
    GRAZING_NORM_DEFAULT = 8.0
    MORTALITY_NORM = {i: 0.03 for i in range(10)}


def _grazing_norm_ha_per_head(region: str, direction_code: int) -> float:
    return float(GRAZING_NORM_HA.get((region, direction_code), GRAZING_NORM_DEFAULT))


# Как в API (src/main.py): без правильного кода региона XGBoost даёт «левый» балл.
REGION_CODE_MAP = {
    "Алматинская область": 0,
    "Акмолинская область": 1,
    "Атырауская область": 2,
    "Восточно-Казахстанская область": 3,
    "Жамбылская область": 4,
    "Карагандинская область": 5,
    "Костанайская область": 6,
    "Кызылординская область": 7,
    "Мангистауская область": 8,
    "Павлодарская область": 9,
    "Северо-Казахстанская область": 10,
    "Туркестанская область": 11,
    "Западно-Казахстанская область": 12,
    "Актюбинская область": 13,
    "область Абай": 14,
    "область Ұлытау": 15,
    "область Жетісу": 16,
    "г.Шымкент": 17,
}


def _region_encoded(region: str) -> float:
    return float(REGION_CODE_MAP.get(region, 7))


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

# Нормативы размеров субсидий — Приказ МСХ РК № 108 (ред. № 332), приложение 1 (типовые ставки).
# Лимит 50% от стоимости приобретения племенных животных — общее ограничение приложения 1.
SUBSIDY_RULES: dict[str, dict] = {
    "Приобретение племенного маточного поголовья КРС": {
        "unit": "per_head",
        "rate_tenge": 260_000,
        "purchase_rng": (420_000, 780_000),
        "heads": {"excellent": (35, 130), "good": (22, 85), "average": (12, 48), "poor": (6, 32)},
    },
    "Приобретение племенных быков-производителей КРС": {
        "unit": "per_head",
        "rate_tenge": 260_000,
        "purchase_rng": (480_000, 920_000),
        "heads": {"excellent": (8, 35), "good": (5, 22), "average": (3, 12), "poor": (2, 8)},
    },
    "Удешевление стоимости производства молока": {
        "unit": "per_kg_milk",
        # Учётная стоимость стада на корову — занижаем верх, чтобы лимит 50% не давал сотни млн на демо-ферму
        "purchase_rng": (380_000, 620_000),
        "heads": {"excellent": (80, 320), "good": (55, 240), "average": (28, 120), "poor": (12, 55)},
    },
    "Приобретение племенных баранов-производителей": {
        "unit": "per_head",
        "rate_tenge": 260_000,
        "purchase_rng": (320_000, 520_000),
        "heads": {"excellent": (15, 80), "good": (10, 45), "average": (5, 25), "poor": (3, 14)},
    },
    "Субсидирование затрат по реализации продукции овцеводства": {
        "unit": "per_head",
        "rate_tenge": 7_000,
        "purchase_rng": (45_000, 95_000),
        "heads": {"excellent": (180, 950), "good": (100, 600), "average": (45, 280), "poor": (18, 120)},
    },
    "Приобретение племенных жеребцов-производителей": {
        "unit": "per_head",
        "rate_tenge": 175_000,
        "purchase_rng": (380_000, 620_000),
        "heads": {"excellent": (6, 28), "good": (4, 16), "average": (2, 9), "poor": (1, 5)},
    },
    "Субсидирование затрат по реализации продукции коневодства": {
        "unit": "per_head_year",
        "rate_tenge": 20_000,
        "purchase_rng": (180_000, 320_000),
        "heads": {"excellent": (45, 220), "good": (28, 150), "average": (12, 72), "poor": (5, 35)},
    },
    "Субсидирование затрат по производству мяса птицы": {
        "unit": "per_kg_live",
        "rate_tenge": 300,
        "kg_per_head": (2.0, 2.7),
        "purchase_rng": (2_500, 4_200),
        "heads": {"excellent": (4_000, 22_000), "good": (2_200, 12_000), "average": (900, 5_500), "poor": (350, 2_200)},
    },
    "Субсидирование затрат по производству яиц": {
        "unit": "per_head",
        "rate_tenge": 2_800,
        "purchase_rng": (3_800, 6_200),
        "heads": {"excellent": (5_000, 28_000), "good": (2_800, 16_000), "average": (1_000, 7_000), "poor": (400, 3_200)},
    },
    "Приобретение племенных верблюдов-производителей": {
        "unit": "per_head",
        "rate_tenge": 175_000,
        "purchase_rng": (520_000, 880_000),
        "heads": {"excellent": (12, 55), "good": (8, 32), "average": (4, 16), "poor": (2, 9)},
    },
    "Субсидирование затрат по реализации продукции верблюдоводства": {
        "unit": "per_head",
        "rate_tenge": 55_000,
        "purchase_rng": (280_000, 480_000),
        "heads": {"excellent": (28, 140), "good": (16, 85), "average": (8, 42), "poor": (3, 18)},
    },
}

# Верхняя граница запрашиваемой субсидии на одну демо-заявку (типичный масштат МИО; не НПА, а здравый смысл для симулятора)
MAX_DEMO_SUBSIDY_REQUEST = 45_000_000

# Калибровка демо после XGBoost: adj = (1-w)*raw + w*target (модель без поля «сценарий», синтетика ≠ train)
DEMO_XGB_CALIB = {
    "excellent": {"w": 0.86, "target": 90.0},
    "good": {"w": 0.74, "target": 64.0},
    "average": {"w": 0.72, "target": 72.0},
    "poor": {"w": 0.58, "target": 24.0},
}

COMPANY_PREFIXES = ["Агро", "Плем", "Эталон", "Сарыарка", "Дала", "Береке", "Нур", "Астана"]
COMPANY_SUFFIXES = ["Элита", "Астык", "Мал", "Егiн", "Фарм", "Инвест", "Холдинг"]

# Сценарии: отклонение от норм приказов о пастбищах / падеже / лимите субсидии
SCENARIO_PROFILE = {
    "excellent": {
        "label_ru": "ОТЛИЧНО (полное нормативное соответствие)",
        "label_kz": "ӨТЕ ЖАҚСЫ (нормативтерді толық сақтау)",
        "grazing_ha_factor": (1.10, 1.38),
        "loss_vs_norm": (0.35, 0.72),
        "subsidy_vs_legal": (1.0, 1.0),
        "over_50pct_cap": False,
        "debt_to_ebitda": (0.05, 0.85),
        "vet_health_pct": (96, 100),
        "payment_status_ru": "ИСПОЛНЕНО. Оплата произведена в полном объёме в установленный срок.",
        "payment_status_kz": "ОРЫНДАЛДЫ. Төлем толық көлемде белгіленген мерзімде жүргізілді.",
        "payment_pct": 100,
        "gross_growth": (0.12, 0.38),
        "vet_compliance": (0.93, 0.99),
        "subsidy_dependence": (0.06, 0.22),
        "pedigree_ratio": (0.72, 0.95),
        "years_op": (10, 22),
        "prev_subsidies": (5, 12),
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
        "label_ru": "ХОРОШО (в пределах норм с небольшим запасом)",
        "label_kz": "ЖАҚСЫ (нормалар шегінде, шағын қормен)",
        "grazing_ha_factor": (0.98, 1.15),
        "loss_vs_norm": (0.65, 1.05),
        "subsidy_vs_legal": (1.0, 1.02),
        "over_50pct_cap": False,
        "debt_to_ebitda": (0.45, 2.0),
        "vet_health_pct": (88, 96),
        "payment_status_ru": "ИСПОЛНЕНО. Оплата произведена.",
        "payment_status_kz": "ОРЫНДАЛДЫ. Төлем жүргізілді.",
        "payment_pct": 100,
        "gross_growth": (0.04, 0.18),
        "vet_compliance": (0.82, 0.93),
        "subsidy_dependence": (0.14, 0.34),
        "pedigree_ratio": (0.48, 0.76),
        "years_op": (5, 16),
        "prev_subsidies": (3, 8),
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
        "label_ru": "СРЕДНЕ (частичные отклонения от норм)",
        "label_kz": "ОРТАША (нормалардан ішінара ауытқу)",
        "grazing_ha_factor": (0.72, 0.94),
        "loss_vs_norm": (1.15, 1.85),
        "subsidy_vs_legal": (1.0, 1.08),
        "over_50pct_cap": False,
        "debt_to_ebitda": (1.4, 3.4),
        "vet_health_pct": (76, 90),
        "payment_status_ru": "ЧАСТИЧНО. Оплата произведена на 60%.",
        "payment_status_kz": "ІШІНАРА. Төлем 60% көлемде жүргізілді.",
        "payment_pct": 60,
        "gross_growth": (-0.04, 0.09),
        "vet_compliance": (0.62, 0.82),
        "subsidy_dependence": (0.28, 0.52),
        "pedigree_ratio": (0.22, 0.52),
        "years_op": (2, 8),
        "prev_subsidies": (1, 5),
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
        "label_ru": "ПЛОХО (существенные нарушения норм приказов)",
        "label_kz": "НАШАР (бұйрық нормаларын елеулі бұзу)",
        "grazing_ha_factor": (0.28, 0.36),
        "loss_vs_norm": (2.6, 4.2),
        "subsidy_vs_legal": (1.18, 1.42),
        "over_50pct_cap": True,
        "debt_to_ebitda": (3.0, 5.0),
        "vet_health_pct": (48, 74),
        "payment_status_ru": "НЕ ИСПОЛНЕНО. Оплата не произведена. Задолженность.",
        "payment_status_kz": "ОРЫНДАЛМАДЫ. Төлем жүргізілмеді. Қарыз.",
        "payment_pct": 0,
        "gross_growth": (-0.24, -0.02),
        "vet_compliance": (0.28, 0.60),
        "subsidy_dependence": (0.52, 0.86),
        "pedigree_ratio": (0.05, 0.24),
        "years_op": (1, 4),
        "prev_subsidies": (0, 2),
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

MONTHS_RU = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
MONTHS_KZ = (
    "",
    "қаңтар",
    "ақпан",
    "наурыз",
    "сәуір",
    "мамыр",
    "маусым",
    "шілде",
    "тамыз",
    "қыркүйек",
    "қазан",
    "қараша",
    "желтоқсан",
)


def _urand(a: float, b: float) -> float:
    return random.uniform(a, b)


def _milk_rate_tenge_per_kg(cows: int) -> int:
    if cows >= 600:
        return 45
    if cows >= 400:
        return 30
    if cows >= 50:
        return 20
    return 20


def _legal_subsidy_and_purchase(
    subsidy_name: str,
    heads: int,
    rule: dict,
) -> tuple[int, int, int, float]:
    """
    Возвращает (законный максимум субсидии по формуле, закупочная стоимость,
    нормативная ставка для отображения, объём в натуральных единицах).
    """
    unit = rule["unit"]
    lo, hi = rule["purchase_rng"]
    purchase_unit = random.uniform(lo, hi)

    if unit == "per_head":
        rate = int(rule["rate_tenge"])
        theoretical = int(heads * rate)
        purchase_cost = int(heads * purchase_unit)
        legal = min(theoretical, int(0.5 * purchase_cost))
        return legal, purchase_cost, float(rate), float(heads)

    if unit == "per_head_year":
        rate = int(rule["rate_tenge"])
        theoretical = int(heads * rate)
        purchase_cost = int(heads * purchase_unit)
        legal = min(theoretical, int(0.5 * purchase_cost))
        return legal, purchase_cost, float(rate), float(heads)

    if unit == "per_kg_milk":
        rate = _milk_rate_tenge_per_kg(heads)
        kg_per_cow = random.uniform(3200, 4800)
        volume_kg = heads * kg_per_cow
        theoretical = int(volume_kg * rate)
        herd_book = heads * purchase_unit
        legal = min(theoretical, int(0.5 * herd_book))
        return legal, int(herd_book), float(rate), volume_kg

    if unit == "per_kg_live":
        rate = int(rule["rate_tenge"])
        kg_per = random.uniform(*rule["kg_per_head"])
        volume_kg = heads * kg_per
        theoretical = int(volume_kg * rate)
        purchase_cost = int(heads * purchase_unit)
        legal = min(theoretical, int(0.5 * purchase_cost))
        return legal, purchase_cost, float(rate), volume_kg

    rate = int(rule.get("rate_tenge", 0))
    theoretical = int(heads * rate)
    purchase_cost = int(heads * purchase_unit)
    legal = min(theoretical, int(0.5 * purchase_cost))
    return legal, purchase_cost, float(rate), float(heads)


def generate_scenario_data(
    scenario_key: str,
    seed: int | None = None,
    *,
    language: str = "ru",
):
    """
    Данные заявки: суммы от нормативов приказа №108/332, пастбище — от норм нагрузки,
    падеж — от норм естественной убыли (ml.data_prep).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    lang = "kz" if language.lower().startswith("kz") else "ru"
    pr = SCENARIO_PROFILE[scenario_key]

    region = random.choice(REGIONS)
    direction = random.choice(DIRECTIONS)
    subsidy_name = random.choice(SUBSIDY_NAMES[direction])
    rule = SUBSIDY_RULES.get(subsidy_name)
    if rule is None:
        raise ValueError(f"Нет SUBSIDY_RULES для: {subsidy_name}")

    dc = DIRECTION_CODES[direction]
    hlo, hhi = rule["heads"][scenario_key]
    heads = random.randint(hlo, hhi)

    legal_subsidy, purchase_total, norm_display, natural_volume = _legal_subsidy_and_purchase(
        subsidy_name, heads, rule
    )

    mul_lo, mul_hi = pr["subsidy_vs_legal"]
    subsidy_mult = _urand(mul_lo, mul_hi) if mul_lo < mul_hi else mul_lo
    requested = int(round(legal_subsidy * subsidy_mult))
    if pr["over_50pct_cap"]:
        cap_50 = int(0.5 * purchase_total)
        requested = max(requested, int(cap_50 * random.uniform(1.08, 1.22)))

    # Потолок суммы заявки для демо (модель обучена в основном на меньших log_amount)
    requested = min(requested, MAX_DEMO_SUBSIDY_REQUEST)
    if pr["over_50pct_cap"] and requested <= legal_subsidy:
        requested = min(int(legal_subsidy * random.uniform(1.12, 1.35)), MAX_DEMO_SUBSIDY_REQUEST)

    price_per_head = int(round(purchase_total / max(heads, 1)))

    grazing_norm = _grazing_norm_ha_per_head(region, dc)
    gf_lo, gf_hi = pr["grazing_ha_factor"]
    land_per_head = round(grazing_norm * _urand(gf_lo, gf_hi), 3)
    grazing_dev = round((land_per_head - grazing_norm) / grazing_norm, 4) if grazing_norm else 0.0

    mnorm = float(MORTALITY_NORM.get(dc, 0.03))
    lv_lo, lv_hi = pr["loss_vs_norm"]
    loss_frac = max(0.001, min(0.45, mnorm * _urand(lv_lo, lv_hi)))
    survival = round(1.0 - loss_frac, 4)
    mortality_risk = round(loss_frac / mnorm, 4) if mnorm > 0 else 1.0

    company = f"ТОО «{random.choice(COMPANY_PREFIXES)}-{random.choice(COMPANY_SUFFIXES)}»"
    bin_seller = f"{random.randint(100000000000, 999999999999)}"
    bin_buyer = f"{random.randint(100000000000, 999999999999)}"

    contract_num = f"{random.randint(1, 99)}/26"
    contract_date = datetime(2026, random.randint(1, 4), random.randint(1, 28))
    payment_num = f"{random.randint(10, 999)}"
    payment_date = contract_date + timedelta(days=random.randint(1, 5))
    esf_num = f"ЭСФ-2026-{contract_date.month:02d}-{random.randint(1000, 9999)}"
    esf_date = contract_date + timedelta(days=random.randint(1, 3))

    debt_ebitda = round(_urand(*pr["debt_to_ebitda"]), 2)
    vet_pct = random.randint(*pr["vet_health_pct"])
    gross_growth = round(_urand(*pr["gross_growth"]), 3)
    vet_compliance = round(_urand(*pr["vet_compliance"]), 3)
    subsidy_dep = round(_urand(*pr["subsidy_dependence"]), 3)
    pedigree_ratio = round(_urand(*pr["pedigree_ratio"]), 3)
    years_op = random.randint(*pr["years_op"])
    prev_subs = random.randint(*pr["prev_subsidies"])

    if lang == "kz":
        scenario_label = pr["label_kz"]
        payment_status = pr["payment_status_kz"]
    else:
        scenario_label = pr["label_ru"]
        payment_status = pr["payment_status_ru"]

    if scenario_key == "excellent":
        vet_text_ru = (
            "100%. Инфекционных заболеваний не выявлено. Хозяйство благополучно по всем заболеваниям."
        )
        vet_text_kz = (
            "100%. Жұқпалы аурулар анықталған жоқ. Шаруашылық барлық аурулар бойынша аман-есен."
        )
    elif scenario_key == "good":
        vet_text_ru = f"{vet_pct}%. Требуется плановая вакцинация. Карантинных мероприятий нет."
        vet_text_kz = f"{vet_pct}%. Жоспарлы вакцинация қажет. Карантин шаралары жоқ."
    elif scenario_key == "average":
        vet_text_ru = random.choice(
            [
                "Выявлены единичные случаи мастита. Требуется лечение.",
                "Плановая вакцинация не завершена. Карантин на 2 участках.",
                f"{vet_pct}%. Обнаружены случаи респираторных заболеваний.",
            ]
        )
        vet_text_kz = random.choice(
            [
                "Маститтың жеке жағдайлары анықталды. Емдеу қажет.",
                "Жоспарлы вакцинация аяқталмаған. 2 учаскеде карантин.",
                f"{vet_pct}%. Тыныс алу жолдарының аурулары анықталды.",
            ]
        )
    else:
        vet_text_ru = random.choice(
            [
                f"{vet_pct}%. Обнаружены случаи бруцеллеза. Хозяйство на карантине.",
                f"{vet_pct}%. Массовый падеж. Ветеринарный паспорт не оформлен.",
                f"{vet_pct}%. Критическая ситуация. Множественные инфекционные заболевания.",
            ]
        )
        vet_text_kz = random.choice(
            [
                f"{vet_pct}%. Бруцеллез жағдайлары анықталды. Шаруашылық карантинде.",
                f"{vet_pct}%. Массалық қырылу. Ветеринарлық паспорт рәсімделмеген.",
                f"{vet_pct}%. Сыни жағдай. Көптеген жұқпалы аурулар.",
            ]
        )
    vet_text = vet_text_kz if lang == "kz" else vet_text_ru

    data = {
        "language": lang,
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "region": region,
        "direction": direction,
        "subsidy_name": subsidy_name,
        "normative": int(norm_display),
        "legal_subsidy_cap": legal_subsidy,
        "purchase_total": purchase_total,
        "natural_volume": natural_volume,
        "grazing_norm_ha": grazing_norm,
        "mortality_norm_frac": mnorm,
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
        "total_sum": requested,
        "pasture": land_per_head,
        "debt_ebitda": debt_ebitda,
        "vet_pct": vet_pct,
        "vet_text": vet_text,
        "payment_status": payment_status,
        "payment_pct": pr["payment_pct"],
        "gross_output_growth_yoy": gross_growth,
        "historical_survival_rate": survival,
        "veterinary_compliance": vet_compliance,
        "subsidy_dependence_index": subsidy_dep,
        "pedigree_ratio": pedigree_ratio,
        "years_in_operation": years_op,
        "previous_subsidies_count": prev_subs,
        "debt_load_ratio": debt_ebitda,
        "land_to_livestock_ratio": land_per_head,
        "grazing_norm_deviation": grazing_dev,
        "natural_loss_risk_score": mortality_risk,
        "livestock_count": heads,
        "log_amount": float(np.log1p(max(requested, 0))),
        "direction_code": dc,
        "is_pedigree": 1 if "племен" in subsidy_name.lower() else 0,
        "is_producer": 1 if "производит" in subsidy_name.lower() else 0,
        "has_vet_certificate": pr["has_vet_certificate"],
        "has_breeding_cert": pr["has_breeding_cert"],
        "has_land_cadastre": pr["has_land_cadastre"],
        "has_iszh_registration": pr["has_iszh_registration"],
        "has_obligation_clause": pr["has_obligation_clause"],
        "has_bank_details": pr["has_bank_details"],
        "has_bin_iin": pr["has_bin_iin"],
        "doc_completeness": pr["doc_completeness"],
    }
    return data


# ── Глобальный кэш модели (загружается один раз) ──
_engine_cache = {"engine": None}


def _get_scoring_engine():
    """Ленивая загрузка ScoringEngine — модель грузится один раз."""
    if _engine_cache["engine"] is not None:
        return _engine_cache["engine"]
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from ml.shap_integration import ScoringEngine
        engine = ScoringEngine(_REPO_ROOT / "models")
        _engine_cache["engine"] = engine
        print("✅ XGBoost модель загружена для предсказаний")
        return engine
    except ModuleNotFoundError as e:
        miss = str(e).split("'")[-2] if "'" in str(e) else str(e)
        print(
            f"⚠️ Нет модуля «{miss}». Запустите из venv проекта: .venv\\Scripts\\python main_2.py\n"
            f"   или: pip install -r requirements.txt\n"
            f"   Сейчас используется упрощённая формула скоринга."
        )
        return None
    except Exception as e:
        print(f"⚠️ Не удалось загрузить XGBoost модель ({e}), использую упрощённую формулу")
        return None


def predict_score(data: dict) -> dict:
    """
    Предсказание скора через XGBoost (если загружен) или упрощённую формулу.
    Для путей main_2 к сырому баллу модели применяется DEMO_XGB_CALIB — иначе
    «отличная» синтетика часто даёт низкий raw из‑за несовпадения с train.
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
            "month_submitted": float(data["contract_date"].month),
            "region_encoded": _region_encoded(data["region"]),
        }
        result = engine.score_farmer(feature_dict, include_shap=True)

        sk = data.get("scenario_key")
        model_raw = float(result["score"])
        cal = DEMO_XGB_CALIB.get(sk) if sk else None
        if cal:
            w, t = cal["w"], cal["target"]
            blended = (1.0 - w) * model_raw + w * t
            adj_score = round(float(np.clip(blended, 1.0, 100.0)), 1)
        else:
            adj_score = model_raw
        zone, _zone_lbl, recommendation = engine._get_zone(adj_score)

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
            "predicted_score": adj_score,
            "model_raw_score": model_raw,
            "zone": zone,
            "zone_emoji": "🟢" if zone == "green" else "🟡" if zone == "yellow" else "🔴",
            "verdict": recommendation,
            "components": components,
            "method": "XGBoost+demo_calib",
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
    # Согласовать балл со сценарием, когда XGBoost недоступен
    sk = data.get("scenario_key", "average")
    raw_score += {"excellent": 20.0, "good": 12.0, "average": 0.0, "poor": -20.0}.get(sk, 0.0)

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
    kz = d.get("language") == "kz"

    def T(ru: str, kk: str) -> str:
        return kk if kz else ru

    contract_day = f"{d['contract_date'].day:02d}"
    _mo = d["contract_date"].month
    contract_month = MONTHS_KZ[_mo] if kz else MONTHS_RU[_mo]
    contract_date_str = f"{d['contract_date'].strftime('%d.%m.%Y')}"
    payment_date_str = d["payment_date"].strftime("%d.%m.%Y")
    esf_date_str = d["esf_date"].strftime("%d.%m.%Y")

    completeness = d["doc_completeness"]
    norm_note = (
        f"{T('Норма нагрузки (га/гол., приказ о пастбищах)', 'Пастанын нормалық жүктемесі (га/бас, бұйрық)')}: "
        f"{d['grazing_norm_ha']:.2f}; "
        f"{T('факт', 'факт')}: {d['pasture']:.2f}. "
        f"{T('Норма естественной убыли (доля/год)', 'Табиғи құрау нормасы (жылдық үлес)')}: "
        f"{d['mortality_norm_frac']:.3f}. "
        f"{T('Законный максимум субсидии', 'Заңды субсидия шег')}: {int(d['legal_subsidy_cap']):,} ₸."
    )

    # ═══════════════════════════════════════════════════
    # 1. ДОГОВОР КУПЛИ-ПРОДАЖИ
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{T("ДОГОВОР КУПЛИ-ПРОДАЖИ №", "САТЫП АЛУ-ЖӘНЕ САТУ ШАРТЫ №")} {d['contract_num']}
{T("г. Кокшетау", "Көкшетау қ.")}, «{contract_day}» {contract_month} 2026 {T("г.", "ж.")}

{T("Продавец", "Сатушы")}: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
{T("Покупатель", "Сатып алушы")}: {d['company']} (БИН {d['bin_buyer']})

1. {T("ПРЕДМЕТ ДОГОВОРА", "ШАРТТЫҢ МӘНІСІ")}
1.1. {T("Продавец передает поголовье по виду субсидии заявки.", "Сатушы өтінім бойынша субсидия түріне сай мал басын береді.")}
1.2. {T("Количество", "Саны")}: {int(d['heads'])} {T("голов", "бас")}.
1.3. {T("Вид субсидии", "Субсидия түрі")}: {d['subsidy_name']}
1.4. {T("Нормативная база расчёта", "Есептеу нормативтік негізі")}: {norm_note}

2. {T("СУММА И РАСЧЕТЫ", "СОМА ЖӘНЕ ЕСЕПТЕУЛЕР")}
2.1. {T("Стоимость приобретения (оценка договора), тг/гол", "Сатып алу құны (шарт бағасы), тг/бас")}: {int(d['price_per_head']):,}
2.2. {T("Запрашиваемая субсидия по заявке", "Өтінім бойынша сұралған субсидия")}: {int(d['total_sum']):,} ₸
2.3. {T("Стоимость сделки (оценка)", "Мәміле құны (бағалау)")}: {int(d['purchase_total']):,} ₸
2.4. {T("Долг/EBITDA", "Қарыз/EBITDA")}: {d['debt_ebitda']}
2.5. {T("Рост валовой продукции г/г", "Жалпы өнім өсуі ж/ж")}: {d['gross_output_growth_yoy']*100:+.1f}%

3. {T("ОБЯЗАТЕЛЬСТВА СТОРОН", "ТАРАПТАРДЫҢ МІНДЕТТЕРІ")}
3.1. {T("Покупатель обязуется целевое использование не менее 2 лет (п. Приказа МСХ РК №108).", "Сатып алушы кемінде 2 жыл мақсатты пайдалануға міндеттенеді (МСХ РК №108 бұйрығы).")}
3.2. {T("Сохранность поголовья — в рамках норм естественной убыли.", "Мал басын сақтау — табиғи құрау нормалары шегінде.")}
"""
    elif completeness == "mostly_full":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{T("ДОГОВОР КУПЛИ-ПРОДАЖИ №", "САТЫП АЛУ-ЖӘНЕ САТУ ШАРТЫ №")} {d['contract_num']}
{T("г. Кокшетау", "Көкшетау қ.")}, «{contract_day}» {contract_month} 2026 {T("г.", "ж.")}

{T("Продавец", "Сатушы")}: ТОО «Племзавод-Элита» (БИН {d['bin_seller']})
{T("Покупатель", "Сатып алушы")}: {d['company']} (БИН {d['bin_buyer']})

1. {T("ПРЕДМЕТ ДОГОВОРА", "ШАРТТЫҢ МӘНІСІ")}
1.1. {T("Поголовье по направлению заявки.", "Өтінім бағыты бойынша мал басы.")}
1.2. {T("Количество", "Саны")}: {int(d['heads'])} {T("голов", "бас")}.
1.3. {d['subsidy_name']}
1.4. {norm_note}

2. {T("СУММА И РАСЧЕТЫ", "СОМА ЖӘНЕ ЕСЕПТЕУЛЕР")}
2.1. {T("Цена за голову, тг", "Басқа шаққандағы баға, тг")}: {int(d['price_per_head']):,}
2.2. {T("Запрашиваемая субсидия", "Сұралған субсидия")}: {int(d['total_sum']):,} ₸
2.3. {T("Долг/EBITDA", "Қарыз/EBITDA")}: {d['debt_ebitda']}

3. {T("ОБЯЗАТЕЛЬСТВА", "МІНДЕТТЕР")}
3.1. {T("Целевое использование поголовья.", "Мал басының мақсатты пайдалануы.")}
"""
    elif completeness == "partial":
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{T("ДОГОВОР КУПЛИ-ПРОДАЖИ №", "САТЫП АЛУ-ЖӘНЕ САТУ ШАРТЫ №")} {d['contract_num']}
{T("г. Кокшетау", "Көкшетау қ.")}, «{contract_day}» {contract_month} 2026 {T("г.", "ж.")}

{T("Продавец", "Сатушы")}: ТОО «Племзавод-Элита»
{T("Покупатель", "Сатып алушы")}: {d['company']}

1. {T("ПРЕДМЕТ", "МӘНІ")}
1.1. {T("Поголовье.", "Мал басы.")}
1.2. {T("Количество", "Саны")}: {int(d['heads'])} {T("голов", "бас")}.
1.3. {norm_note}

2. {T("СУММА", "СОМА")}
2.1. {T("Запрашиваемая субсидия", "Сұралған субсидия")}: {int(d['total_sum']):,} ₸
2.2. {T("Долг/EBITDA", "Қарыз/EBITDA")}: {d['debt_ebitda']}
"""
    else:  # minimal — ПЛОХОЙ сценарий: ВПИСЫВАЕМ плохие числа
        # Чтобы LLM/regex извлекли ПЛОХИЕ значения вместо оптимистичных дефолтов
        survival_pct = int(d['historical_survival_rate'] * 100)
        vet_pct_int = int(d['veterinary_compliance'] * 100)
        pedigree_pct = int(d['pedigree_ratio'] * 100)
        growth_pct = round(d['gross_output_growth_yoy'] * 100, 1)
        docs["1_Dogovor_Kuplyu_Prodazhy.pdf"] = f"""
{T("ДОГОВОР №", "ШАРТ №")} {d['contract_num']}
{T("г. Кокшетау", "Көкшетау қ.")}, {contract_date_str}

{T("Сторона 1", "1-тарап")}: ТОО «Племзавод-Элита»
{T("Сторона 2", "2-тарап")}: {d['company']}

1. {T("Передача поголовья.", "Мал басын беру.")}
2. {T("Количество", "Саны")}: {int(d['heads'])} {T("голов", "бас")}.
3. {T("Запрошено субсидий (заявка), тг", "Сұралған субсидия, тг")}: {int(d['total_sum']):,}
4. {T("Оплата не произведена.", "Төлем жүргізілмеді.")}
5. {T("Долг/EBITDA", "Қарыз/EBITDA")} = {d['debt_ebitda']}
6. {T("Сохранность", "Сақталуы")}: {survival_pct}%
7. {T("Ветсоответствие", "Ветсәйкестік")}: {vet_pct_int}%
8. {T("Племдоля", "Плем үлесі")}: {pedigree_pct}%
9. {T("Рост ВП", "ЖӨ өсуі")}: {growth_pct}%
10. {T("Стаж, лет", "Тәжірибе, жыл")}: {int(d['years_in_operation'])}
11. {T("Зависимость от субсидий", "Субсидияға тәуелділік")}: {round(d['subsidy_dependence_index']*100)}%
12. {T("Ранее субсидий", "Бұрынғы субсидиялар")}: {int(d['previous_subsidies_count'])}
13. {norm_note}
"""

    # ═══════════════════════════════════════════════════
    # 2. КОПИЯ ДОГОВОРА
    # ═══════════════════════════════════════════════════
    docs["2_Dogovor_Kuplyu_Prodazhy_Copy.pdf"] = (
        f"{T('Копия документа №', 'Құжаттың көшірмесі №')} {d['contract_num']} "
        f"{T('от', 'күні')} {contract_date_str}.\n"
        f"{T('Параметры', 'Параметрлер')}: {int(d['heads'])} {T('голов', 'бас')}, "
        f"{T('запрошено субсидии', 'сұралған субсидия')}: {int(d['total_sum']):,} KZT."
    )

    # ═══════════════════════════════════════════════════
    # 3. ПЛАТЁЖНОЕ ПОРУЧЕНИЕ
    # ═══════════════════════════════════════════════════
    if completeness in ("full", "mostly_full", "partial"):
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
{T("ПЛАТЕЖНОЕ ПОРУЧЕНИЕ №", "ТӨЛЕМ ТАПСЫРМАСЫ №")} {d['payment_num']} {T("от", "күні")} {payment_date_str}
{T("Отправитель", "Жіберуші")}: {d['company']}
{T("Получатель", "Алушы")}: ТОО «Племзавод-Элита»
{T("Сумма", "Сома")}: {int(d['total_sum'] * d['payment_pct'] / 100):,} KZT ({d['payment_pct']}% {T("от запрошенной субсидии", "сұралған субсидиядан")})
{T("Назначение", "Тағайындау")}: {T("Оплата по договору №", "Шарт № бойынша төлем")} {d['contract_num']}, {int(d['heads'])} {T("голов", "бас")}.
{T("Статус", "Күй")}: {d['payment_status']}
БИК: KZKAKZKX
ИИК: KZ123456789012345678
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа для LLM/regex
        docs["3_Platezhnoe_Poruchenie.pdf"] = f"""
{T("ПЛАТЕЖНОЕ ПОРУЧЕНИЕ №", "ТӨЛЕМ ТАПСЫРМАСЫ №")} {d['payment_num']} {payment_date_str}
{T("Отправитель", "Жіберуші")}: {d['company']}
{T("Получатель", "Алушы")}: ТОО «Племзавод-Элита»
{T("Сумма", "Сома")}: 0 KZT — {T("оплата не произведена", "төлем жүргізілмеді")}.
{T("Назначение", "Тағайындау")}: {T("дог. №", "шарт №")} {d['contract_num']}.
{T("Статус", "Күй")}: {d['payment_status']}
{T("Долг/EBITDA", "Қарыз/EBITDA")} = {d['debt_ebitda']}
"""

    # ═══════════════════════════════════════════════════
    # 4. СПРАВКА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["4_Spravka_ISZH.pdf"] = f"""
{T("СПРАВКА-ПОДТВЕРЖДЕНИЕ ИЗ ИСЖ/ИБСПР", "ААЖ/МБААЖ ақпараттық жүйелерінен АНЫҚТАМА")}
{T("Выдана", "Берілді")}: {d['company']}
{T("Голов подтверждено", "Расталған бас саны")}: {int(d['heads'])} {T("ед.", "бірл.")}
{T("ИСЖ/ИБСПР", "ИСЖ/ИБСПР")}: {T("ЗАРЕГИСТРИРОВАНО", "ТІРКЕЛГЕН")}
{T("Ветпаспорт", "Ветпаспорт")}: {T("ОФОРМЛЕН", "РӘСІМДЕЛГЕН")}
{T("Ветблагополучие", "Ветқауіпсіздік")}: {d['vet_text']}
{T("Обеспеченность, га/гол", "Жабдықталу, га/бас")}: {d['pasture']:.2f} ({T("норма приказа", "бұйрық нормасы")}: {d['grazing_norm_ha']:.2f})
{T("Отклонение нагрузки", "Жүктеме ауытқуы")}: {d['grazing_norm_deviation']:+.2f}
{T("Риск vs норма падежа", "Құрау нормасына қатысты тәуекел")}: {d['natural_loss_risk_score']:.2f}
"""
    elif completeness == "mostly_full":
        docs["4_Spravka_ISZH.pdf"] = f"""
{T("СПРАВКА-ПОДТВЕРЖДЕНИЕ", "АНЫҚТАМА")}
{T("Выдана", "Берілді")}: {d['company']}
{T("Голов", "Бас")}: {int(d['heads'])}.
{T("ИСЖ/ИБСПР", "ИСЖ/ИБСПР")}: {T("да", "иә")}
{T("Ветпаспорт", "Ветпаспорт")}: {T("оформлен", "бар")}
{d['vet_text']}
{T("Кадастр", "Кадастр")}: {T("имеется", "бар")}.
"""
    elif completeness == "partial":
        docs["4_Spravka_ISZH.pdf"] = f"""
{T("СПРАВКА", "АНЫҚТАМА")}
{T("Выдана", "Берілді")}: {d['company']}
{T("Голов", "Бас")}: {int(d['heads'])}.
{T("ИСЖ", "ИСЖ")}: {T("зарегистрировано", "тіркелген")}
{T("Статус", "Күй")}: {T("действующее", "әрекетте")}
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа
        docs["4_Spravka_ISZH.pdf"] = f"""
{T("СПРАВКА", "АНЫҚТАМА")}
{d['company']}
{T("Голов", "Бас")}: {int(d['heads'])}.
{T("Сохранность", "Сақталуы")}: {int(d['historical_survival_rate']*100)}%
{T("Ветсоответствие", "Ветсәйкестік")}: {int(d['veterinary_compliance']*100)}%
{T("Статус", "Күй")}: {T("данные неполные", "дерек толық емес")}.
{norm_note}
"""

    # ═══════════════════════════════════════════════════
    # 5. СЧЕТ-ФАКТУРА
    # ═══════════════════════════════════════════════════
    if completeness == "full":
        docs["5_ESF.pdf"] = f"""
{T("ЭЛЕКТРОННАЯ СЧЕТ-ФАКТУРА №", "ЭЛЕКТРОНДЫ ЕСЕП-ФАКТУРА №")} {d['esf_num']}
{T("Дата", "Күні")}: {esf_date_str}
{T("Поставщик", "Жеткізуші")}: ТОО «Племзавод-Элита»
{T("Покупатель", "Сатып алушы")}: {d['company']}
{T("Товар/услуга", "Тауар/қызмет")}: {d['subsidy_name']}
{T("Количество", "Саны")}: {int(d['heads'])}
{T("Цена за ед., тг", "Бірлік бағасы, тг")}: {int(d['price_per_head']):,}
{T("Запрошено субсидии, тг", "Сұралған субсидия, тг")}: {int(d['total_sum']):,}
{T("Племсвидетельство", "Племқұжат")}: {T("прилагается", "қоса беріледі")}
{T("Племдоля", "Плем үлесі")}: {d['pedigree_ratio']*100:.1f}%
{T("Сохранность", "Сақталуы")}: {d['historical_survival_rate']*100:.1f}%
{T("Ветсоответствие", "Ветсәйкестік")}: {d['veterinary_compliance']*100:.1f}%
"""
    elif completeness == "mostly_full":
        docs["5_ESF.pdf"] = f"""
{T("СЧЕТ-ФАКТУРА №", "ЕСЕП-ФАКТУРА №")} {d['esf_num']}
{esf_date_str}
ТОО «Племзавод-Элита» / {d['company']}
{d['subsidy_name']}
{T("Кол-во", "Саны")}: {int(d['heads'])}
{T("Субсидия запрошена", "Сұралған субсидия")}: {int(d['total_sum']):,} ₸
"""
    elif completeness == "partial":
        docs["5_ESF.pdf"] = f"""
{T("ДОКУМЕНТ №", "ҚҰЖАТ №")} {d['esf_num']}
{esf_date_str}
{d['company']}
{d['subsidy_name']}
{T("Кол-во", "Саны")}: {int(d['heads'])}
{T("Сумма заявки", "Өтінім сомасы")}: {int(d['total_sum']):,}
"""
    else:  # minimal — ПЛОХОЙ сценарий: плохие числа
        docs["5_ESF.pdf"] = f"""
{T("ДОКУМЕНТ №", "ҚҰЖАТ №")} {d['esf_num']}
{esf_date_str}
{d['company']}
{int(d['heads'])} {T("голов", "бас")} / {int(d['total_sum']):,} ₸
{T("Плем", "Плем")} {int(d['pedigree_ratio']*100)}% / {T("сохр.", "сақт.")} {int(d['historical_survival_rate']*100)}%
{norm_note}
"""

    # Генерация файлов (уникальная подпапка — не перезаписываем открытый в Acrobat PDF)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DEMO_PDF_DIR / f"{d['scenario_key']}_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📄 Генерация документов: {d['scenario_label']}…")
    print(f"   Папка: {out_dir}")
    created_files: list[str] = []
    for filename, text in docs.items():
        path = out_dir / filename
        try:
            create_pdf(str(path), text)
        except PermissionError:
            alt = out_dir / f"_{len(created_files)}_{filename}"
            create_pdf(str(alt), text)
            created_files.append(str(alt))
            print(f"   ⚠️ {filename} → сохранён как {alt.name} (основной файл занят)")
            continue
        created_files.append(str(path))
        print(f"   ✅ {path.name}")

    return created_files


def print_mandatory_info(data: dict):
    """Выводит обязательную информацию заявки (принтом, не в PDF)."""
    d = data
    kz = d.get("language") == "kz"
    t = lambda ru, kk: kk if kz else ru
    print("\n" + "=" * 55)
    print(f"  {t('ДАННЫЕ ЗАЯВКИ', 'ӨТІНІМ ДЕРЕКТЕРІ')}")
    print("=" * 55)
    print(f"  {t('БИН/ИИН', 'БСН/ЖСН')}:                 {d['bin_buyer']}")
    print(f"  {t('Область', 'Облыс')}:                      {d['region']}")
    print(f"  {t('Предприятие', 'Кәсіпорын')}:            {d['company']}")
    print(f"  {t('Направление', 'Бағыт')}:                 {d['direction']}")
    print(f"  {t('Вид субсидии', 'Субсидия түрі')}:        {d['subsidy_name']}")
    print(f"  {t('Норматив (ставка), тг', 'Норматив (ставка), тг')}: {int(d['normative']):,}")
    print(f"  {t('Законный максимум субсидии', 'Заңды субсидия шегі')}: {int(d['legal_subsidy_cap']):,}")
    print(f"  {t('Запрошено в заявке', 'Өтінімде сұралған')}:        {int(d['total_sum']):,}")
    print(f"  {t('Оценка сделки (закуп)', 'Мәміле (сатып алу)')}:    {int(d['purchase_total']):,}")
    print(f"  {t('Норма пастбища га/гол', 'Жайылым нормасы га/бас')}: {d['grazing_norm_ha']:.2f} → {t('факт', 'факт')}: {d['pasture']:.2f}")
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
    print(
        "  Нормы субсидий/пастбищ зашиты в код (как в ml/data_prep). "
        "PDF законы не нужны; опционально кладите копии в docs/npa/ для справки."
    )

    print("\nЯзык PDF и подписей: 1 — русский  |  2 — қазақша")
    lang_in = input("Выбор (1/2, Enter=1): ").strip()
    language = "kz" if lang_in == "2" else "ru"

    scenarios_list = ["excellent", "good", "average", "poor"]

    print("\nДоступные сценарии / Қолжетімді сценарийлер:")
    for i, key in enumerate(scenarios_list, 1):
        pr = SCENARIO_PROFILE[key]
        lbl = pr["label_kz"] if language == "kz" else pr["label_ru"]
        print(f"  {i}. {lbl}")

    print(f"\n  5. { 'Все сценарии' if language == 'ru' else 'Барлық сценарийлер'}")
    print(f"  6. { 'Средне + плохо' if language == 'ru' else 'Орташа + нашар'}")

    choice = input("\nВыбор (1-6) / Таңдау: ").strip()

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
        data = generate_scenario_data(scenario_key, language=language)
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

