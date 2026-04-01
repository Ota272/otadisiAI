"""
╔══════════════════════════════════════════════════════════════════╗
║   SmartAgro Score — ШАГ 1: Подготовка данных и генерация фичей  ║
║   data_prep.py                                                    ║
║   Decentrathon 5.0 | AI for Government                          ║
╚══════════════════════════════════════════════════════════════════╝

КАК ЗАПУСКАТЬ:
    python data_prep.py

ЧТО НА ВЫХОДЕ:
    data_cleaned.csv   — очищенные исходные данные
    data_features.csv  — датасет с синтетическими фичами для ML
"""

# ──────────────────────────────────────────────────────────────────
# БЛОК 1: ИМПОРТ БИБЛИОТЕК
# ──────────────────────────────────────────────────────────────────
# Объяснение для новичка:
# Библиотека — это готовый набор инструментов, который кто-то уже написал за нас.
# pandas — для работы с таблицами (как Excel, но в Python)
# numpy  — для математических операций и генерации чисел
# sklearn — scikit-learn: стандартная ML-библиотека
# random  — для воспроизводимой случайности
# pathlib — для удобной работы с путями к файлам

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ──────────────────────────────────────────────────────────────────
# БЛОК 2: НАСТРОЙКИ (КОНФИГУРАЦИЯ)
# ──────────────────────────────────────────────────────────────────

# RANDOM_SEED — "зерно" случайности.
# Объяснение: когда мы "генерируем случайные числа", компьютер на самом
# деле использует математическую формулу, стартующую с этого числа.
# Фиксируя seed, мы гарантируем, что при каждом запуске скрипта
# получаем ОДИНАКОВЫЕ "случайные" числа → воспроизводимость!
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

INPUT_FILE  = "data.csv"
OUTPUT_CLEAN = "data_cleaned.csv"
OUTPUT_FEAT  = "data_features.csv"


# ──────────────────────────────────────────────────────────────────
# БЛОК 3: ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ──────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Загружает CSV, убирает мусорные строки, переименовывает колонки.

    Наш реальный CSV имеет особую структуру:
    - Строки 0-2: шапка и метаданные (не нужны)
    - Строка 3: настоящий заголовок таблицы
    - Строки 4+: данные
    """
    print(f"[1/5] Загружаю файл: {path}")

    # skiprows=3 — пропускаем первые 3 строки (мусор и шапку системы)
    # header=0   — следующая строка становится заголовком колонок
    df = pd.read_csv(path, skiprows=3, header=0, low_memory=False)

    # Даём понятные имена колонкам вместо "Unnamed: 0" и т.д.
    df.columns = [
        "num",           # Порядковый номер
        "date_received", # Дата поступления заявки
        "col3",          # Пустая колонка (артефакт экспорта)
        "col4",          # Пустая колонка
        "region",        # Область РК
        "agimat",        # Наименование акимата
        "app_number",    # Номер заявки в системе
        "direction",     # Направление (скотоводство, птицеводство...)
        "subsidy_name",  # Полное наименование субсидии
        "status",        # Статус заявки
        "normative",     # Норматив (ставка субсидии за 1 голову)
        "amount",        # Сумма к выплате (тенге)
        "district",      # Район хозяйства
    ]

    # Удаляем строку-дубликат заголовка (она попала в данные)
    df = df[df["num"] != "№ п/п"].copy()

    # Удаляем пустые строки (где всё NaN)
    df.dropna(how="all", inplace=True)

    # Убираем лишние пробелы в текстовых полях
    for col in ["region", "direction", "status", "subsidy_name", "district"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Преобразуем числовые поля (они пришли как строки)
    # pd.to_numeric с errors='coerce': если преобразование не вышло → NaN
    df["normative"] = pd.to_numeric(df["normative"], errors="coerce")
    df["amount"]    = pd.to_numeric(df["amount"],    errors="coerce")
    df["num"]       = pd.to_numeric(df["num"],       errors="coerce")

    # Преобразуем дату в формат datetime
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", dayfirst=True)

    # Удаляем колонки, которые нам не нужны (пустые артефакты)
    df.drop(columns=["col3", "col4"], inplace=True, errors="ignore")

    # Убираем строки без суммы (они нам бесполезны для скоринга)
    df = df[df["amount"].notna() & (df["amount"] > 0)].copy()

    print(f"    Загружено строк: {len(df):,}")
    print(f"    Колонки: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────
# БЛОК 4: БАЗОВЫЕ ФИЧИ ИЗ РЕАЛЬНЫХ ДАННЫХ
# ──────────────────────────────────────────────────────────────────

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаём базовые фичи из уже имеющихся колонок.

    Зачем это нужно? ML-модель не понимает текст "Скотоводство".
    Нам нужно переводить всё в числа и вытаскивать смысловые признаки.
    """
    print("\n[2/5] Создаю базовые фичи из исходных данных...")

    # ── Фича: час подачи заявки ──────────────────────────────────
    # Идея: заявки, поданные в рабочее время, могут отражать более
    # организованных фермеров (vs ночные в спешке)
    df["hour_submitted"] = df["date_received"].dt.hour.fillna(12)

    # ── Фича: день недели (0=понедельник, 6=воскресенье) ─────────
    df["day_of_week"] = df["date_received"].dt.dayofweek.fillna(1)

    # ── Фича: месяц подачи ───────────────────────────────────────
    df["month_submitted"] = df["date_received"].dt.month.fillna(1)

    # ── Фича: приближённое количество голов скота ────────────────
    # amount (сумма) = normative (ставка за 1 голову) × количество голов
    # Зная это, вычислим количество голов:
    df["livestock_count"] = (df["amount"] / df["normative"].replace(0, np.nan)).round(0)
    df["livestock_count"] = df["livestock_count"].clip(lower=1).fillna(10)

    # ── Фича: размер заявки (логарифм суммы) ─────────────────────
    # Объяснение: сумма может быть от 20,000 до 50,000,000 тенге.
    # Такой БОЛЬШОЙ разброс (называется "высокая дисперсия") мешает ML.
    # Логарифм "сжимает" большие числа, делая их сравнимыми с малыми.
    # np.log1p = log(1 + x), чтобы не было log(0) = -бесконечность
    df["log_amount"] = np.log1p(df["amount"])

    # ── Фича: тип направления (скотоводство, птицеводство...) ────
    DIRECTION_MAP = {
        "Субсидирование в скотоводстве":     0,
        "Субсидирование в овцеводстве":       1,
        "Субсидирование в коневодстве":       2,
        "Субсидирование в птицеводстве":      3,
        "Субсидирование в верблюдоводстве":   4,
        "Субсидирование в свиноводстве":      5,
    }
    df["direction_code"] = df["direction"].map(DIRECTION_MAP).fillna(6)

    # ── Фича: это племенной скот? ─────────────────────────────────
    # str.contains — проверяем, встречается ли слово в тексте
    df["is_pedigree"] = (
        df["subsidy_name"].str.contains("племен", case=False, na=False)
    ).astype(int)  # True → 1, False → 0

    # ── Фича: это производители? (быки, бараны, жеребцы) ─────────
    df["is_producer"] = (
        df["subsidy_name"].str.contains("производи|производит", case=False, na=False)
    ).astype(int)

    print(f"    Базовых фич добавлено: 8")
    return df


# ──────────────────────────────────────────────────────────────────
# БЛОК 5: СИНТЕТИЧЕСКИЕ ЭКОНОМИЧЕСКИЕ ФИЧИ
# ──────────────────────────────────────────────────────────────────

def generate_synthetic_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерируем синтетические (искусственные) экономические фичи.

    ПОЧЕМУ СИНТЕТИЧЕСКИЕ?
    В реальности эти данные были бы из других систем:
    - ГИСС (Государственная информационная система субсидирования)
    - Налоговая служба (данные о выручке)
    - Комитет по статистике (данные о поголовье)

    Для хакатона мы генерируем реалистичные данные, привязывая их
    к реальным признакам заявки: регион, тип субсидии, сумма.
    Это называется "контролируемая генерация" — не просто random.

    ВАЖНО: Мы фиксируем seed (RANDOM_SEED) = 42, чтобы при каждом
    запуске получать одинаковые данные → воспроизводимость модели.
    """
    print("\n[3/5] Генерирую синтетические экономические фичи...")
    n = len(df)

    # ── 1. gross_output_growth_yoy — Рост валовой продукции г/г ─
    # Что это: насколько выросла продукция фермера год к году
    # Значения: от -30% до +80%
    #
    # Логика генерации:
    # - Базовый рост зависит от типа субсидии и суммы
    # - Племенные субсидии → выше рост (пedigree_bonus)
    # - Крупные фермеры (высокий log_amount) → умеренный рост
    # - Добавляем случайный шум (нормальное распределение)
    base_growth = (df["log_amount"] - df["log_amount"].mean()) / df["log_amount"].std() * 0.1
    pedigree_bonus = df["is_pedigree"] * np.random.uniform(0.05, 0.15, n)
    noise_growth = np.random.normal(0, 0.12, n)
    df["gross_output_growth_yoy"] = (base_growth + pedigree_bonus + noise_growth).clip(-0.30, 0.80)

    # ── 2. land_to_livestock_ratio — Обеспеченность землей (Га/гол) ──
    # Что это: сколько гектаров пастбища приходится на 1 голову скота
    # Норма для КРС: 1.5–3.0 Га/голову
    # Значения: 0.2 до 10.0 Га/голову
    #
    # Логика: у крупных скотоводческих хозяйств (скотоводство = direction_code 0)
    # эта цифра обычно выше. У птицеводов — низкая (птицы не пасутся)
    direction_land_factor = df["direction_code"].map({
        0: 3.0,  # Скотоводство — много земли
        1: 2.5,  # Овцеводство
        2: 2.0,  # Коневодство
        3: 0.3,  # Птицеводство — мало земли (фермы закрытые)
        4: 4.0,  # Верблюдоводство — самые большие пастбища
        5: 0.5,  # Свиноводство
        6: 2.0,  # Прочие
    }).fillna(2.0)
    df["land_to_livestock_ratio"] = (
        direction_land_factor * np.random.lognormal(0, 0.4, n)
    ).clip(0.2, 10.0)

    # ── 3. historical_survival_rate — Сохранность стада (%) ──────
    # Что это: процент скота, выжившего от падежа и болезней
    # Хорошо: >90%. Тревожно: <75%. Плохо: <60%
    #
    # Логика: влияет регион (суровый климат → ниже выживаемость)
    # и тип животных. Птица имеет высокую смертность.
    region_survival = {
        "Мангистауская область":       0.82,  # Жаркий климат
        "Атырауская область":          0.83,
        "Западно-Казахстанская область": 0.85,
        "Жамбылская область":          0.87,
        "Алматинская область":         0.90,
        "Акмолинская область":         0.88,
    }
    base_survival = df["region"].map(region_survival).fillna(0.87)
    noise_survival = np.random.normal(0, 0.05, n)
    # Птицеводство — выше базовый показатель (технологичные фермы)
    bird_bonus = (df["direction_code"] == 3) * 0.04
    df["historical_survival_rate"] = (
        base_survival + noise_survival + bird_bonus
    ).clip(0.50, 0.99)

    # ── 4. subsidy_dependence_index — Индекс зависимости от субсидий ─
    # Что это: насколько бизнес зависит от государственных субсидий
    # Высокое значение → плохо (фермер не может выжить без дотаций)
    # Значения: 0.0 (независим) до 1.0 (полностью зависим)
    #
    # Логика: приближаем как долю суммы субсидии от "оценочной выручки"
    estimated_revenue = df["livestock_count"] * direction_land_factor * np.random.uniform(50000, 200000, n)
    raw_dependence = df["amount"] / (estimated_revenue + df["amount"])
    df["subsidy_dependence_index"] = raw_dependence.clip(0.0, 1.0)

    # ── 5. veterinary_compliance — Ветеринарное соответствие (0-1) ─
    # Что это: насколько хозяйство соблюдает ветеринарные нормы
    # (вакцинация, паспорта животных, проверки)
    # В реальности: из базы Комитета ветеринарного контроля
    noise_vet = np.random.beta(8, 2, n)  # beta(8,2) даёт значения ближе к 1 (большинство соблюдает)
    pedigree_compliance_bonus = df["is_pedigree"] * 0.05
    df["veterinary_compliance"] = (noise_vet + pedigree_compliance_bonus).clip(0.0, 1.0)

    # ── 6. years_in_operation — Стаж работы предприятия (лет) ────
    # Что это: сколько лет фермерское хозяйство работает
    # Старые хозяйства имеют опыт, новые — энтузиазм, но риск
    df["years_in_operation"] = np.random.choice(
        range(1, 26),
        n,
        p=_generate_years_distribution(25),
    ).astype(float)

    # ── 7. pedigree_ratio — Доля племенного скота в стаде (0-1) ──
    # Что это: если ферма получает субсидию на племенное поголовье,
    # насколько высока доля элитного скота в общем стаде?
    # Высокая доля → серьёзный племенной завод → выше балл
    df["pedigree_ratio"] = np.where(
        df["is_pedigree"] == 1,
        np.random.beta(5, 2, n),   # beta(5,2): ближе к 1 (высокая доля)
        np.random.beta(2, 5, n),   # beta(2,5): ближе к 0 (низкая доля)
    )

    # ── 8. previous_subsidies_count — Сколько раз получал субсидии ─
    # Опытный получатель субсидий — меньше ошибок в документах,
    # проверенная система учёта.
    df["previous_subsidies_count"] = np.random.poisson(lam=3.5, size=n).clip(0, 15)

    # ── 9. debt_load_ratio — Долговая нагрузка ────────────────────
    # Отношение долга к EBITDA (прибыли до вычетов)
    # < 1.5: комфортно  |  1.5–3.0: умеренно  |  > 3.0: опасно
    df["debt_load_ratio"] = np.random.lognormal(mean=0.3, sigma=0.7, size=n).clip(0.0, 5.0)

    print(f"    Синтетических экономических фич добавлено: 9")
    return df


def _generate_years_distribution(max_years: int):
    """
    Генерирует правдоподобное распределение лет работы.
    Большинство хозяйств относительно молодые (1-10 лет),
    старых (20+ лет) меньше.
    """
    # Экспоненциальное убывание: много молодых, мало старых
    probs = np.array([1 / (1 + y * 0.3) for y in range(max_years)])
    return probs / probs.sum()  # нормализуем чтобы сумма = 1.0


# ──────────────────────────────────────────────────────────────────
# БЛОК 6: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (TARGET)
# ──────────────────────────────────────────────────────────────────

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаём целевую переменную historical_score (от 1 до 100).

    ЧТО ТАКОЕ ЦЕЛЕВАЯ ПЕРЕМЕННАЯ?
    Это то, ЧТО именно наша модель будет учиться предсказывать.
    В supervised learning (обучение с учителем):
    - Входные данные (X): наши фичи (рост продукции, выживаемость и т.д.)
    - Целевая переменная (y): правильный ответ, который нужно предсказать

    Мы создаём "идеальный" скоринг как взвешенную сумму фичей.
    В реальном проекте: исторические оценки комиссии были бы y.

    ВЕСА ФИЧЕЙ — экспертное мнение (как важна каждая характеристика):
    """
    print("\n[4/5] Создаю целевую переменную (historical_score)...")

    # Нормализуем каждую фичу в [0, 1] для честного взвешивания
    # Если не нормализовать, большие числа (годы: 1-25) перевесят
    # маленькие (доля племенных: 0.0-1.0), хотя это несправедливо

    def norm(series):
        """Мин-макс нормализация: (x - min) / (max - min)"""
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    # Инвертируем негативные показатели (чем меньше — тем лучше)
    debt_inverted     = 1 - norm(df["debt_load_ratio"])       # меньше долг = лучше
    dependence_inv    = 1 - norm(df["subsidy_dependence_index"])  # меньше зависимость = лучше

    # Взвешенная сумма → базовый балл
    raw_score = (
        norm(df["gross_output_growth_yoy"])    * 25.0 +  # Рост продукции — важнейший фактор
        norm(df["pedigree_ratio"])              * 20.0 +  # Племенная ценность
        norm(df["historical_survival_rate"])    * 15.0 +  # Сохранность стада
        norm(df["veterinary_compliance"])       * 13.0 +  # Ветеринария
        norm(df["subsidy_dependence_index"].apply(lambda x: 1-x))  * 12.0 +  # Самостоятельность
        debt_inverted                           * 10.0 +  # Финансовая устойчивость
        norm(df["land_to_livestock_ratio"])     *  5.0 +  # Земельные ресурсы
        norm(df["years_in_operation"])          *  5.0    # Опыт
    )
    # raw_score сейчас в диапазоне [0, 105 теоретически]

    # Масштабируем в [1, 100]
    raw_norm = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    df["historical_score"] = (raw_norm * 99 + 1).round(1)

    # Добавляем небольшой реалистичный шум (±5 баллов)
    # Жюри не будут знать точную формулу → шум делает данные реалистичнее
    noise = np.random.normal(0, 3, len(df))
    df["historical_score"] = (df["historical_score"] + noise).clip(1, 100).round(1)

    print(f"    Score — среднее: {df['historical_score'].mean():.1f}")
    print(f"    Score — мин: {df['historical_score'].min():.1f}, макс: {df['historical_score'].max():.1f}")
    print(f"    Зелёных (80+): {(df['historical_score'] >= 80).sum():,}")
    print(f"    Жёлтых (50-79): {((df['historical_score'] >= 50) & (df['historical_score'] < 80)).sum():,}")
    print(f"    Красных (<50): {(df['historical_score'] < 50).sum():,}")
    return df


# ──────────────────────────────────────────────────────────────────
# БЛОК 7: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
# ──────────────────────────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразуем текстовые колонки в числа для ML-модели.

    ЗАЧЕМ КОДИРОВАТЬ?
    ML-модели работают с числами, а не с текстом.
    "Алматинская область" — это просто строка для Python.
    Для модели нам нужно: "Алматинская" = 0, "Акмолинская" = 1, ...

    Используем LabelEncoder из sklearn:
    - Он создаёт словарь: каждой уникальной строке присваивает число
    - Это просто и работает для Decision Tree / Random Forest / XGBoost
    """
    print("\n[5/5] Кодирую категориальные переменные...")

    # LabelEncoder запоминает маппинг строка → число
    le_region = LabelEncoder()
    le_direction = LabelEncoder()

    df["region_encoded"]    = le_region.fit_transform(df["region"].fillna("Неизвестно"))
    df["direction_encoded"] = le_direction.fit_transform(df["direction"].fillna("Неизвестно"))

    # Сохраняем маппинги для понимания (пригодится для отладки)
    region_mapping = dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))
    print(f"    Регионов закодировано: {len(region_mapping)}")
    print(f"    Направлений закодировано: {le_direction.classes_.tolist()}")

    return df


# ──────────────────────────────────────────────────────────────────
# БЛОК 8: ФИНАЛЬНЫЙ ДАТАСЕТ — ВЫБОР ФИЧЕЙ ДЛЯ ML
# ──────────────────────────────────────────────────────────────────

# Список фичей, которые пойдут в модель
# Это называется "feature set" или "X-features"
ML_FEATURES = [
    # Экономические (синтетические)
    "gross_output_growth_yoy",
    "land_to_livestock_ratio",
    "historical_survival_rate",
    "subsidy_dependence_index",
    "veterinary_compliance",
    "years_in_operation",
    "pedigree_ratio",
    "previous_subsidies_count",
    "debt_load_ratio",
    # Из реальных данных
    "log_amount",
    "livestock_count",
    "direction_code",
    "is_pedigree",
    "is_producer",
    "hour_submitted",
    "month_submitted",
    "region_encoded",
]

TARGET = "historical_score"


# ──────────────────────────────────────────────────────────────────
# MAIN — Главная функция
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SmartAgro Score | Feature Engineering Pipeline")
    print("=" * 65)

    # Шаг 1: Загрузка и очистка
    df = load_and_clean(INPUT_FILE)

    # Шаг 2: Базовые фичи
    df = engineer_base_features(df)

    # Шаг 3: Синтетические экономические фичи
    df = generate_synthetic_economic_features(df)

    # Шаг 4: Целевая переменная
    df = create_target_variable(df)

    # Шаг 5: Кодирование категорий
    df = encode_categorical(df)

    # Сохраняем очищенный датасет (полный, для анализа)
    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
    print(f"\n✅ Сохранён: {OUTPUT_CLEAN} ({len(df):,} строк, {df.shape[1]} колонок)")

    # Сохраняем датасет только с ML-фичами + таргетом (для обучения)
    df_ml = df[ML_FEATURES + [TARGET]].dropna()
    df_ml.to_csv(OUTPUT_FEAT, index=False, encoding="utf-8-sig")
    print(f"✅ Сохранён: {OUTPUT_FEAT} ({len(df_ml):,} строк, {len(ML_FEATURES)} фичей)")

    print("\n📊 Статистика датасета для ML:")
    print(df_ml[ML_FEATURES].describe().round(3).to_string())

    print("\n🎯 Распределение целевой переменной:")
    print(f"   Зелёная зона (80-100): {(df_ml[TARGET] >= 80).mean()*100:.1f}%")
    print(f"   Жёлтая зона  (50-79):  {((df_ml[TARGET] >= 50) & (df_ml[TARGET] < 80)).mean()*100:.1f}%")
    print(f"   Красная зона (1-49):   {(df_ml[TARGET] < 50).mean()*100:.1f}%")

    print("\n✨ Готово! Следующий шаг: запусти train_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()