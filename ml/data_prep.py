import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

INPUT_FILE  = str(Path(__file__).parent.parent / "data" / "data_combined.csv")
OUTPUT_CLEAN = str(Path(__file__).parent.parent / "data" / "data_cleaned_combined.csv")
OUTPUT_FEAT  = str(Path(__file__).parent.parent / "data" / "data_features_combined.csv")

def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/5] Загружаю файл: {path}")

    # data_combined.csv — обычный CSV с заголовком (без skiprows)
    df = pd.read_csv(path, low_memory=False)

    # Если это старый data.csv — обрабатываем как раньше
    if "num" not in df.columns:
        df.columns = [
            "num",
            "date_received",
            "col3",
            "col4",
            "region",
            "agimat",
            "app_number",
            "direction",
            "subsidy_name",
            "status",
            "normative",
            "amount",
            "district",
        ]
        df = df[df["num"] != "№ п/п"].copy()
        df.drop(columns=["col3", "col4"], inplace=True, errors="ignore")
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", dayfirst=True)
        df = df[df["amount"].notna() & (df["amount"] > 0)].copy()
    else:
        # data_combined.csv — уже готовый, только чистим
        df.dropna(subset=["amount"], inplace=True)
        df = df[df["amount"] > 0].copy()

    df.dropna(how="all", inplace=True)

    for col in ["region", "direction", "status", "subsidy_name", "district"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["normative"] = pd.to_numeric(df["normative"], errors="coerce")
    df["amount"]    = pd.to_numeric(df["amount"],    errors="coerce")
    df["num"]       = pd.to_numeric(df["num"],       errors="coerce")

    print(f"    Загружено строк: {len(df):,}")
    print(f"    Колонки: {list(df.columns)}")
    return df

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Создаю базовые фичи из исходных данных...")

    df["hour_submitted"] = df["date_received"].dt.hour.fillna(12)

    df["day_of_week"] = df["date_received"].dt.dayofweek.fillna(1)

    df["month_submitted"] = df["date_received"].dt.month.fillna(1)

    df["livestock_count"] = (df["amount"] / df["normative"].replace(0, np.nan)).round(0)
    df["livestock_count"] = df["livestock_count"].clip(lower=1).fillna(10)

    df["log_amount"] = np.log1p(df["amount"])

    DIRECTION_MAP = {
        # RU
        "Субсидирование в скотоводстве":                          0,
        "Субсидирование в овцеводстве":                           1,
        "Субсидирование в коневодстве":                           2,
        "Субсидирование в птицеводстве":                          3,
        "Субсидирование в верблюдоводстве":                       4,
        "Субсидирование в свиноводстве":                          5,
        "Субсидирование в козоводстве":                           6,
        "Субсидирование в пчеловодстве":                          7,
        "Субсидирование затрат по искусственному осеменению":     8,
        # KZ
        "Мал шаруашылығын субсидиялау":                           0,
        "Қой шаруашылығын субсидиялау":                           1,
        "Жылқы шаруашылығын субсидиялау":                         2,
        "Құс шаруашылығын субсидиялау":                           3,
        "Түйе шаруашылығын субсидиялау":                          4,
        "Шошқа шаруашылығын субсидиялау":                         5,
        "Ешкі шаруашылығын субсидиялау":                          6,
        "Ара шаруашылығын субсидиялау":                           7,
        "Жасанды ұрықтандыру шығындарын субсидиялау":             8,
    }
    df["direction_code"] = df["direction"].map(DIRECTION_MAP).fillna(9)

    # Племенное — ru + kz
    df["is_pedigree"] = (
        df["subsidy_name"].str.contains("племен|тұқымдық", case=False, na=False)
    ).astype(int)

    # Производитель — ru + kz
    df["is_producer"] = (
        df["subsidy_name"].str.contains("производи|производит|өндіруші|өндіріс", case=False, na=False)
    ).astype(int)

    print(f"    Базовых фич добавлено: 8")
    return df

def generate_synthetic_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Генерирую синтетические экономические фичи...")
    n = len(df)

    base_growth = (df["log_amount"] - df["log_amount"].mean()) / df["log_amount"].std() * 0.1
    pedigree_bonus = df["is_pedigree"] * np.random.uniform(0.05, 0.15, n)
    noise_growth = np.random.normal(0, 0.12, n)
    df["gross_output_growth_yoy"] = (base_growth + pedigree_bonus + noise_growth).clip(-0.30, 0.80)

    direction_land_factor = df["direction_code"].map({
        0: 3.0,   # скотоводство
        1: 2.5,   # овцеводство
        2: 2.0,   # коневодство
        3: 0.3,   # птицеводство
        4: 4.0,   # верблюдоводство
        5: 0.5,   # свиноводство
        6: 2.5,   # козоводство
        7: 0.2,   # пчеловодство
        8: 1.5,   # искусственное осеменение
        9: 2.0,   # неизвестное направление
    }).fillna(2.0)
    df["land_to_livestock_ratio"] = (
        direction_land_factor * np.random.lognormal(0, 0.4, n)
    ).clip(0.2, 10.0)

    region_survival = {
        # RU
        "Мангистауская область":       0.82,
        "Атырауская область":          0.83,
        "Западно-Казахстанская область": 0.85,
        "Жамбылская область":          0.87,
        "Алматинская область":         0.90,
        "Акмолинская область":         0.88,
        "область Ұлытау":              0.86,
        "область Жетісу":              0.89,
        "г.Шымкент":                   0.87,
        # KZ
        "Маңғыстау облысы":            0.82,
        "Атырау облысы":               0.83,
        "Батыс Қазақстан облысы":      0.85,
        "Жамбыл облысы":               0.87,
        "Алматы облысы":               0.90,
        "Ақмола облысы":               0.88,
        "Ұлытау облысы":               0.86,
        "Жетісу облысы":               0.89,
        "Шымкент қ.":                  0.87,
    }
    base_survival = df["region"].map(region_survival).fillna(0.87)
    noise_survival = np.random.normal(0, 0.05, n)

    bird_bonus = (df["direction_code"] == 3) * 0.04
    df["historical_survival_rate"] = (
        base_survival + noise_survival + bird_bonus
    ).clip(0.50, 0.99)

    estimated_revenue = df["livestock_count"] * direction_land_factor * np.random.uniform(50000, 200000, n)
    raw_dependence = df["amount"] / (estimated_revenue + df["amount"])
    df["subsidy_dependence_index"] = raw_dependence.clip(0.0, 1.0)

    noise_vet = np.random.beta(8, 2, n)                                                             
    pedigree_compliance_bonus = df["is_pedigree"] * 0.05
    df["veterinary_compliance"] = (noise_vet + pedigree_compliance_bonus).clip(0.0, 1.0)

    df["years_in_operation"] = np.random.choice(
        range(1, 26),
        n,
        p=_generate_years_distribution(25),
    ).astype(float)

    df["pedigree_ratio"] = np.where(
        df["is_pedigree"] == 1,
        np.random.beta(5, 2, n),                                        
        np.random.beta(2, 5, n),                                       
    )

    df["previous_subsidies_count"] = np.random.poisson(lam=3.5, size=n).clip(0, 15)

    df["debt_load_ratio"] = np.random.lognormal(mean=0.3, sigma=0.7, size=n).clip(0.0, 5.0)

    # ── grazing_norm_deviation ──
    # Отклонение фактической нагрузки пастбищ от нормы для региона/направления
    # Источник: «Предельно допустимые нормы нагрузки на пастбища»
    grazing_norm = df.apply(
        lambda r: GRAZING_NORM_HA.get(
            (r["region"], int(r["direction_code"])),
            GRAZING_NORM_DEFAULT
        ),
        axis=1
    )
    grazing_actual = grazing_norm * np.random.lognormal(0, 0.3, n)
    df["grazing_norm_deviation"] = (
        (grazing_actual - grazing_norm) / grazing_norm
    ).clip(-2.0, 2.0)

    # ── natural_loss_risk_score ──
    # Риск аномальной смертности: фактическая смертность / гос. норматив
    # Источник: «Нормы естественной убыли (падежа)»
    # ИСПРАВЛЕНО: раньше было pure noise (normative сокращался)
    mortality_norm = df["direction_code"].map(MORTALITY_NORM).fillna(0.03)
    # Базовый риск зависит от ветеринарного соответствия и региона
    base_risk = (1.0 - df["veterinary_compliance"]) * 2.0  # чем хуже вет → тем выше риск
    region_risk = df["region"].map({
        # RU
        "Мангистауская область": 0.3,
        "Атырауская область": 0.25,
        "Кызылординская область": 0.2,
        # KZ
        "Маңғыстау облысы": 0.3,
        "Атырау облысы": 0.25,
        "Қызылорда облысы": 0.2,
    }).fillna(0.0)
    noise = np.random.lognormal(0, 0.2, n)
    actual_mortality_ratio = (base_risk + region_risk + 0.5) * noise
    df["natural_loss_risk_score"] = actual_mortality_ratio.clip(0.0, 3.0)

    print(f"    Синтетических экономических фич добавлено: 11")
    return df

def _generate_years_distribution(max_years: int):

    probs = np.array([1 / (1 + y * 0.3) for y in range(max_years)])
    return probs / probs.sum()                                 

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/5] Создаю целевую переменную (historical_score)...")

    def norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    debt_inverted     = 1 - norm(df["debt_load_ratio"])
    dependence_inv    = 1 - norm(df["subsidy_dependence_index"])

    # Новые фичи: grazing_norm_deviation (отрицательный = плохо)
    # и natural_loss_risk_score (меньше = лучше, инвертируем)
    grazing_inverted = norm(df["grazing_norm_deviation"] + 2.0)  # сдвиг в положительный диапазон
    risk_inverted    = 1 - norm(df["natural_loss_risk_score"])

    raw_score = (
        norm(df["gross_output_growth_yoy"])    * 22.0 +
        norm(df["pedigree_ratio"])              * 18.0 +
        norm(df["historical_survival_rate"])    * 13.0 +
        norm(df["veterinary_compliance"])       * 11.0 +
        norm(df["subsidy_dependence_index"].apply(lambda x: 1-x))  * 10.0 +
        debt_inverted                           *  9.0 +
        norm(df["land_to_livestock_ratio"])     *  4.0 +
        norm(df["years_in_operation"])          *  4.0 +
        grazing_inverted                        *  5.0 +
        risk_inverted                           *  4.0
    )

    raw_norm = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    df["historical_score"] = (raw_norm * 99 + 1).round(1)

    noise = np.random.normal(0, 3, len(df))
    df["historical_score"] = (df["historical_score"] + noise).clip(1, 100).round(1)

    print(f"    Score — среднее: {df['historical_score'].mean():.1f}")
    print(f"    Score — мин: {df['historical_score'].min():.1f}, макс: {df['historical_score'].max():.1f}")
    print(f"    Зелёных (80+): {(df['historical_score'] >= 80).sum():,}")
    print(f"    Жёлтых (50-79): {((df['historical_score'] >= 50) & (df['historical_score'] < 80)).sum():,}")
    print(f"    Красных (<50): {(df['historical_score'] < 50).sum():,}")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/5] Кодирую категориальные переменные...")

    le_region = LabelEncoder()
    le_direction = LabelEncoder()

    df["region_encoded"]    = le_region.fit_transform(df["region"].fillna("Неизвестно"))
    df["direction_encoded"] = le_direction.fit_transform(df["direction"].fillna("Неизвестно"))

    # Язык: 0 = ru, 1 = kz
    if "language" in df.columns:
        df["language_code"] = (df["language"] == "kz").astype(int)
        kz_count = df["language_code"].sum()
        ru_count = len(df) - kz_count
        print(f"    Язык: {ru_count:,} ru, {kz_count:,} kz")
    else:
        df["language_code"] = 0
        print(f"    ⚠️ Колонка 'language' не найдена, ставлю 0")

    region_mapping = dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))
    print(f"    Регионов закодировано: {len(region_mapping)}")
    print(f"    Направлений закодировано: {le_direction.classes_.tolist()}")

    return df

# ═══════════════════════════════════════════════════════
# СПРАВОЧНИКИ ИЗ НОРМАТИВНЫХ ДОКУМЕНТОВ МСХ РК
# ═══════════════════════════════════════════════════════

# Нормы нагрузки на пастбища (га/голову, восстановленные угодья)
# Источник: «Предельно допустимые нормы нагрузки на пастбища»
# Ключ: (регион, direction_code), Значение: среднее из диапазона
GRAZING_NORM_HA = {
    # ═══════════════════════════════════════════
    # RU — Акмолинская область
    # ═══════════════════════════════════════════
    ("Акмолинская область", 0): 8.25,
    ("Акмолинская область", 1): 1.65,
    ("Акмолинская область", 2): 9.90,
    ("Акмолинская область", 4): 11.55,
    # KZ — Ақмола облысы
    ("Ақмола облысы", 0): 8.25,
    ("Ақмола облысы", 1): 1.65,
    ("Ақмола облысы", 2): 9.90,
    ("Ақмола облысы", 4): 11.55,

    # RU — Мангистауская область
    ("Мангистауская область", 0): 13.00,
    ("Мангистауская область", 1): 2.60,
    ("Мангистауская область", 2): 15.60,
    ("Мангистауская область", 4): 20.40,
    # KZ — Маңғыстау облысы
    ("Маңғыстау облысы", 0): 13.00,
    ("Маңғыстау облысы", 1): 2.60,
    ("Маңғыстау облысы", 2): 15.60,
    ("Маңғыстау облысы", 4): 20.40,

    # RU — Алматинская область
    ("Алматинская область", 0): 11.25,
    ("Алматинская область", 1): 2.25,
    ("Алматинская область", 2): 13.50,
    ("Алматинская область", 4): 13.70,
    # KZ — Алматы облысы
    ("Алматы облысы", 0): 11.25,
    ("Алматы облысы", 1): 2.25,
    ("Алматы облысы", 2): 13.50,
    ("Алматы облысы", 4): 13.70,

    # RU — Актюбинская область
    ("Актюбинская область", 0): 11.75,
    ("Актюбинская область", 1): 2.35,
    ("Актюбинская область", 2): 14.10,
    ("Актюбинская область", 4): 16.45,
    # KZ — Ақтөбе облысы
    ("Ақтөбе облысы", 0): 11.75,
    ("Ақтөбе облысы", 1): 2.35,
    ("Ақтөбе облысы", 2): 14.10,
    ("Ақтөбе облысы", 4): 16.45,

    # RU — Атырауская область
    ("Атырауская область", 0): 12.00,
    ("Атырауская область", 1): 2.50,
    ("Атырауская область", 2): 14.40,
    ("Атырауская область", 4): 18.00,
    # KZ — Атырау облысы
    ("Атырау облысы", 0): 12.00,
    ("Атырау облысы", 1): 2.50,
    ("Атырау облысы", 2): 14.40,
    ("Атырау облысы", 4): 18.00,

    # RU — Западно-Казахстанская область
    ("Западно-Казахстанская область", 0): 7.50,
    ("Западно-Казахстанская область", 1): 1.50,
    ("Западно-Казахстанская область", 2): 9.00,
    ("Западно-Казахстанская область", 4): 10.50,
    # KZ — Батыс Қазақстан облысы
    ("Батыс Қазақстан облысы", 0): 7.50,
    ("Батыс Қазақстан облысы", 1): 1.50,
    ("Батыс Қазақстан облысы", 2): 9.00,
    ("Батыс Қазақстан облысы", 4): 10.50,

    # RU — Восточно-Казахстанская область
    ("Восточно-Казахстанская область", 0): 9.00,
    ("Восточно-Казахстанская область", 1): 1.80,
    ("Восточно-Казахстанская область", 2): 10.80,
    ("Восточно-Казахстанская область", 4): 12.60,
    # KZ — Шығыс Қазақстан облысы
    ("Шығыс Қазақстан облысы", 0): 9.00,
    ("Шығыс Қазақстан облысы", 1): 1.80,
    ("Шығыс Қазақстан облысы", 2): 10.80,
    ("Шығыс Қазақстан облысы", 4): 12.60,

    # RU — Карагандинская область
    ("Карагандинская область", 0): 10.00,
    ("Карагандинская область", 1): 2.00,
    ("Карагандинская область", 2): 12.00,
    ("Карагандинская область", 4): 14.00,
    # KZ — Қарағанды облысы
    ("Қарағанды облысы", 0): 10.00,
    ("Қарағанды облысы", 1): 2.00,
    ("Қарағанды облысы", 2): 12.00,
    ("Қарағанды облысы", 4): 14.00,

    # RU — Костанайская область
    ("Костанайская область", 0): 7.00,
    ("Костанайская область", 1): 1.40,
    ("Костанайская область", 2): 8.40,
    ("Костанайская область", 4): 9.80,
    # KZ — Қостанай облысы
    ("Қостанай облысы", 0): 7.00,
    ("Қостанай облысы", 1): 1.40,
    ("Қостанай облысы", 2): 8.40,
    ("Қостанай облысы", 4): 9.80,

    # RU — Кызылординская область
    ("Кызылординская область", 0): 11.00,
    ("Кызылординская область", 1): 2.20,
    ("Кызылординская область", 2): 13.20,
    ("Кызылординская область", 4): 15.40,
    # KZ — Қызылорда облысы
    ("Қызылорда облысы", 0): 11.00,
    ("Қызылорда облысы", 1): 2.20,
    ("Қызылорда облысы", 2): 13.20,
    ("Қызылорда облысы", 4): 15.40,

    # RU — Павлодарская область
    ("Павлодарская область", 0): 8.50,
    ("Павлодарская область", 1): 1.70,
    ("Павлодарская область", 2): 10.20,
    ("Павлодарская область", 4): 11.90,
    # KZ — Павлодар облысы
    ("Павлодар облысы", 0): 8.50,
    ("Павлодар облысы", 1): 1.70,
    ("Павлодар облысы", 2): 10.20,
    ("Павлодар облысы", 4): 11.90,

    # RU — Северо-Казахстанская область
    ("Северо-Казахстанская область", 0): 6.50,
    ("Северо-Казахстанская область", 1): 1.30,
    ("Северо-Казахстанская область", 2): 7.80,
    ("Северо-Казахстанская область", 4): 9.10,
    # KZ — Солтүстік Қазақстан облысы
    ("Солтүстік Қазақстан облысы", 0): 6.50,
    ("Солтүстік Қазақстан облысы", 1): 1.30,
    ("Солтүстік Қазақстан облысы", 2): 7.80,
    ("Солтүстік Қазақстан облысы", 4): 9.10,

    # RU — Туркестанская область
    ("Туркестанская область", 0): 9.50,
    ("Туркестанская область", 1): 1.90,
    ("Туркестанская область", 2): 11.40,
    ("Туркестанская область", 4): 13.30,
    # KZ — Түркістан облысы
    ("Түркістан облысы", 0): 9.50,
    ("Түркістан облысы", 1): 1.90,
    ("Түркістан облысы", 2): 11.40,
    ("Түркістан облысы", 4): 13.30,

    # RU — Жамбылская область
    ("Жамбылская область", 0): 10.50,
    ("Жамбылская область", 1): 2.10,
    ("Жамбылская область", 2): 12.60,
    ("Жамбылская область", 4): 14.70,
    # KZ — Жамбыл облысы
    ("Жамбыл облысы", 0): 10.50,
    ("Жамбыл облысы", 1): 2.10,
    ("Жамбыл облысы", 2): 12.60,
    ("Жамбыл облысы", 4): 14.70,

    # RU — область Абай
    ("область Абай", 0): 9.00,
    ("область Абай", 1): 1.80,
    ("область Абай", 2): 10.80,
    ("область Абай", 4): 12.60,
    # KZ — Абай облысы
    ("Абай облысы", 0): 9.00,
    ("Абай облысы", 1): 1.80,
    ("Абай облысы", 2): 10.80,
    ("Абай облысы", 4): 12.60,

    # RU — область Ұлытау
    ("область Ұлытау", 0): 10.00,
    ("область Ұлытау", 1): 2.00,
    ("область Ұлытау", 2): 12.00,
    ("область Ұлытау", 4): 14.00,
    # KZ — Ұлытау облысы
    ("Ұлытау облысы", 0): 10.00,
    ("Ұлытау облысы", 1): 2.00,
    ("Ұлытау облысы", 2): 12.00,
    ("Ұлытау облысы", 4): 14.00,

    # RU — область Жетісу
    ("область Жетісу", 0): 10.50,
    ("область Жетісу", 1): 2.10,
    ("область Жетісу", 2): 12.60,
    ("область Жетісу", 4): 14.70,
    # KZ — Жетісу облысы
    ("Жетісу облысы", 0): 10.50,
    ("Жетісу облысы", 1): 2.10,
    ("Жетісу облысы", 2): 12.60,
    ("Жетісу облысы", 4): 14.70,

    # RU — г.Шымкент
    ("г.Шымкент", 0): 9.50,
    ("г.Шымкент", 1): 1.90,
    ("г.Шымкент", 2): 11.40,
    ("г.Шымкент", 4): 13.30,
    # KZ — Шымкент қ.
    ("Шымкент қ.", 0): 9.50,
    ("Шымкент қ.", 1): 1.90,
    ("Шымкент қ.", 2): 11.40,
    ("Шымкент қ.", 4): 13.30,
}

# Дефолтная норма если регион/направление не найдены (га/голову)
GRAZING_NORM_DEFAULT = 5.0

# Нормы естественной убыли (падежа) — годовая доля
# Источник: «Нормы естественной убыли (падежа)»
# Ключ: direction_code, Значение: доля (0.025 = 2.5%)
MORTALITY_NORM = {
    0: 0.025,   # КРС: среднее (мясной 2%, молочный 3%)
    1: 0.030,   # Овцы: 3%
    2: 0.027,   # Лошади (табунные, молодняк): 2.7%
    3: 0.075,   # Птица (мясная+яичная): 7.5%
    4: 0.022,   # Верблюды (среднее: самки 1.7%, самцы 2.7%)
    5: 0.010,   # Свиньи (на откорме): 1%
    6: 0.028,   # Козы: ~2.8%
    7: 0.050,   # Пчёлы: ~5% (зимовка)
    8: 0.025,   # Искусственное осеменение — как КРС
    9: 0.030,   # Неизвестное направление — дефолт
}

ML_FEATURES = [

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

    "log_amount",
    "livestock_count",
    "direction_code",
    "is_pedigree",
    "is_producer",
    "hour_submitted",
    "month_submitted",
    "region_encoded",
    "language_code",
]

TARGET = "historical_score"

def main():
    print("=" * 65)
    print("  SmartAgro Score | Feature Engineering Pipeline")
    print("=" * 65)

    df = load_and_clean(INPUT_FILE)

    # data_combined.csv уже содержит все фичи — пропускаем engineer_base_features
    is_combined = "data_combined" in INPUT_FILE
    if is_combined:
        print("\n[2/5] Датасет уже содержит фичи — пропускаю engineer_base_features")
    else:
        df = engineer_base_features(df)

    if is_combined:
        print("\n[3/5] Датасет уже содержит синтетические фичи — пропускаю")
    else:
        df = generate_synthetic_economic_features(df)

    if is_combined:
        print("\n[4/5] Датасет уже содержит historical_score — пропускаю")
    else:
        df = create_target_variable(df)

    df = encode_categorical(df)

    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
    print(f"\n✅ Сохранён: {OUTPUT_CLEAN} ({len(df):,} строк, {df.shape[1]} колонок)")

    df_ml = df[ML_FEATURES + [TARGET]].dropna()
    df_ml.to_csv(OUTPUT_FEAT, index=False, encoding="utf-8-sig")
    print(f"✅ Сохранён: {OUTPUT_FEAT} ({len(df_ml):,} строк, {len(ML_FEATURES)} фичей)")

    print("\n📊 Статистика датасета для ML:")
    print(df_ml[ML_FEATURES].describe().round(3).to_string())

    print("\n🎯 Распределение целевой переменной:")
    print(f"   Зелёная зона (80-100): {(df_ml[TARGET] >= 80).mean()*100:.1f}%")
    print(f"   Жёлтая зона  (50-79):  {((df_ml[TARGET] >= 50) & (df_ml[TARGET] < 80)).mean()*100:.1f}%")
    print(f"   Красная зона (1-49):   {(df_ml[TARGET] < 50).mean()*100:.1f}%")

    if is_combined:
        print("\n🌐 Распределение по языку:")
        if "language" in df_ml.columns:
            print(f"   RU: {(df_ml['language_code'] == 0).sum():,}")
            print(f"   KZ: {(df_ml['language_code'] == 1).sum():,}")

    print("\n✨ Готово! Следующий шаг: запусти train_model.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
