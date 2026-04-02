
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

INPUT_FILE  = "data.csv"
OUTPUT_CLEAN = "data_cleaned.csv"
OUTPUT_FEAT  = "data_features.csv"

def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/5] Загружаю файл: {path}")

    df = pd.read_csv(path, skiprows=3, header=0, low_memory=False)

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

    df.dropna(how="all", inplace=True)

    for col in ["region", "direction", "status", "subsidy_name", "district"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["normative"] = pd.to_numeric(df["normative"], errors="coerce")
    df["amount"]    = pd.to_numeric(df["amount"],    errors="coerce")
    df["num"]       = pd.to_numeric(df["num"],       errors="coerce")

    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", dayfirst=True)

    df.drop(columns=["col3", "col4"], inplace=True, errors="ignore")

    df = df[df["amount"].notna() & (df["amount"] > 0)].copy()

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
        "Субсидирование в скотоводстве":     0,
        "Субсидирование в овцеводстве":       1,
        "Субсидирование в коневодстве":       2,
        "Субсидирование в птицеводстве":      3,
        "Субсидирование в верблюдоводстве":   4,
        "Субсидирование в свиноводстве":      5,
    }
    df["direction_code"] = df["direction"].map(DIRECTION_MAP).fillna(6)

    df["is_pedigree"] = (
        df["subsidy_name"].str.contains("племен", case=False, na=False)
    ).astype(int)                       

    df["is_producer"] = (
        df["subsidy_name"].str.contains("производи|производит", case=False, na=False)
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
        0: 3.0,                              
        1: 2.5,               
        2: 2.0,               
        3: 0.3,                                              
        4: 4.0,                                            
        5: 0.5,                
        6: 2.0,          
    }).fillna(2.0)
    df["land_to_livestock_ratio"] = (
        direction_land_factor * np.random.lognormal(0, 0.4, n)
    ).clip(0.2, 10.0)

    region_survival = {
        "Мангистауская область":       0.82,                 
        "Атырауская область":          0.83,
        "Западно-Казахстанская область": 0.85,
        "Жамбылская область":          0.87,
        "Алматинская область":         0.90,
        "Акмолинская область":         0.88,
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

    print(f"    Синтетических экономических фич добавлено: 9")
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

    raw_score = (
        norm(df["gross_output_growth_yoy"])    * 25.0 +                                     
        norm(df["pedigree_ratio"])              * 20.0 +                      
        norm(df["historical_survival_rate"])    * 15.0 +                     
        norm(df["veterinary_compliance"])       * 13.0 +               
        norm(df["subsidy_dependence_index"].apply(lambda x: 1-x))  * 12.0 +                     
        debt_inverted                           * 10.0 +                           
        norm(df["land_to_livestock_ratio"])     *  5.0 +                     
        norm(df["years_in_operation"])          *  5.0          
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

    region_mapping = dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))
    print(f"    Регионов закодировано: {len(region_mapping)}")
    print(f"    Направлений закодировано: {le_direction.classes_.tolist()}")

    return df

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

def main():
    print("=" * 65)
    print("  SmartAgro Score | Feature Engineering Pipeline")
    print("=" * 65)

    df = load_and_clean(INPUT_FILE)

    df = engineer_base_features(df)

    df = generate_synthetic_economic_features(df)

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

    print("\n✨ Готово! Следующий шаг: запусти train_model.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
