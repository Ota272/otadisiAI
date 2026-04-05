"""
Перегенерировать SHAP explainer из текущей модели.
Запускать когда model и explainer рассинхронизированы.
"""
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

MODELS_DIR = Path("models")

def main():
    print("🔄 Перегенерация SHAP explainer...")

    # Загружаем модель и scaler
    model = joblib.load(MODELS_DIR / "xgb_scorer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "feature_names.json", encoding="utf-8") as f:
        import json
        feature_names = json.load(f)

    print(f"   Модель: {model.__class__.__name__}")
    print(f"   Фичей: {len(feature_names)}")

    # Генерируем синтетические данные для обучения explainer
    # (нужно ~100-200 семплов для стабильного background)
    n_samples = 200
    np.random.seed(42)

    # Создаём реалистичные данные из медиан и std
    medians = {
        "gross_output_growth_yoy": 0.05,
        "land_to_livestock_ratio": 6.0,
        "historical_survival_rate": 0.88,
        "subsidy_dependence_index": 0.25,
        "veterinary_compliance": 0.80,
        "years_in_operation": 8.0,
        "pedigree_ratio": 0.35,
        "previous_subsidies_count": 3.0,
        "debt_load_ratio": 1.0,
        "grazing_norm_deviation": 0.0,
        "natural_loss_risk_score": 1.0,
        "log_amount": 14.0,
        "livestock_count": 50.0,
        "direction_code": 0.0,
        "is_pedigree": 0.0,
        "is_producer": 0.0,
        "hour_submitted": 12.0,
        "month_submitted": 6.0,
        "region_encoded": 7.0,
    }

    background_data = {}
    for feat in feature_names:
        med = medians.get(feat, 0.0)
        if feat in ["is_pedigree", "is_producer", "direction_code", "region_encoded"]:
            background_data[feat] = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples)
        elif feat in ["previous_subsidies_count"]:
            background_data[feat] = np.random.poisson(med, n_samples)
        elif feat in ["years_in_operation"]:
            background_data[feat] = np.random.uniform(1, 25, n_samples)
        else:
            background_data[feat] = np.random.normal(med, med * 0.5 if med > 0 else 0.5, n_samples)

    X_background = pd.DataFrame(background_data, columns=feature_names)
    X_scaled = scaler.transform(X_background)

    print(f"   Генерирую background: {n_samples} семплов...")

    # Создаём новый explainer
    print("   Создаю TreeExplainer (это займёт 10-30 сек)...")
    explainer = shap.TreeExplainer(model)

    # Проверка
    test_sample = X_scaled[:1]
    model_pred = model.predict(test_sample)[0]
    shap_base = explainer.expected_value
    shap_vals = explainer.shap_values(test_sample)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]
    shap_pred = float(shap_base) + sum(shap_vals)
    diff = abs(model_pred - shap_pred)

    print(f"\n   Model pred: {model_pred:.2f}")
    print(f"   SHAP pred:  {shap_pred:.2f}")
    print(f"   Разница:    {diff:.4f}")

    if diff < 0.01:
        print("   ✅ SHAP сходится идеально!")
    elif diff < 1.0:
        print("   ✅ SHAP сходится хорошо")
    else:
        print("   ⚠️ Разница большая — проверь данные")

    # Сохраняем
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    print(f"\n💾 Сохранён: {MODELS_DIR}/shap_explainer.joblib")
    print("✅ Готово! Теперь перезапусти приложение.")

if __name__ == "__main__":
    main()
