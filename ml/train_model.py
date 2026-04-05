
import json
import warnings
from pathlib import Path

import joblib                                                       
import matplotlib
matplotlib.use("Agg")                                               
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap                                                         
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")                                          

RANDOM_SEED  = 42
INPUT_FILE   = str(Path(__file__).parent.parent / "data" / "data_features_combined.csv")
MODELS_DIR   = Path(__file__).parent.parent / "models"
REPORTS_DIR  = Path(__file__).parent.parent / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

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

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    print(f"[1/6] Загружаю датасет: {path}")

    df = pd.read_csv(path)
    print(f"    Загружено: {len(df):,} строк, {df.shape[1]} колонок")

    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Отсутствуют фичи: {missing}\nЗапустите сначала data_prep.py!")

    X = df[ML_FEATURES].copy()
    y = df[TARGET].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    print(f"    После очистки: {len(X):,} строк")
    print(f"    Фичей: {len(ML_FEATURES)}")
    print(f"    Таргет — мин: {y.min():.1f}, макс: {y.max():.1f}, среднее: {y.mean():.1f}")

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    print("\n[2/6] Делю данные на train/test (80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,                            
        random_state=RANDOM_SEED,                               
    )

    print(f"    Train: {len(X_train):,} строк ({len(X_train)/len(X)*100:.0f}%)")
    print(f"    Test:  {len(X_test):,}  строк ({len(X_test)/len(X)*100:.0f}%)")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test) -> tuple:
    print("\n[3/6] Нормализую данные (StandardScaler)...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=ML_FEATURES, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=ML_FEATURES, index=X_test.index)

    print(f"    Среднее после скейлинга (expect ~0): {X_train_scaled.mean().mean():.4f}")
    print(f"    Std после скейлинга (expect ~1):     {X_train_scaled.std().mean():.4f}")

    return X_train_scaled, X_test_scaled, scaler

def train_xgboost(X_train, y_train, X_test, y_test) -> XGBRegressor:
    print("\n[4/6] Обучаю XGBoost модель...")
    print("    Это может занять 30-60 секунд...")

    model = XGBRegressor(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.7,

        reg_lambda=1.0,                               
        reg_alpha=0.1,                                                  

        min_child_weight=5,

        early_stopping_rounds=50,

        random_state=RANDOM_SEED,
        n_jobs=-1,                                  
        verbosity=0,                                                 
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print(f"\n    Обучено деревьев: {model.best_iteration + 1} (из {500})")
    print(f"    Лучший тест MAE: {model.best_score:.3f}")

    return model

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    print("\n[5/6] Оцениваю качество модели...")

    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_pred_test  = np.clip(y_pred_test,  1, 100)
    y_pred_train = np.clip(y_pred_train, 1, 100)

    metrics = {
        "train": {
            "MAE":  round(mean_absolute_error(y_train, y_pred_train), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 3),
            "R2":   round(r2_score(y_train, y_pred_train), 4),
        },
        "test": {
            "MAE":  round(mean_absolute_error(y_test, y_pred_test), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 3),
            "R2":   round(r2_score(y_test, y_pred_test), 4),
        },
    }

    print("    Запускаю 5-fold Cross-Validation (это займёт минуту)...")
    cv_scores = cross_val_score(
        XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        ),
        X_train, y_train,
        cv=5,                                         
        scoring="neg_mean_absolute_error",                                     
        n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()                      
    metrics["cross_val_mae"] = round(cv_mae, 3)

    df_errors = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test})
    df_errors["zone"] = pd.cut(df_errors["y_true"], bins=[0, 50, 80, 100],
                                labels=["red", "yellow", "green"])
    zone_errors = df_errors.groupby("zone", observed=True).apply(
        lambda g: round(mean_absolute_error(g["y_true"], g["y_pred"]), 2)
    ).to_dict()
    metrics["mae_by_zone"] = zone_errors

    print("\n    ┌─────────────────────────────────────────┐")
    print("    │           МЕТРИКИ КАЧЕСТВА МОДЕЛИ       │")
    print("    ├────────────────┬────────────┬───────────┤")
    print("    │ Метрика        │ Train      │ Test      │")
    print("    ├────────────────┼────────────┼───────────┤")
    print(f"    │ MAE (баллы)    │ {metrics['train']['MAE']:<10} │ {metrics['test']['MAE']:<9} │")
    print(f"    │ RMSE (баллы)   │ {metrics['train']['RMSE']:<10} │ {metrics['test']['RMSE']:<9} │")
    print(f"    │ R² (0-1)       │ {metrics['train']['R2']:<10} │ {metrics['test']['R2']:<9} │")
    print("    ├────────────────┴────────────┴───────────┤")
    print(f"    │ Cross-Val MAE (5-fold): {cv_mae:.3f}           │")
    print("    ├─────────────────────────────────────────┤")
    print("    │ MAE по зонам:                           │")
    for zone, mae in zone_errors.items():
        print(f"    │   {zone:<10}: {mae} баллов{' ' * (20 - len(str(mae)))}│")
    print("    └─────────────────────────────────────────┘")

    overfit_gap = abs(metrics["train"]["MAE"] - metrics["test"]["MAE"])
    if overfit_gap < 1.0:
        print("\n    ✅ Переобучения НЕТ (Train MAE ≈ Test MAE)")
    elif overfit_gap < 3.0:
        print(f"\n    ⚠️  Небольшое переобучение (разрыв: {overfit_gap:.1f} балла)")
    else:
        print(f"\n    ❌ Переобучение! Разрыв Train/Test MAE: {overfit_gap:.1f} балла")
        print("       Попробуй: уменьшить n_estimators или max_depth")

    return metrics

def plot_feature_importance(model: XGBRegressor):
    print("\n    Строю график важности фичей...")

    importances = pd.Series(
        model.feature_importances_,
        index=ML_FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d32f2f" if imp < importances.median() else "#1976d2" for imp in importances]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Важность фичей XGBoost (Feature Importance)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Важность (чем больше, тем важнее фича для модели)")
    ax.axvline(importances.median(), color="orange", linestyle="--", alpha=0.7, label="Медиана")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Сохранён: {REPORTS_DIR}/feature_importance.png")

def create_shap_explainer(model: XGBRegressor, X_train: pd.DataFrame) -> shap.TreeExplainer:
    print("\n    Создаю SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("    SHAP Explainer готов!")
    return explainer

def save_artifacts(model, scaler, explainer, metrics: dict):
    print("\n[6/6] Сохраняю артефакты модели...")

    joblib.dump(model,     MODELS_DIR / "xgb_scorer.joblib")
    joblib.dump(scaler,    MODELS_DIR / "scaler.joblib")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(ML_FEATURES, f, ensure_ascii=False, indent=2)

    report_text = f"""
SmartAgro Score — Model Training Report
========================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: XGBoost Regressor
Features: {len(ML_FEATURES)}

METRICS:
  Train MAE:  {metrics['train']['MAE']} баллов
  Train RMSE: {metrics['train']['RMSE']} баллов
  Train R²:   {metrics['train']['R2']}

  Test MAE:   {metrics['test']['MAE']} баллов
  Test RMSE:  {metrics['test']['RMSE']} баллов
  Test R²:    {metrics['test']['R2']}

  Cross-Val MAE (5-fold): {metrics['cross_val_mae']} баллов

  MAE по зонам: {metrics['mae_by_zone']}

FEATURE LIST:
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(ML_FEATURES))}
"""
    with open(REPORTS_DIR / "model_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"    ✅ {MODELS_DIR}/xgb_scorer.joblib")
    print(f"    ✅ {MODELS_DIR}/scaler.joblib")
    print(f"    ✅ {MODELS_DIR}/shap_explainer.joblib")
    print(f"    ✅ {MODELS_DIR}/feature_names.json")
    print(f"    ✅ {REPORTS_DIR}/model_report.txt")
    print(f"    ✅ {REPORTS_DIR}/feature_importance.png")

def demo_single_prediction(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    print("\n" + "=" * 55)
    print("  ДЕМО: Предсказание для одного фермера")
    print("=" * 55)

    one_farmer = X_test.iloc[[0]]
    true_score = y_test.iloc[0]

    one_farmer_scaled = scaler.transform(one_farmer)

    predicted_score = float(np.clip(model.predict(one_farmer_scaled)[0], 1, 100))

    print(f"\n  Реальный балл:     {true_score:.1f}")
    print(f"  Предсказанный:     {predicted_score:.1f}")
    print(f"  Ошибка:            {abs(predicted_score - true_score):.1f} баллов")

    if predicted_score >= 80:
        zone = "🟢 GREEN — Строго рекомендовано"
    elif predicted_score >= 50:
        zone = "🟡 YELLOW — Требует рассмотрения"
    else:
        zone = "🔴 RED — Не рекомендовано"

    print(f"  Зона:              {zone}")

    print("\n  Данные этого фермера:")
    for feat, val in one_farmer.iloc[0].items():
        print(f"    {feat}: {val:.4f}")

def main():
    print("=" * 65)
    print("  SmartAgro Score | Model Training Pipeline (XGBoost)")
    print("=" * 65)

    X, y = load_data(INPUT_FILE)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)

    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

    plot_feature_importance(model)

    explainer = create_shap_explainer(model, X_train_scaled)

    save_artifacts(model, scaler, explainer, metrics)

    demo_single_prediction(model, scaler, X_test, y_test)

    print("\n" + "=" * 65)
    print("  ✨ Обучение завершено!")
    print("  Следующий шаг: изучи shap_integration.py и main.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
