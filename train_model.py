"""
╔══════════════════════════════════════════════════════════════════╗
║   SmartAgro Score — ШАГ 2: Обучение ML-модели                   ║
║   train_model.py                                                  ║
║   Decentrathon 5.0 | AI for Government                          ║
╚══════════════════════════════════════════════════════════════════╝

КАК ЗАПУСКАТЬ (после data_prep.py):
    python train_model.py

ЧТО НА ВЫХОДЕ:
    models/xgb_scorer.joblib   — обученная XGBoost-модель
    models/scaler.joblib        — нормализатор данных
    models/feature_names.json   — список фичей в правильном порядке
    models/shap_explainer.joblib — SHAP-объяснитель для XAI
    reports/model_report.txt    — отчёт о качестве модели
"""

# ──────────────────────────────────────────────────────────────────
# БЛОК 1: ИМПОРТ БИБЛИОТЕК
# ──────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path

import joblib          # Сохранение объектов Python в файл (.joblib)
import matplotlib
matplotlib.use("Agg")  # Без GUI (запускаем на сервере без монитора)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap            # SHAP — библиотека для объяснения ML-моделей
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")  # Скрываем несущественные предупреждения

# ──────────────────────────────────────────────────────────────────
# БЛОК 2: КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────

RANDOM_SEED  = 42
INPUT_FILE   = "data_features.csv"          # Результат data_prep.py
MODELS_DIR   = Path("models")               # Папка для сохранения модели
REPORTS_DIR  = Path("reports")             # Папка для отчётов

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Список фичей — должен совпадать с data_prep.py!
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


# ──────────────────────────────────────────────────────────────────
# БЛОК 3: ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Загружает датасет и разделяет на X (фичи) и y (таргет).

    X — матрица признаков (входные данные для модели)
    y — вектор целевых значений (что мы хотим предсказать)
    """
    print(f"[1/6] Загружаю датасет: {path}")

    df = pd.read_csv(path)
    print(f"    Загружено: {len(df):,} строк, {df.shape[1]} колонок")

    # Проверяем что все нужные фичи присутствуют
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Отсутствуют фичи: {missing}\nЗапустите сначала data_prep.py!")

    X = df[ML_FEATURES].copy()
    y = df[TARGET].copy()

    # Удаляем строки с NaN (пропущенными значениями)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    print(f"    После очистки: {len(X):,} строк")
    print(f"    Фичей: {len(ML_FEATURES)}")
    print(f"    Таргет — мин: {y.min():.1f}, макс: {y.max():.1f}, среднее: {y.mean():.1f}")

    return X, y


# ──────────────────────────────────────────────────────────────────
# БЛОК 4: РАЗДЕЛЕНИЕ НА TRAIN / TEST
# ──────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Разделяем данные на обучающую (train) и тестовую (test) выборки.

    ЗАЧЕМ РАЗДЕЛЯТЬ?
    Если мы обучим модель на ВСЕХ данных и потом проверим её на них же —
    это нечестно! Модель просто "запомнит" правильные ответы.
    Нам важно знать, как она работает на НОВЫХ, невиданных данных.

    Аналогия: в школе нельзя дать ученику задачи ЕГЭ за неделю до экзамена
    и потом удивляться высоким оценкам. Тест должен быть новым!

    80/20: 80% данных → обучение, 20% → проверка качества
    """
    print("\n[2/6] Делю данные на train/test (80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,          # 20% уйдёт в тест
        random_state=RANDOM_SEED, # Воспроизводимость разделения
    )

    print(f"    Train: {len(X_train):,} строк ({len(X_train)/len(X)*100:.0f}%)")
    print(f"    Test:  {len(X_test):,}  строк ({len(X_test)/len(X)*100:.0f}%)")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────
# БЛОК 5: НОРМАЛИЗАЦИЯ ДАННЫХ
# ──────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test) -> tuple:
    """
    Нормализуем фичи с помощью StandardScaler.

    ЗАЧЕМ НОРМАЛИЗОВАТЬ?
    Представь: у нас есть фича "livestock_count" (от 1 до 1000 голов)
    и фича "pedigree_ratio" (от 0.0 до 1.0).

    Без нормализации модель думает: "livestock_count меняется на 999 единиц,
    а pedigree_ratio — всего на 1 единицу. Значит первая фича важнее!"
    Но это неправда — масштаб не равен важности!

    StandardScaler делает:
    x_scaled = (x - mean) / std
    После этого все фичи имеют среднее = 0 и отклонение = 1.

    ВАЖНОЕ ПРАВИЛО: fit только на train, transform на обоих!
    Почему? Если мы посчитаем mean/std по тестовым данным, модель
    "увидит" информацию о будущем → это называется data leakage (утечка данных)
    и делает оценку модели нечестной.

    ПРИМЕЧАНИЕ: XGBoost не очень чувствителен к масштабу, но нам
    нужен скейлер для FastAPI (чтобы масштабировать входящие данные одинаково).
    """
    print("\n[3/6] Нормализую данные (StandardScaler)...")

    scaler = StandardScaler()

    # fit_transform: считаем mean/std по train И масштабируем train
    X_train_scaled = scaler.fit_transform(X_train)

    # transform: масштабируем test ИСПОЛЬЗУЯ параметры от train
    X_test_scaled  = scaler.transform(X_test)

    # Возвращаем как DataFrame (сохраняем имена колонок для SHAP)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=ML_FEATURES, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=ML_FEATURES, index=X_test.index)

    print(f"    Среднее после скейлинга (expect ~0): {X_train_scaled.mean().mean():.4f}")
    print(f"    Std после скейлинга (expect ~1):     {X_train_scaled.std().mean():.4f}")

    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────────────
# БЛОК 6: ОБУЧЕНИЕ XGBOOST
# ──────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test) -> XGBRegressor:
    """
    Обучаем XGBoost — один из лучших алгоритмов для табличных данных.

    ЧТО ТАКОЕ XGBOOST?
    XGBoost = eXtreme Gradient Boosting.
    Это ансамблевый метод: он строит много маленьких деревьев решений
    последовательно. Каждое следующее дерево исправляет ошибки предыдущего.

    Аналогия: представь команду врачей. Первый ставит диагноз.
    Второй смотрит где первый ошибся и добавляет коррекцию.
    Третий исправляет ошибки двух предыдущих. И так 300 раз.
    Итого: очень точный "комитет врачей"!

    ПОЧЕМУ XGBoost для нашей задачи?
    1. Работает с табличными данными лучше нейросетей
    2. Устойчив к выбросам
    3. Встроенная важность фичей
    4. SHAP работает идеально с деревьями!
    5. Быстро обучается (параллельные вычисления)

    ГИПЕРПАРАМЕТРЫ (настройки модели):
    """
    print("\n[4/6] Обучаю XGBoost модель...")
    print("    Это может занять 30-60 секунд...")

    model = XGBRegressor(
        # ── Количество деревьев ───────────────────────────────────
        # Больше деревьев = точнее, но медленнее и риск переобучения
        n_estimators=500,

        # ── Максимальная глубина каждого дерева ──────────────────
        # Глубина 6: дерево принимает до 6 последовательных решений
        # Глубокие деревья = сложнее, риск переобучения
        max_depth=6,

        # ── Скорость обучения (learning rate) ─────────────────────
        # Насколько сильно каждое дерево корректирует предыдущее.
        # Маленький lr = нужно больше деревьев, но точнее
        learning_rate=0.05,

        # ── Регуляризация (борьба с переобучением) ───────────────
        # subsample: каждое дерево обучается на 80% случайных строк
        # colsample_bytree: каждое дерево видит 70% случайных фичей
        # Это снижает корреляцию между деревьями → лучше обобщение
        subsample=0.8,
        colsample_bytree=0.7,

        # ── Lambda и alpha: L2 и L1 регуляризация ────────────────
        # Наказывают модель за слишком большие веса → менее переобученная модель
        reg_lambda=1.0,  # L2: штраф за квадраты весов
        reg_alpha=0.1,   # L1: штраф за сами веса (делает некоторые = 0)

        # ── Минимальное количество данных для разветвления ────────
        # Ветка не делается, если в ней меньше N наблюдений → меньше шум
        min_child_weight=5,

        # ── Early stopping: остановка при переобучении ───────────
        # Если за последние 50 деревьев нет улучшения на тесте → стоп
        early_stopping_rounds=50,

        # ── Техническое ──────────────────────────────────────────
        random_state=RANDOM_SEED,
        n_jobs=-1,       # Использовать все CPU ядра
        verbosity=0,     # Не выводить лог XGBoost (мы сами печатаем)
    )

    # fit — обучение модели
    # eval_set: дополнительно передаём тестовые данные для early stopping
    # verbose=50: выводить прогресс каждые 50 деревьев
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print(f"\n    Обучено деревьев: {model.best_iteration + 1} (из {500})")
    print(f"    Лучший тест MAE: {model.best_score:.3f}")

    return model


# ──────────────────────────────────────────────────────────────────
# БЛОК 7: ОЦЕНКА КАЧЕСТВА МОДЕЛИ (МЕТРИКИ)
# ──────────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """
    Оцениваем качество модели на тестовой выборке.

    МЕТРИКИ — ЧТО ИЗМЕРЯЕМ:

    MAE (Mean Absolute Error) — Средняя абсолютная ошибка:
        MAE = среднее(|предсказанное - реальное|)
        Интерпретация: "В среднем модель ошибается на X баллов"
        MAE = 4.5 → "в среднем ошибка ±4.5 балла из 100" ✅ хорошо

    RMSE (Root Mean Squared Error) — Среднеквадратичная ошибка:
        RMSE = sqrt(среднее((предсказанное - реальное)²))
        Интерпретация: штрафует за БОЛЬШИЕ ошибки больше, чем MAE
        RMSE намного больше MAE → у модели есть грубые ошибки на отдельных случаях

    R² (R-squared) — Коэффициент детерминации:
        R² = 1 - (сумма квадратов ошибок / сумма квадратов отклонений от среднего)
        Интерпретация:
        R² = 1.0 → идеальная модель (предсказывает точно)
        R² = 0.0 → модель не лучше, чем предсказывать среднее
        R² = 0.85 → модель объясняет 85% вариации в данных ✅ хорошо
    """
    print("\n[5/6] Оцениваю качество модели...")

    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Обрезаем предсказания в допустимый диапазон [1, 100]
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

    # Перекрёстная проверка (Cross-Validation)
    # Объяснение: делим данные на 5 частей, обучаем на 4, тестируем на 1-й.
    # Повторяем 5 раз, меняя тестовую часть. Итого: 5 оценок → надёжно!
    print("    Запускаю 5-fold Cross-Validation (это займёт минуту)...")
    cv_scores = cross_val_score(
        XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        ),
        X_train, y_train,
        cv=5,                    # 5 "складок" (folds)
        scoring="neg_mean_absolute_error",  # sklearn минимизирует, поэтому neg
        n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()  # Убираем знак минус
    metrics["cross_val_mae"] = round(cv_mae, 3)

    # Анализ ошибок по зонам (green/yellow/red)
    df_errors = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test})
    df_errors["zone"] = pd.cut(df_errors["y_true"], bins=[0, 50, 80, 100],
                                labels=["red", "yellow", "green"])
    zone_errors = df_errors.groupby("zone", observed=True).apply(
        lambda g: round(mean_absolute_error(g["y_true"], g["y_pred"]), 2)
    ).to_dict()
    metrics["mae_by_zone"] = zone_errors

    # Вывод результатов
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

    # Оцениваем: не переобучена ли модель?
    overfit_gap = abs(metrics["train"]["MAE"] - metrics["test"]["MAE"])
    if overfit_gap < 1.0:
        print("\n    ✅ Переобучения НЕТ (Train MAE ≈ Test MAE)")
    elif overfit_gap < 3.0:
        print(f"\n    ⚠️  Небольшое переобучение (разрыв: {overfit_gap:.1f} балла)")
    else:
        print(f"\n    ❌ Переобучение! Разрыв Train/Test MAE: {overfit_gap:.1f} балла")
        print("       Попробуй: уменьшить n_estimators или max_depth")

    return metrics


# ──────────────────────────────────────────────────────────────────
# БЛОК 8: ВАЖНОСТЬ ФИЧЕЙ
# ──────────────────────────────────────────────────────────────────

def plot_feature_importance(model: XGBRegressor):
    """
    Строит и сохраняет график важности фичей.

    Feature Importance показывает, какие фичи модель использует
    больше всего для принятия решений.
    Это первый, самый простой вид объяснимости — ещё не SHAP!
    """
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


# ──────────────────────────────────────────────────────────────────
# БЛОК 9: СОЗДАНИЕ SHAP-ОБЪЯСНИТЕЛЯ
# ──────────────────────────────────────────────────────────────────

def create_shap_explainer(model: XGBRegressor, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """
    Создаём SHAP TreeExplainer для нашей модели.

    КАК РАБОТАЕТ SHAP?
    SHAP = SHapley Additive exPlanations.
    Это математически строгий метод из теории игр.

    Представь: фермер получил балл 72 из 100.
    Средний балл по всем фермерам = 55.

    SHAP объясняет: "Почему именно 72, а не 55?"
    - pedigree_ratio = 0.9 (высокая): +12 баллов (это хорошо)
    - veterinary_compliance = 0.95:    +8 баллов (тоже хорошо)
    - debt_load_ratio = 3.2 (высокий): -6 баллов (это плохо)
    - historical_survival_rate = 0.75: -3 балла
    Итого: 55 + 12 + 8 - 6 - 3 = 66 ≈ 72 (небольшая погрешность)

    SHAP сохраняет свойства:
    1. Точность: сумма SHAP-значений = предсказание - среднее
    2. Консистентность: если фича стала важнее → её SHAP растёт
    3. "Нулевой игрок": ненужные фичи получают SHAP = 0

    TreeExplainer — специальная, быстрая версия SHAP для деревьев.
    """
    print("\n    Создаю SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("    SHAP Explainer готов!")
    return explainer


# ──────────────────────────────────────────────────────────────────
# БЛОК 10: СОХРАНЕНИЕ МОДЕЛИ
# ──────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, explainer, metrics: dict):
    """
    Сохраняем все артефакты для использования в FastAPI.

    ЗАЧЕМ СОХРАНЯТЬ ОТДЕЛЬНО?
    Обучение модели — долгий процесс (минуты/часы на реальных данных).
    Когда FastAPI-сервер получает запрос, он не должен каждый раз
    обучать модель заново! Он просто загружает сохранённые веса.

    Аналогия: повар учится готовить годами (обучение).
    Но рецепт записывает в тетрадь (сохранение). Любой другой
    повар может взять тетрадь и приготовить блюдо (inference).
    """
    print("\n[6/6] Сохраняю артефакты модели...")

    # joblib — стандарт для сохранения sklearn/XGBoost объектов
    # Лучше чем pickle: поддерживает numpy массивы эффективнее
    joblib.dump(model,     MODELS_DIR / "xgb_scorer.joblib")
    joblib.dump(scaler,    MODELS_DIR / "scaler.joblib")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    # Сохраняем список фичей в JSON (чтобы FastAPI знал правильный порядок)
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(ML_FEATURES, f, ensure_ascii=False, indent=2)

    # Сохраняем отчёт о метриках
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


# ──────────────────────────────────────────────────────────────────
# БЛОК 11: ДЕМО — ПРЕДСКАЗАНИЕ ДЛЯ ОДНОГО ФЕРМЕРА
# ──────────────────────────────────────────────────────────────────

def demo_single_prediction(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Демонстрируем, как использовать модель для одного фермера.
    Это похоже на то, что будет делать FastAPI.
    """
    print("\n" + "=" * 55)
    print("  ДЕМО: Предсказание для одного фермера")
    print("=" * 55)

    # Берём первого фермера из тестовой выборки
    one_farmer = X_test.iloc[[0]]
    true_score = y_test.iloc[0]

    # Масштабируем (scaler уже обучен!)
    one_farmer_scaled = scaler.transform(one_farmer)

    # Предсказываем балл
    predicted_score = float(np.clip(model.predict(one_farmer_scaled)[0], 1, 100))

    print(f"\n  Реальный балл:     {true_score:.1f}")
    print(f"  Предсказанный:     {predicted_score:.1f}")
    print(f"  Ошибка:            {abs(predicted_score - true_score):.1f} баллов")

    # Определяем зону
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


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SmartAgro Score | Model Training Pipeline (XGBoost)")
    print("=" * 65)

    # Шаг 1: Загрузка данных
    X, y = load_data(INPUT_FILE)

    # Шаг 2: Разделение train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Шаг 3: Нормализация
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Шаг 4: Обучение XGBoost
    model = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)

    # Шаг 5: Оценка качества
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Шаг 5b: График важности фичей
    plot_feature_importance(model)

    # Шаг 5c: Создаём SHAP-объяснитель
    explainer = create_shap_explainer(model, X_train_scaled)

    # Шаг 6: Сохранение артефактов
    save_artifacts(model, scaler, explainer, metrics)

    # Демо
    demo_single_prediction(model, scaler, X_test, y_test)

    print("\n" + "=" * 65)
    print("  ✨ Обучение завершено!")
    print("  Следующий шаг: изучи shap_integration.py и main.py")
    print("=" * 65)


if __name__ == "__main__":
    main()