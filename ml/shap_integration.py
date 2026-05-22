
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path("models")

FEATURE_LABELS = {
    "ru": {
        "gross_output_growth_yoy":    "Рост валовой продукции (г/г)",
        "land_to_livestock_ratio":    "Обеспеченность пастбищами (Га/голова)",
        "historical_survival_rate":   "Сохранность поголовья (%)",
        "subsidy_dependence_index":   "Индекс зависимости от субсидий",
        "veterinary_compliance":      "Ветеринарное соответствие",
        "years_in_operation":         "Стаж работы предприятия (лет)",
        "pedigree_ratio":             "Доля племенного поголовья",
        "previous_subsidies_count":   "Количество предыдущих субсидий",
        "debt_load_ratio":            "Долговая нагрузка (Долг/EBITDA)",
        "grazing_norm_deviation":     "Отклонение нагрузки пастбищ от нормы",
        "natural_loss_risk_score":    "Риск аномальной смертности",
        "log_amount":                 "Масштаб заявки (log суммы)",
        "livestock_count":            "Количество голов скота",
        "direction_code":             "Направление животноводства",
        "is_pedigree":                "Племенное направление",
        "is_producer":                "Субсидия на производителей",
        "hour_submitted":             "Час подачи заявки",
        "month_submitted":            "Месяц подачи заявки",
        "region_encoded":             "Регион хозяйства",
        "language_code":              "Язык заявки (0=RU, 1=KZ)",
    },
    "kz": {
        "gross_output_growth_yoy":    "Жалпы өнімнің өсуі (ж/ж)",
        "land_to_livestock_ratio":    "Жайылыммен қамтамасыз ету (Га/бас)",
        "historical_survival_rate":   "Мал сақталуы (%)",
        "subsidy_dependence_index":   "Субсидияға тәуелділік индексі",
        "veterinary_compliance":      "Ветеринарлық сәйкестік",
        "years_in_operation":         "Кәсіпорынның жұмыс стажы (жыл)",
        "pedigree_ratio":             "Тұқымдық мал үлесі",
        "previous_subsidies_count":   "Алдыңғы субсидиялар саны",
        "debt_load_ratio":            "Борыш жүктемесі (Борыш/EBITDA)",
        "grazing_norm_deviation":     "Жайылым жүктемесінің ауытқуы",
        "natural_loss_risk_score":    "Аномалды өлім қаупі",
        "log_amount":                 "Өтінім масштабы (log сома)",
        "livestock_count":            "Мал басының саны",
        "direction_code":             "Мал шаруашылығы бағыты",
        "is_pedigree":                "Тұқымдық бағыт",
        "is_producer":                "Өндірушілерге субсидия",
        "hour_submitted":             "Өтінім беру сағаты",
        "month_submitted":            "Өтінім беру айы",
        "region_encoded":             "Шаруашылық аймағы",
        "language_code":              "Өтінім тілі (0=RU, 1=KZ)",
    },
}

DIRECTION_NAMES = {
    "ru": {
        0: "скотоводство (КРС)",
        1: "овцеводство",
        2: "коневодство",
        3: "птицеводство",
        4: "верблюдоводство",
        5: "свиноводство",
        6: "прочее",
    },
    "kz": {
        0: "мал шаруашылығы (ІҚМ)",
        1: "қой шаруашылығы",
        2: "жылқы шаруашылығы",
        3: "құс шаруашылығы",
        4: "түйе шаруашылығы",
        5: "шошқа шаруашылығы",
        6: "басқа",
    },
}

# These keys are expected by the feature merge logic in src/main.py.
DOC_FEATURE_KEYS = frozenset({
    "gross_output_growth_yoy",
    "land_to_livestock_ratio",
    "historical_survival_rate",
    "subsidy_dependence_index",
    "veterinary_compliance",
    "years_in_operation",
    "pedigree_ratio",
    "previous_subsidies_count",
    "debt_load_ratio",
    "livestock_count",
    "land_area_ha",
    "has_vet_passport",
})

class ScoringEngine:

    def __init__(self, models_dir: Path = MODELS_DIR):
        print("🚀 Загружаю SmartAgro Scoring Engine...")

        self.model = joblib.load(models_dir / "xgb_scorer.joblib")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.explainer = joblib.load(models_dir / "shap_explainer.joblib")

        with open(models_dir / "feature_names.json", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        self._validate_shap_consistency()

        print(f"✅ Движок готов: {len(self.feature_names)} фичей, модель загружена")

    def _validate_shap_consistency(self):
        """Проверяет что SHAP explainer даёт те же предсказания что модель."""
        try:
            # Тестовый вектор из медиан
            test_vals = [0.05, 6.0, 0.88, 0.25, 0.80, 8.0, 0.35, 3.0, 1.0, 0.0, 1.0, 14.0, 50.0, 0.0, 0.0, 0.0, 12.0, 6.0, 7.0]
            X = pd.DataFrame([dict(zip(self.feature_names, test_vals))], columns=self.feature_names)
            X_scaled = self.scaler.transform(X)

            model_pred = self.model.predict(X_scaled)[0]
            shap_base = getattr(self.explainer, "expected_value", 0.0)
            # Извлекаем скаляр из 0-мерного массива, если нужно
            if isinstance(shap_base, np.ndarray):
                shap_base = shap_base.item()
            shap_vals = self.explainer.shap_values(X_scaled)
            if shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
            shap_pred = float(shap_base) + sum(shap_vals)

            diff = abs(model_pred - shap_pred)
            if diff > 5.0:
                print(f"⚠️ SHAP не сходится с моделью! Разница: {diff:.1f} баллов")
                print(f"   Model pred: {model_pred:.1f}, SHAP pred: {shap_pred:.1f}")
                print(f"   Рекомендуется пересохранить shap_explainer.joblib")
            else:
                print(f"✅ SHAP consistency OK (разница: {diff:.2f})")
        except Exception as e:
            print(f"⚠️ Не удалось проверить SHAP consistency: {e}")

    def score_farmer(
        self,
        raw_features: dict,
        llm_context: Optional[str] = None,
        *,
        lang: str = "ru",
        include_shap: bool = True,
        imputed_fields: Optional[list[str]] = None,
        feature_sources: Optional[dict[str, str]] = None,
    ) -> dict:

        X = self._prepare_feature_vector(raw_features)

        X_scaled = self._scale(X)

        raw_score = float(self.model.predict(X_scaled)[0])
        score = round(float(np.clip(raw_score, 1.0, 100.0)), 1)

        imputed_set = set(imputed_fields or [])
        sources_map = feature_sources or {}

        if include_shap:
            shap_values = self._compute_shap(X_scaled)
            shap_base = self._get_shap_base_value(X_scaled)
            top_positive, top_negative = self._build_explanations(
                shap_values,
                raw_features,
                lang=lang,
                n_factors=3,
                imputed_fields=imputed_set,
                feature_sources=sources_map,
            )
            all_shap_values = {
                name: round(float(val), 3)
                for name, val in zip(self.feature_names, shap_values)
            }
            explainability = "SHAP TreeExplainer"
        else:
            shap_base = 0.0
            top_positive, top_negative = [], []
            all_shap_values = {}
            explainability = "отключено (демо-пакет)"

        zone, zone_label, recommendation = self._get_zone(score, lang=lang)

        verdict_text = self._generate_verdict(
            score, zone, top_positive, top_negative, llm_context, lang=lang
        )

        return {
            "score": score,
            "zone": zone,
            "zone_label": zone_label,
            "recommendation": recommendation,
            "verdict": verdict_text,
            "shap_base_value": shap_base,
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "all_shap_values": all_shap_values,
            "raw_features_used": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in raw_features.items()
                if k in self.feature_names
            },
            "model_version": "XGBoost-v1.0",
            "explainability": explainability,
        }

    # Медианы для missing features (вместо 0.0 который даёт экстремальный outlier)
    _FEATURE_MEDIANS = {
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
        "language_code": 0.0,
    }

    def _prepare_feature_vector(self, raw: dict) -> pd.DataFrame:
        row = {}
        for feat in self.feature_names:
            val = raw.get(feat)
            if val is None:
                # Используем медиану вместо 0.0 (0.0 = экстремальный outlier после StandardScaler)
                row[feat] = self._FEATURE_MEDIANS.get(feat, 0.0)
            else:
                row[feat] = float(val)
        return pd.DataFrame([row], columns=self.feature_names)

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)

    def _compute_shap(self, X_scaled: np.ndarray) -> np.ndarray:
        shap_vals = self.explainer.shap_values(X_scaled)

        if shap_vals.ndim == 2:
            return shap_vals[0]
        return shap_vals

    def _get_shap_base_value(self, X_scaled: np.ndarray) -> float:
        base = getattr(self.explainer, "expected_value", 0.0)
        try:

            if isinstance(base, (list, tuple, np.ndarray)):

                return float(np.array(base).reshape(-1)[0])
            return float(base)
        except Exception:
            return 0.0

    def _build_explanations(
        self,
        shap_values: np.ndarray,
        raw_features: dict,
        lang: str = "ru",
        n_factors: int = 3,
        imputed_fields: Optional[set] = None,
        feature_sources: Optional[dict[str, str]] = None,
    ) -> tuple[list[dict], list[dict]]:
        imputed_fields = imputed_fields or set()
        feature_sources = feature_sources or {}
        source_suffix = {
            "ru": {
                "form": " — из анкеты",
                "pdf_regex": " — из текста PDF (распознавание)",
                "pdf_llm": " — из PDF (LLM)",
                "pdf": " — из PDF",
                "imputed": " — нет в PDF/анкете, подставлено среднее для модели",
            },
            "kz": {
                "form": " — анкетадан",
                "pdf_regex": " — PDF мәтінінен",
                "pdf_llm": " — PDF (LLM)",
                "pdf": " — PDF-тен",
                "imputed": " — PDF/анкетада жоқ, модель үшін орташа мән",
            },
        }
        suf = source_suffix.get(lang, source_suffix["ru"])
        factors = []
        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            raw_val = raw_features.get(name, 0.0)
            if raw_val is None:
                raw_val = 0.0
            label = FEATURE_LABELS[lang].get(name, name)

            explanation_text = self._explain_feature(name, raw_val, float(shap_val), lang=lang)
            src = feature_sources.get(name) or ("imputed" if name in imputed_fields else "model")
            explanation_text += suf.get(src, suf.get("imputed", ""))
            if name in imputed_fields:
                raw_display = None
            else:
                raw_display = round(float(raw_val), 4) if isinstance(raw_val, (int, float)) else raw_val

            factors.append({
                "feature": name,
                "label": label,
                "shap_value": round(float(shap_val), 2),
                "raw_value": raw_display,
                "imputed": name in imputed_fields,
                "source": src,
                "direction": "positive" if shap_val > 0 else "negative",
                "explanation": explanation_text,
                "impact_text": f"{'+'if shap_val>0 else ''}{shap_val:.1f} балл: {explanation_text}",
            })

        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        positive = [f for f in factors if f["shap_value"] > 0][:n_factors]
        negative = [f for f in factors if f["shap_value"] < 0][:n_factors]

        return positive, negative

    def _explain_feature(self, name: str, value: float, shap_val: float, lang: str = "ru") -> str:
        positive = shap_val > 0

        TEXT = {
            "ru": {
                "gross_pos": lambda v: f"Рост валовой продукции {v:+.1f}% г/г — положительная динамика",
                "gross_neg": lambda v: f"Спад валовой продукции {v:+.1f}% г/г — отрицательная динамика",
                "pedigree_pos": lambda v: f"Высокая доля племенного поголовья {v:.0f}% — значительный племенной потенциал",
                "pedigree_neg": lambda v: f"Низкая доля племенного поголовья {v:.0f}% — слабая племенная база",
                "survival_pos": lambda v: f"Высокая сохранность стада {v:.1f}% — хорошее ветеринарное управление",
                "survival_neg": lambda v: f"Низкая сохранность стада {v:.1f}% — повышенный падёж животных",
                "subsidy_pos": lambda v: f"Умеренная зависимость от субсидий {v:.0f}% — экономическая самостоятельность",
                "subsidy_neg": lambda v: f"Высокая зависимость от субсидий {v:.0f}% — бизнес на дотациях",
                "vet_pos": lambda v: f"Высокое ветеринарное соответствие {v:.0f}% — нормы соблюдены",
                "vet_neg": lambda v: f"Нарушение ветеринарных норм (соответствие {v:.0f}%)",
                "debt_pos": lambda v: f"Низкая долговая нагрузка (Долг/EBITDA = {v:.2f})",
                "debt_neg": lambda v: f"Высокая долговая нагрузка (Долг/EBITDA = {v:.2f})",
                "years_pos": lambda v: f"Опытное предприятие ({v:.0f} лет) — проверенная история",
                "years_neg": lambda v: f"Молодое предприятие ({v:.0f} лет) — ограниченная история",
                "land_pos": lambda v: f"Хорошая обеспеченность пастбищами ({v:.1f} Га/голову)",
                "land_neg": lambda v: f"Низкая обеспеченность пастбищами ({v:.1f} Га/голову)",
                "prev_pos": lambda v: f"Успешная история субсидирования ({v:.0f} субсидий)",
                "prev_neg": lambda v: f"Нет истории субсидирования — новый участник",
                "livestock_pos": lambda v: f"Крупное хозяйство ({v:.0f} голов)",
                "livestock_neg": lambda v: f"Небольшое хозяйство ({v:.0f} голов)",
                "pedigree_dir": "Субсидия на племенное поголовье — стратегическое направление",
                "pedigree_dir_no": "Субсидия на товарное производство",
                "grazing_pos": lambda v: f"Нагрузка на пастбища в норме (отклонение {v:+.2f} от норматива)",
                "grazing_neg": lambda v: f"Превышение/дефицит нормы выпаса (отклонение {v:+.2f})",
                "risk_pos": lambda v: f"Умеренный риск естественной убыли (индекс {v:.2f})",
                "risk_neg": lambda v: f"Повышенный риск падежа относительно нормы (индекс {v:.2f})",
                "log_amt_pos": lambda v: f"Крупный запрашиваемый объём субсидии (log-сумма {v:.1f})",
                "log_amt_neg": lambda v: f"Небольшой запрашиваемый объём (log-сумма {v:.1f})",
                "dir_pedigree": "Направление субсидии — племенное животноводство",
                "dir_other": "Направление субсидии — неплеменное/иное",
                "producer_yes": "Заявка от производителя продукции",
                "producer_no": "Не отмечен как производитель",
                "region": lambda v: f"Регион закодирован для модели (код {v:.0f})",
                "season": lambda v: f"Месяц подачи заявки: {int(v)}",
                "hour": lambda v: f"Час подачи заявки: {int(v)}",
                "lang_ru": "Заявка на русском языке",
                "lang_kz": "Заявка на казахском языке",
                "default": lambda lbl, d: f"Показатель «{lbl}» влияет {d} на балл модели",
            },
            "kz": {
                "gross_pos": lambda v: f"Жалпы өнімнің өсуі {v:+.1f}% ж/ж — оң динамика",
                "gross_neg": lambda v: f"Жалпы өнімнің төмендеуі {v:+.1f}% ж/ж — теріс динамика",
                "pedigree_pos": lambda v: f"Тұқымдық мал үлесі жоғары {v:.0f}% — елеуметтік әлеует",
                "pedigree_neg": lambda v: f"Тұқымдық мал үлесі төмен {v:.0f}% — әлсіз тұқымдық база",
                "survival_pos": lambda v: f"Мал сақталуы жоғары {v:.1f}% — жақсы ветеринарлық басқару",
                "survival_neg": lambda v: f"Мал сақталуы төмен {v:.1f}% — жоғары шығын",
                "subsidy_pos": lambda v: f"Субсидияға тәуелділік төмен {v:.0f}% — экономикалық дербестік",
                "subsidy_neg": lambda v: f"Субсидияға жоғары тәуелділік {v:.0f}% — субсидиядағы бизнес",
                "vet_pos": lambda v: f"Ветеринарлық сәйкестік жоғары {v:.0f}% — нормалар сақталған",
                "vet_neg": lambda v: f"Ветеринарлық нормалардың бұзылуы (сәйкестік {v:.0f}%)",
                "debt_pos": lambda v: f"Борыш жүктемесі төмен (Борыш/EBITDA = {v:.2f})",
                "debt_neg": lambda v: f"Борыш жүктемесі жоғары (Борыш/EBITDA = {v:.2f})",
                "years_pos": lambda v: f"Тәжірибелі кәсіпорын ({v:.0f} жыл) — тексерілген тарих",
                "years_neg": lambda v: f"Жас кәсіпорын ({v:.0f} жыл) — шектеулі тарих",
                "land_pos": lambda v: f"Жайылыммен жақсы қамтамасыз етілген ({v:.1f} Га/бас)",
                "land_neg": lambda v: f"Жайылыммен нашар қамтамасыз етілген ({v:.1f} Га/бас)",
                "prev_pos": lambda v: f"Сәтті субсидиялау тарихы ({v:.0f} субсидия)",
                "prev_neg": lambda v: f"Субсидиялау тарихы жоқ — жаңа қатысушы",
                "livestock_pos": lambda v: f"Ірі шаруашылық ({v:.0f} бас)",
                "livestock_neg": lambda v: f"Кіші шаруашылық ({v:.0f} бас)",
                "pedigree_dir": "Тұқымдық малға субсидия — стратегиялық бағыт",
                "pedigree_dir_no": "Тауарлық өндіріске субсидия",
                "grazing_pos": lambda v: f"Жайылым жүктемесі нормада (ауытқу {v:+.2f})",
                "grazing_neg": lambda v: f"Жайылым нормасынан ауытқу ({v:+.2f})",
                "risk_pos": lambda v: f"Табиғи құрау тәуекелі орташа (индекс {v:.2f})",
                "risk_neg": lambda v: f"Құрау тәуекелі жоғары (индекс {v:.2f})",
                "log_amt_pos": lambda v: f"Ірі субсидия сомасы (log {v:.1f})",
                "log_amt_neg": lambda v: f"Кіші субсидия сомасы (log {v:.1f})",
                "dir_pedigree": "Субсидия бағыты — тұқымдық мал",
                "dir_other": "Субсидия бағыты — басқа",
                "producer_yes": "Өндіруші ретінде көрсетілген",
                "producer_no": "Өндіруші емес",
                "region": lambda v: f"Аймақ коды ({v:.0f})",
                "season": lambda v: f"Өтінім айы: {int(v)}",
                "hour": lambda v: f"Өтінім сағаты: {int(v)}",
                "lang_ru": "Өтінім орыс тілінде",
                "lang_kz": "Өтінім қазақ тілінде",
                "default": lambda lbl, d: f"«{lbl}» көрсеткіші {d} әсер етеді",
            },
        }
        t = TEXT[lang]

        if name == "gross_output_growth_yoy":
            pct = value * 100
            return t["gross_pos"](pct) if positive else t["gross_neg"](pct)
        elif name == "pedigree_ratio":
            pct = value * 100
            return t["pedigree_pos"](pct) if positive else t["pedigree_neg"](pct)
        elif name == "historical_survival_rate":
            pct = value * 100
            return t["survival_pos"](pct) if positive else t["survival_neg"](pct)
        elif name == "subsidy_dependence_index":
            pct = value * 100
            return t["subsidy_pos"](pct) if positive else t["subsidy_neg"](pct)
        elif name == "veterinary_compliance":
            pct = value * 100
            return t["vet_pos"](pct) if positive else t["vet_neg"](pct)
        elif name == "debt_load_ratio":
            return t["debt_pos"](value) if positive else t["debt_neg"](value)
        elif name == "years_in_operation":
            return t["years_pos"](value) if positive else t["years_neg"](value)
        elif name == "land_to_livestock_ratio":
            return t["land_pos"](value) if positive else t["land_neg"](value)
        elif name == "previous_subsidies_count":
            return t["prev_pos"](value) if positive else t["prev_neg"](value)
        elif name == "livestock_count":
            return t["livestock_pos"](value) if positive else t["livestock_neg"](value)
        elif name == "is_pedigree":
            return t["pedigree_dir"] if value == 1 else t["pedigree_dir_no"]
        elif name == "is_producer":
            return t["producer_yes"] if value == 1 else t["producer_no"]
        elif name == "grazing_norm_deviation":
            return t["grazing_pos"](value) if positive else t["grazing_neg"](value)
        elif name == "natural_loss_risk_score":
            return t["risk_pos"](value) if positive else t["risk_neg"](value)
        elif name == "log_amount":
            return t["log_amt_pos"](value) if positive else t["log_amt_neg"](value)
        elif name == "direction_code":
            return t["dir_pedigree"] if value <= 2 else t["dir_other"]
        elif name == "region_encoded":
            return t["region"](value)
        elif name == "month_submitted":
            return t["season"](value)
        elif name == "hour_submitted":
            return t["hour"](value)
        elif name == "language_code":
            return t["lang_kz"] if value >= 0.5 else t["lang_ru"]
        else:
            lbl = FEATURE_LABELS[lang].get(name, name)
            direction = "оң" if positive else "теріс" if lang == "kz" else "положительно" if positive else "отрицательно"
            return t["default"](lbl, direction)

    def _get_zone(self, score: float, lang: str = "ru") -> tuple[str, str, str]:
        if score >= 80:
            if lang == "kz":
                return ("green", "Жасыл аймақ (80–100)", "Қысқа тізімге басымдықпен ұсынылады")
            return ("green", "Зелёная зона (80–100)", "Строго рекомендовано к включению в шорт-лист")
        elif score >= 50:
            if lang == "kz":
                return ("yellow", "Сары аймақ (50–79)", "Комиссиямен қосымша қарау ұсынылады")
            return ("yellow", "Жёлтая зона (50–79)", "Рекомендуется дополнительное рассмотрение комиссией")
        else:
            if lang == "kz":
                return ("red", "Қызыл аймақ (0–49)", "Ұсынылмайды — елеуметті тәуекелдер анықталды")
            return ("red", "Красная зона (0–49)", "Не рекомендовано — выявлены существенные риски")

    def _generate_verdict(
        self,
        score: float,
        zone: str,
        top_positive: list,
        top_negative: list,
        llm_context: Optional[str],
        lang: str = "ru",
    ) -> str:
        verdict_parts = []

        ZONE_INTROS = {
            "ru": {
                "green": f"Предприятие получило высокий балл {score:.0f}/100 и рекомендуется к приоритетному рассмотрению.",
                "yellow": f"Предприятие получило балл {score:.0f}/100. Рекомендуется детальное рассмотрение комиссией.",
                "red":    f"Предприятие получило балл {score:.0f}/100. Система выявила существенные риски.",
            },
            "kz": {
                "green": f"Кәсіпорын {score:.0f}/100 балл алды және басымдықпен қарауға ұсынылады.",
                "yellow": f"Кәсіпорын {score:.0f}/100 балл алды. Комиссиямен толық қарау ұсынылады.",
                "red":    f"Кәсіпорын {score:.0f}/100 балл алды. Жүйе елеуметті тәуекелдерді анықтады.",
            },
        }
        verdict_parts.append(ZONE_INTROS[lang][zone])

        strong_label = "✅ Сильные стороны:" if lang == "ru" else "✅ Күшті жақтары:"
        risk_label = "⚠️ Факторы риска:" if lang == "ru" else "⚠️ Тәуекел факторлары:"
        doc_label = "📄 Данные из документов:" if lang == "ru" else "📄 Құжаттардан алынған деректер:"

        if top_positive:
            verdict_parts.append(f"\n{strong_label}")
            for factor in top_positive:
                verdict_parts.append(f"  • {factor['explanation']}")

        if top_negative:
            verdict_parts.append(f"\n{risk_label}")
            for factor in top_negative:
                verdict_parts.append(f"  • {factor['explanation']}")

        if llm_context:
            verdict_parts.append(f"\n{doc_label}\n  {llm_context}")

        DISCLAIMER = {
            "ru": "\n⚖️ Данная оценка является рекомендацией ИИ-системы. Окончательное решение принимается уполномоченной комиссией Министерства сельского хозяйства РК.",
            "kz": "\n⚖️ Бұл баға ЖЖ жүйесінің ұсынысы болып табылады. Қорытынды шешімді ҚР Ауыл шаруашылығы министрлігінің уәкілетті комиссиясы қабылдайды.",
        }
        verdict_parts.append(DISCLAIMER[lang])

        return "\n".join(verdict_parts)

def _strip_markdown_json_fence(s: str) -> str:
    import re

    t = s.strip()
    m = re.match(r"^```(?:json)?\s*\r?\n?(.*)\r?\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _doc_feature_system_prompt() -> str:
    return """Ты — аналитик Министерства сельского хозяйства РК.
Твоя задача: извлечь структурированные факты из документов сельхозпредприятия
для системы скоринга субсидий. Текст может быть на русском и/или казахском (двуязычные PDF).

Ищи явные подписи (примеры): «Ветсоответствие» / «Ветсәйкестік», «Племдоля» / «Плем үлесі»,
«Сохранность» / «Сақталуы», «Рост валовой продукции г/г» / «Жалпы өнім өсуі ж/ж»,
«Долг/EBITDA» / «Қарыз/EBITDA», «Стаж, лет», «Зависимость от субсидий»,
«Ранее субсидий», «Отклонение нагрузки», «Риск vs норма падежа», «Обеспеченность, га/гол».

КРИТИЧНО для полей years_in_operation, subsidy_dependence_index, gross_output_growth_yoy:
в JSON возвращай ТОЛЬКО числа (integer или float), БЕЗ единиц измерения и без текста.
Недопустимо: "20 лет", "20", "+25%", "менее 4%". Допустимо: 20, 0.25, 0.04.
— years_in_operation: целое число полных лет.
  Если в документе указан только год основания (например, 2005), ты ОБЯЗАН вычислить
  количество лет до текущего 2026 года (2026 минус год основания) и записать это число
  в years_in_operation (для 2005 года это 21). Не оставляй null, если год основания явно дан.
— gross_output_growth_yoy: доля, не проценты: рост +25% за год → 0.25; спад -5% → -0.05.
— subsidy_dependence_index: доля от 0 до 1. ПРИОРИТЕТ: если в тексте есть «менее N%»
  (например «зависимость менее 4%»), верни строго N/100 как число — для 4% это 0.04.
  Не подставляй «типичные для отрасли» значения; не игнорируй «менее» в пользу других процентов.
— veterinary_compliance и historical_survival_rate: доля 0–1 (100% → 1.0, 99.2% → 0.992).

Отвечай ТОЛЬКО валидным JSON без markdown-блоков, пояснений и вводных слов.

Формат ответа (строго эти ключи, никаких дополнительных полей):
{
  "livestock_count": <число голов или null>,
  "land_area_ha": <площадь сельхозугодий/пастбища в гектарах или null>,
  "land_to_livestock_ratio": <га на одну голову скота; если явно нет — null>,
  "has_vet_passport": <true/false или null>,
  "veterinary_compliance": <0.0-1.0 на основе документов или null>,
  "historical_survival_rate": <0.0-1.0 сохранность/выживаемость или null>,
  "years_in_operation": <число лет (только число) или null>,
  "gross_output_growth_yoy": <только число: доля роста г/г (0.05 = +5%); null если нет>,
  "subsidy_dependence_index": <только число 0.0-1.0 доля; null если нет>,
  "pedigree_ratio": <0.0-1.0 доля племенных или null>,
  "previous_subsidies_count": <число ранее полученных субсидий или null>,
  "debt_load_ratio": <коэффициент Долг/EBITDA или null>,
  "llm_summary": "<1-2 предложения: ключевые выводы по документам>"
}"""


def _doc_feature_user_message(documents_text: str) -> str:
    return f"""Проанализируй следующие документы сельхозпредприятия и извлеки данные:

---НАЧАЛО ДОКУМЕНТОВ---
{documents_text}
---КОНЕЦ ДОКУМЕНТОВ---

Верни только JSON с извлечёнными данными. Если данных нет — ставь null."""


def _parse_doc_features_llm_response(response_text: str, unknown_log_tag: str) -> dict:
    response_text = _strip_markdown_json_fence((response_text or "").strip())
    if not response_text:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": "empty_model_response",
        }

    try:
        try:
            extracted = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise
            extracted = json.loads(json_match.group())

        if not isinstance(extracted, dict):
            return {
                "features": {},
                "llm_summary": None,
                "extraction_status": "response_not_object",
            }

        llm_summary = extracted.pop("llm_summary", None)

        for key in ["has_vet_passport"]:
            if key in extracted and isinstance(extracted[key], bool):
                extracted[key] = 1.0 if extracted[key] else 0.0

        unknown = [k for k in extracted.keys() if k not in DOC_FEATURE_KEYS]
        if unknown:
            print(f"[extract_features_from_documents] Пропускаю неизвестные ключи ({unknown_log_tag}): {unknown}")
        extracted = {k: v for k, v in extracted.items() if k in DOC_FEATURE_KEYS}

        return {
            "features": extracted,
            "llm_summary": llm_summary,
            "extraction_status": "success",
        }

    except json.JSONDecodeError as e:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": f"json_parse_error: {e}",
        }


def extract_features_from_documents(documents_text: str, api_key: str) -> dict:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    genai.configure(api_key=api_key)
    system_prompt = _doc_feature_system_prompt()
    user_message = _doc_feature_user_message(documents_text)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
        )
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.15,
        )
        message = model.generate_content(
            user_message,
            generation_config=generation_config,
        )
        return _parse_doc_features_llm_response((message.text or "").strip(), "Gemini")
    except Exception as e:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": f"api_error: {e}",
        }


def extract_features_from_documents_groq(documents_text: str) -> dict:
    from ml.llm_routing import groq_chat

    system_prompt = _doc_feature_system_prompt()
    user_message = _doc_feature_user_message(documents_text)
    try:
        raw = groq_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.15,
            max_tokens=8192,
        )
        return _parse_doc_features_llm_response(raw, "Groq")
    except Exception as e:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": f"api_error: {e}",
        }


def extract_features_from_documents_openai(documents_text: str) -> dict:
    from ml.llm_routing import openai_doc_chat

    system_prompt = _doc_feature_system_prompt()
    user_message = _doc_feature_user_message(documents_text)
    try:
        raw = openai_doc_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.15,
            max_tokens=8192,
        )
        return _parse_doc_features_llm_response(raw, "OpenAI/OpenRouter")
    except Exception as e:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": f"api_error: {e}",
        }


def extract_features_from_documents_auto(documents_text: str) -> dict:
    """OpenAI-совместимый (OpenRouter), Groq или Gemini по LLM_PROVIDER / ключам."""
    from ml.llm_routing import primary_cloud_llm

    backend = primary_cloud_llm()
    gq = (os.getenv("GROQ_API_KEY") or "").strip()
    gm = (os.getenv("GEMINI_API_KEY") or "").strip()
    oa = (os.getenv("OPENAI_API_KEY") or "").strip()

    if backend == "openai" and oa:
        return extract_features_from_documents_openai(documents_text)
    if backend == "groq" and gq:
        return extract_features_from_documents_groq(documents_text)
    if backend == "gemini" and gm:
        return extract_features_from_documents(documents_text, gm)
    if oa:
        return extract_features_from_documents_openai(documents_text)
    if gm:
        return extract_features_from_documents(documents_text, gm)
    if gq:
        return extract_features_from_documents_groq(documents_text)
    return {
        "features": {},
        "llm_summary": None,
        "extraction_status": "no_cloud_llm",
    }

def extract_text_from_pdf(pdf_path: str) -> str:
    max_pages = 40

    def _ok(s: str) -> bool:
        return bool(s and len(s.strip()) > 30)

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            parts = []
            for page in pdf.pages[:max_pages]:
                t = page.extract_text()
                if t:
                    parts.append(t)
            out = "\n\n".join(parts).strip()
            if _ok(out):
                return out
    except Exception:
        pass

    try:
        import fitz           
        doc = fitz.open(pdf_path)
        try:
            parts = []
            for i in range(min(max_pages, doc.page_count)):
                parts.append(doc.load_page(i).get_text("text"))
            out = "\n\n".join(parts).strip()
            if _ok(out):
                return out
        finally:
            doc.close()
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        parts = [page.extract_text() or "" for page in reader.pages[:max_pages]]
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _is_gemini_quota_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "429" in str(exc) or "quota" in s or "resource exhausted" in s or "resourceexhausted" in type(exc).__name__.lower()


def _gemini_429_retry_delay(exc: BaseException) -> float:
    m = re.search(r"retry in ([0-9.]+)\s*s", str(exc), re.I)
    if m:
        return min(120.0, float(m.group(1)) + 3.0)
    return 25.0


def _expert_chat_completion(
    *,
    api_key: str,
    base_url: Optional[str],
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """OpenAI-совместимый Chat Completions (OpenAI, Groq, локальный прокси и т.д.)."""
    try:
        from openai import OpenAI
    except ImportError as ie:
        raise RuntimeError("Установите пакет openai: pip install openai") from ie
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Пустой ответ модели")
    return text


def _expert_opinion_via_groq(system_prompt: str, user_message: str) -> str:
    """Бесплатный tier Groq (отдельная квота от Gemini). Ключ: https://console.groq.com/keys"""
    gkey = os.getenv("GROQ_API_KEY", "").strip()
    if not gkey:
        raise RuntimeError("GROQ_API_KEY не задан")
    model = (os.getenv("GROQ_EXPERT_MODEL") or "llama-3.1-8b-instant").strip()
    return _expert_chat_completion(
        api_key=gkey,
        base_url="https://api.groq.com/openai/v1",
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
    )


def _expert_opinion_via_openai(system_prompt: str, user_message: str) -> str:
    """Платный OpenAI или совместимый endpoint (OPENAI_API_BASE)."""
    okey = os.getenv("OPENAI_API_KEY", "").strip()
    if not okey:
        raise RuntimeError("OPENAI_API_KEY не задан")
    model = (os.getenv("OPENAI_EXPERT_MODEL") or "gpt-4o-mini").strip()
    base = os.getenv("OPENAI_API_BASE", "").strip() or None
    return _expert_chat_completion(
        api_key=okey,
        base_url=base,
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
    )


def generate_gemini_expert_opinion(app_data: dict, api_key: Optional[str] = None) -> str:
    from ml.llm_routing import expert_opinion_provider

    rules_summary = """
=== НОРМАТИВНАЯ БАЗА: СУБСИДИРОВАНИЕ ПЛЕМЕННОГО ЖИВОТНОВОДСТВА (РК) ===
Базовый акт: Приказ Министра сельского хозяйства РК от 15.03.2019 № 108 «Правила субсидирования развития
племенного животноводства, повышения продуктивности и качества продукции животноводства»;
зарегистрирован в МЮ РК 20.03.2019 № 18404; отменён приказ № 256 от 15.06.2018 (Правила 2018 года).
Преамбула и понятия актуализировались приказами, в т.ч.: № 207 от 13.07.2021; № 332 от 18.09.2023 (ред. Правил
с 01.01.2024); № 331 от 30.09.2024; № 217 от 25.06.2024; № 189 от 04.06.2025 (цель субсидирования п. 1-1,
определение «субсидирование», корректировки п. 2); № 352 от 03.10.2025 (в т.ч. п. 16 подтверждение ЭСФ с 01.01.2026,
п. 17–18 спецкомиссия и мониторинг мощностей); № 428 от 18.11.2025 (Приложение 1 — нормативы, п. 6 распределение).
Актуальные редакции проверять по опубликованному тексту на eGov/Әділет.

--- СВЯЗАННЫЙ АКТ: НОРМЫ ЕСТЕСТВЕННОЙ УБЫЛИ (ПАДЕЖА) ---
Приказ Министра СХ РК от 03.12.2015 № 3-3/1061 (МЮ № 12488, введён со 02.06.2016). Используется при толковании
обязательств «кроме норм естественного убыли (падежа)»: допустимые доли падежа по видам (примеры):
КРС мясное направление — импорт 1-й год после карантина: авиа 2,5%, авто >2000 км 5%, комбинированная морская+авто 7,5%;
яловое маточное 5% от осеменённого/слученного; телята до отъёма 6–8 мес. 2%; молодняк на откорме до 15 мес. 2%;
маточное поголовье (год) 2%. КРС молочное — импорт 3–9%; маточное 3%; телята до 20 суток 3,5% и др. по таблице приказа.
МРС: взрослые овцы/козы 3%; ягнята/козлята до отъёма 4 мес. 5%; ремонтный молодняк до 18 мес. 2%; откорм 1%.
Свиньи: поросята-сосуны до отъёма 12,5%; доращивание 5,2%; откорм 1%. Птица, рыба, пчёлы — свои табличные нормы в том же приказе.
При оценке заявки сверяй фактическую «сохранность» с этими ориентирами: падеж в пределах нормы не равен нарушению обязательств.

--- ГЛАВА 1. ЦЕЛИ И ПОНЯТИЯ (фрагмент) ---
Цель (п. 1-1): развитие племенного животноводства, продуктивность и качество, доступность племенных животных и услуг,
удешевление затрат на производство животноводческой продукции.
Товаропроизводитель: физ./юр. лицо, производящее с/х продукцию; племенное хозяйство; племенной/дистрибьюторский центр;
техник-осеменатор и др. по тексту п. 2.
ГИСС — государственная информационная система субсидирования; заявка подписывается ЭЦП (п. 20–21).
Маточное поголовье (возраст половозрелости по Правилам): КРС племенные от 13 мес, товарные от 18 мес; овцы от 12 мес;
лошади от 36 мес; свиньи племенные/товарные от 8 мес, ремонтное от 4 мес.
Целевое использование — воспроизводство в сроки и на условиях Правил.
Откормочная площадка — с учётным номером по Правилам присвоения учётных номеров объектам (приказ МСХ № 7-1/37 от 23.01.2015, ЗРНУА № 10466).
ИБСПР, ИСЖ, ИС ЭСФ, ЕГКН — как в Правилах. Аномальные погодные условия для кормов (п. 2 п. 1-1): засуха, град ≥20 мм,
ливень ≥50 мм за 12 ч (в селеопасных регионах ≥30 мм за 12 ч) и др.

--- ГЛАВА 2–3. БЮДЖЕТ, МИО, РЕГИСТРАЦИЯ ---
Субсидии — из местного бюджета; МИО областей/г. респ. значения формируют объёмы по Приложению 1, согласуют с Министерством;
перераспределение между видами при нехватке заявок; отчётность до 1 февраля (ф. 4-2, аналитика KPI животноводства) — п. 6.
Регистрация в ГИСС: БИН/ИИН, наименование, руководитель, контакты, ИИК/БИК банка второго уровня.
Подтверждение сделок: электронные счета-фактуры через ИС ЭСФ (интеграция с ГИСС); импорт — ТД или заявление о ввозе (НК, ЕАЭС).
Встречные обязательства по валовой продукции АПК (п. 14-1) — при субсидиях от 100 млн ₸ в текущем году (проверять редакцию).

--- СПЕЦИАЛЬНАЯ КОМИССИЯ И МОЩНОСТИ (п. 17–18, ред. № 352) ---
МИО создаёт комиссию по производственной мощности и инфраструктуре; заключение по форме Приложения 5; срок осмотра до 5 раб. дней
(+5 по согласованию); в ГИСС — в течение 2 раб. дней после подписания; не менее 3 специалистов; сверка раз в 3 года и при смене критериев;
ежегодный мониторинг; отзыв заключения при несоответствии или непредставлении документов/фото/видео; обжалование в суде.

--- ГЛАВА 4. ЗАЯВКИ, ОЧЕРЕДЬ, ВЫПЛАТЫ ---
Приём заявок по срокам Приложения 2 независимо от наличия лимита бюджета; ЭЦП товаропроизводителя (п. 20).
МИО проверяет полноту за 2 раб. дня; одобренные — реестр/лист ожидания; очерёдность по дате-времени регистрации (п. 21).
Отказ — мотивированно, основания п. 23 и п. 9 Перечня; частичная выплата при нехватке средств (п. 24-1); перенос на след. год (п. 24-2).
Счета в «Казначейство-Клиент»; исправление реквизитов по заявлению.

--- ГЛАВА 5. ЖАЛОБЫ ---
Жалоба МИО — до 5 раб. дней; уполномоченный орган качества ГУ — до 15 (+10) раб. дней; досудебный порядок (АППК).

--- П. 25. МОНИТОРИНГ РОСХ ---
Ежеквартально в ГИСС раздел «Мониторинг исполнения обязательств»; при нарушении сохранности (кроме норм естественной убыли)
и целевого использования — уведомление о возврате; возврат в местный бюджет за 90 раб. дней; новая заявка после погашения задолженности.

--- ПРИЛОЖЕНИЕ 1: НОРМАТИВЫ (ключевые величины; уточнять редакцию № 428 и последующие) ---
МЯСНОЕ / МЯСО-МОЛОЧНОЕ КРС: бык-производитель отеч. 260 000 ₸/гол.; матка отеч. 260 000; СНГ/Украина 390 000; Австралия/Америка/Европа 525 000;
племенной молодняк мясного напр. 15 000 ₸/гол.; удешевление мужской особи на откорм/убой 300 ₸/кг ж.в.; говядина переработка 175 ₸/кг.
МОЛОЧНОЕ: матка отеч. 350 000; СНГ/Украина 390 000; импорт Австралия/Америка/Европа 700 000; молоко: ≥600 голов 45 ₸/кг; ≥400 — 30; ≥50 — 20;
СПК 20 ₸/кг; эмбрионы КРС 80 000 ₸/шт.
ОБЩЕЕ СКОТОВОДСТВО: ИО КРС 5 000 ₸/осеменённую голову; семя быка однополое 10 000 ₸/дозу, двуполое 5 000 ₸/дозу.
МЯСНОЕ ПТИЦЕВОДСТВО (актуальная таблица): племенной суточный молодняк родительской/прародительской формы 600 000 ₸/гол.;
удешевление мяса курицы: при производстве от 15 000 т — 80 ₸/кг; от 10 000 т — 70; от 5 000 т — 60; от 500 т — 50 (реализация/перемещение на свои перерабатывающие мощности или в цеха).
ЯИЧНОЕ: суточный молодняк финальной формы яичного направления от племенной птицы 60 000 ₸/гол.
ОВЦЕВОДСТВО: овцы отеч. 26 000; импорт матки 52 000; бараны-производители импорт 260 000; молодняк МРС 4 000; откорм МРС 3 000 и 7 000 (сезонные поставки);
эмбрионы 80 000; ИО овец 1 500 ₸; шерсть 200/150/25 ₸/кг по качеству.
КОНИ / ВЕРБЛЮДЫ / СВИНЬИ / корма и прочее по МИО — как в таблице Приложения 1.
Ограничения: субсидия на племенных животных не более 50% цены приобретения; производители — при наличии маточного поголовья
(исключения для племцентров и откормплощадок с арендой быков).

--- ПРИЛОЖЕНИЕ 2: КРИТЕРИИ (сжато, для экспертизы заявки) ---
БЫК-ПРОИЗВОДИТЕЛЬ: учётный номер (не кооп. — см. текст); земли СХ назначения (исключения для СПК из ЛПХ); ИСЖ/ИБСПР матки и покупки;
возраст быка 8–26 мес.; соотношение быков к маткам (вольная случка 1:20–30; докрытие при ИО 1:100); обязательство целевого использования
до двух случных сезонов подряд (не менее 18 мес.); срок заявки 20.01–20.12, в пределах 12 мес. с покупки.
МАТОЧНОЕ КРС: учётный номер; земли; ИБСПР/ИСЖ; возраст телок/нетелей (внутри РК и импорт — как в Правилах); обязательство ≥2 лет (кроме норм падежа); те же сроки подачи.
МОЛОКО: учётный номер; земли; ≥50 голов фуражного маточного поголовья (23/28 мес. племенные/товарные); реализация на переработку с учётным номером;
ежемесячно соматика в аккредитованных лабораториях (ИБСПР); положительное заключение спецкомиссии на МТФ; заявка в течение 6 мес. с оплаты за молоко.
ОВЦЫ/БАРАНЫ: учётный номер; земли; ИСЖ/ИБСПР; возраст 4–18 мес.; обязательства 2 года / 2 сезона бараны; соотношения баранов к маткам (20–30; ИО 1:300; докрытие 1:100).

--- ОТКАЗЫ (п. 22–23, Перечень) ---
Несоответствие Приложению 2; неполные/некорректные документы; нет регистрации в ГИСС; задолженность по возврату субсидий; иные основания Перечня.

--- РЕГИОНЫ РК ДЛЯ ПРОВЕРКИ АНКЕТЫ ---
Области: Акмолинская, Актюбинская, Алматинская, Атырауская, ВКО, Жамбылская, ЗКО, Карагандинская, Костанайская, Кызылординская,
Мангистауская, Павлодарская, СКО, Туркестанская; области Абай, Жетісу, Ұлытау; гг. Алматы, Астана, Шымкент.
"""

    raw        = app_data.get("raw_features_used") or {}
    compliance = app_data.get("compliance") or {}
    shap_pos   = app_data.get("top_positive_factors") or []
    shap_neg   = app_data.get("top_negative_factors") or []

    doc_chars = int(app_data.get("documents_text_chars") or 0)
    doc_ok    = bool(app_data.get("documents_extracted_ok"))
    pdf_n     = int(app_data.get("documents_pdf_count") or 0)

    if doc_ok and doc_chars > 0:
        doc_status_line = (
            f"Текст из PDF извлечён: {doc_chars} символов, файлов PDF: {pdf_n}. "
            "Compliance-анализ выполнялся по этому тексту."
        )
    elif pdf_n > 0 and not doc_ok:
        doc_status_line = (
            f"Было загружено PDF ({pdf_n} шт.), но машиночитаемый текст не извлечён "
            "(возможно сканы без OCR). Compliance по содержанию документов не выполнялся."
        )
    else:
        doc_status_line = (
            "PDF-документы к этой заявке в системе не анализировались (нет вложений или заявка из ГИСС/без файлов). "
            "Это не означает отсутствие «всех показателей» — числовые признаки ниже взяты из формы заявки и дефолтов модели."
        )

    if compliance:
        _cf = "; ".join(compliance.get("critical_failures", [])) or "не выявлены"
        _wrn = "; ".join(compliance.get("warnings", [])) or "нет"
        _dq = "; ".join(compliance.get("disqualifiers_found", [])) or "не найдены"
    else:
        _cf = _wrn = _dq = "— (анализ документов не выполнялся или данные не переданы)"

    _req_amt = app_data.get("requested_amount", 0)
    try:
        _req_amt_fmt = f"{float(_req_amt):,.0f} ₸"
    except (TypeError, ValueError):
        _req_amt_fmt = str(_req_amt)

    _zone_disp = (app_data.get("zone") or "—")
    if isinstance(_zone_disp, str) and _zone_disp not in ("—", ""):
        _zone_disp = _zone_disp.upper()

    app_summary = f"""
=== SMARTAGRO SCORE: КАК УСТРОЕН РАСЧЁТ (НЕ ПУТАТЬ СЛОИ) ===
1) XGBoost (балл ML / score_ml) — регрессия по 20 признакам после импутации пропусков; обучающий таргет в исторических данных —
   historical_score. Признаки подмешиваются из анкеты, regex по PDF и (если задан GEMINI_API_KEY на API) JSON-извлечения Gemini.
2) Документный слой (score_doc) — проверка соответствия Правилам по тексту PDF: семантические эмбеддинги (sentence-transformers)
   + при наличии ключа Gemini возможна досылка отдельных требований через LLM; итог в процентах overall_score_pct / doc_completeness.
3) Итоговый балл (score) = ml_weight_used * score_ml + doc_weight_used * score_doc (веса 0.30/0.70, 0.50/0.50 или 0.70/0.30
   в зависимости от полноты документов). Зона green/yellow/red по порогам 80 и 50 от этого итога.
4) SHAP — объяснение вклада признаков в предсказание XGBoost, не прямое «доказательство из PDF».
5) PDF: {doc_status_line}
6) Пустой compliance в этом запросе означает отсутствие блока проверки (нет текста или не передан контекст), а не автоматически
   отсутствие чисел для ML.

=== ДАННЫЕ ЗАЯВКИ ===
Предприятие: {app_data.get('company_name', '—')}
БИН/ИИН: {app_data.get('bin_iin', '—')}
Регион: {app_data.get('region', '—')}
Вид субсидии: {app_data.get('subsidy_type', '—')}
Направление: {app_data.get('direction', '—')}
Запрошенная сумма: {_req_amt_fmt}
Источник: {app_data.get('source_system', '—')}
Версия модели: {app_data.get('model_version', '—')}
Проверка экспертом (is_verified): {app_data.get('is_verified', '—')}
Правки эксперта (verified_payload): {'есть JSON' if app_data.get('verified_payload') else '—'}

ПОКАЗАТЕЛИ (raw_features_used → вход XGBoost после подготовки):
- Лет в работе: {raw.get('years_in_operation', '—')}
- Рост валовой продукции г/г: {raw.get('gross_output_growth_yoy', '—')}
- Долговая нагрузка (Долг/EBITDA): {raw.get('debt_load_ratio', '—')}
- Ветеринарное соответствие (0–1): {raw.get('veterinary_compliance', '—')}
- Сохранность поголовья (0–1), модельный прокси: {raw.get('historical_survival_rate', '—')} (сверяй с нормами естественного падежа приказа 3-3/1061)
- Доля племенного поголовья (0–1): {raw.get('pedigree_ratio', '—')}
- Зависимость от субсидий (0–1): {raw.get('subsidy_dependence_index', '—')}
- Земля на голову (га/гол.): {raw.get('land_to_livestock_ratio', '—')}
- Отклонение от норматива выпаса: {raw.get('grazing_norm_deviation', '—')}
- Риск естественной потери / климатический риск (модель): {raw.get('natural_loss_risk_score', '—')}
- Предыдущие субсидии (раз): {raw.get('previous_subsidies_count', '—')}
- Поголовье (расч.): {raw.get('livestock_count', '—')}
- Племенное / производитель (флаги): is_pedigree={raw.get('is_pedigree', '—')}, is_producer={raw.get('is_producer', '—')}

РЕЗУЛЬТАТЫ СКОРИНГА:
- Итоговый балл (score): {app_data.get('score', '—')} / 100
- ML (score_ml): {app_data.get('score_ml', '—')} / 100
- Документы (score_doc): {app_data.get('score_doc', '—')}
- Веса: ML {float(app_data.get('ml_weight_used') or 1.0):.0%} / документы {float(app_data.get('doc_weight_used') or 0.0):.0%}
- Зона: {_zone_disp}
- Ручная проверка (manual_review_required): {'да' if app_data.get('manual_review_required') else 'нет'}

Сверка суммы с нормативом (ориентир для эксперта): сопоставь requested_amount с «количество единиц × норматив из Приложения 1»
по заявленному виду субсидии (голова/кг/доза/шт.).

SHAP — положительные факторы: {'; '.join([f"{f['label']} ({f.get('shap_value',0):+.1f})" for f in shap_pos]) if shap_pos else 'не определены'}
SHAP — отрицательные факторы: {'; '.join([f"{f['label']} ({f.get('shap_value',0):+.1f})" for f in shap_neg]) if shap_neg else 'не определены'}

COMPLIANCE (чеклист по тексту PDF):
- Статус: {compliance.get('overall_status') if compliance else 'не передан / не проводился'}
- Выполнено (overall_score_pct): {compliance.get('overall_score_pct', '—') if compliance else '—'}%
- Полнота документов (doc_completeness): {compliance.get('doc_completeness', '—') if compliance else '—'}
- Критические нарушения: {_cf}
- Предупреждения: {_wrn}
- Дисквалификаторы: {_dq}

ОБУЧАЮЩИЙ КОНТУР (для справки эксперта): проверенные заявки помечаются is_verified=1 (POST /api/v1/decision или
/api/v1/applications/{{id}}/expert-verify); выборка GET /api/v1/training-samples; в БД колонки score_zone, final_score, verified_payload.
"""

    _years_val  = raw.get("years_in_operation")
    _years_note = (
        f"ВНИМАНИЕ: по данным модели years_in_operation = {_years_val} лет. "
        "Это значение получено из анкеты/PDF — не из заголовка и не из умолчания. "
        f"{'Предприятие ОПЫТНОЕ — запрещено называть его «молодым».' if (_years_val is not None and float(_years_val) >= 10) else ''}"
        if _years_val is not None else ""
    )

    system_prompt = f"""Ты — старший эксперт-аналитик Министерства сельского хозяйства РК.
Твоя задача: написать профессиональное экспертное заключение по заявке на субсидию.

{_years_note}

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ К ЗАКЛЮЧЕНИЮ:
1. Пиши строго на русском языке, официальным но понятным стилем — без markdown-символов (** ## и т.п.)
2. НИКОГДА не утверждай одновременно, что «все показатели отсутствуют», и при этом ссылайся на SHAP-числа — это логическое противоречие.
   SHAP-факторы всегда опираются на числовые признаки входа модели (они указаны в блоке ПОКАЗАТЕЛИ).
3. Если в запросе НЕТ непустого блока «ТЕКСТ ИЗ PDF» и compliance пустой — формулируй, что вложения не анализировались; не приравнивай это к отсутствию всех числовых данных по заявке.
4. Каждый вывод о несоответствии или риске по возможности подкрепляй ссылкой на правило Приказа № 108.
5. Проверяй ВСЕ следующие аспекты:
   а) Соответствие нормативам субсидий (не превышает ли запрошенная сумма допустимый расчётный максимум)
   б) Соответствие критериям Приложения 2 для конкретного вида субсидии
   в) Географию: является ли указанный регион реальной областью/городом РК
   г) Логику данных: противоречия между показателями (например, высокая долговая нагрузка + большая сумма запроса)
   д) Интерпретацию SHAP как вклада признаков модели (не как доказательства из PDF)
   е) Зависимость от субсидий и другие риски по числовым признакам
6. Структура ответа (строго):
   — Первый абзац: общий вывод (2–3 предложения)
   — Нумерованный список конкретных аргументов (минимум 5 пунктов, максимум 8)
   — Последний абзац: итоговая рекомендация с указанием возможности/невозможности выплаты
7. Длина: 250–400 слов
8. НЕ используй символы **, ##, *, — в начале строк и другой markdown-разметки
9. Если в запросе есть блок «ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ» с непустым содержимым — это реальное содержание файлов заявки.
   Обязан опереться на него: даты, номера, наименования, суммы. Запрещено писать, что документы «не читались» или «не проанализированы по тексту».
10. Если блока текста PDF нет или он пуст — честно укажи, что заключение только по анкетным данным и скорингу, без разбора вложений.
11. КРИТИЧНО — override compliance-ошибок при наличии факта в тексте PDF:
    а) Если compliance-чек ставит «Критическое нарушение» по учётному номеру хозяйства (БИН/ИИН),
       но в тексте PDF или в данных заявки явно виден БИН (12-значное число) — ты ОБЯЗАН написать,
       что требование фактически выполнено, и указать найденный БИН. Не делай этот флаг решающим приговором.
    б) Если compliance-чек ставит «не найдено» по земельным участкам, но в PDF/данных заявки
       указана площадь пастбищ (га/голову или Га), кадастровый номер или договор аренды земли —
       перекрой ошибку чекера и укажи, что земля подтверждена.
    в) Если compliance-чек «не найдено» по обязательствам, но в PDF есть «обязуюсь», «в течение 2 лет»,
       «целевое использование», «обязательство по целевому использованию» — доверяй PDF-тексту,
       явно это укажи и снизь вес этого флага в итоговом резюме."""

    _raw_doc = app_data.get("documents_extracted_text")
    _doc_full = (_raw_doc or "").strip()
    _nchars_full = len(_doc_full)
    try:
        _expert_doc_limit = max(8_000, int(os.getenv("GEMINI_EXPERT_MAX_DOC_CHARS", "42000")))
    except ValueError:
        _expert_doc_limit = 42_000
    _doc_txt = _doc_full
    _trunc_meta = ""
    if _nchars_full > _expert_doc_limit:
        _h = (_expert_doc_limit * 2) // 3
        _t = _expert_doc_limit - _h
        _doc_txt = (
            _doc_full[:_h]
            + "\n\n[… фрагмент опущен для лимита токенов; сохранены начало и конец PDF …]\n\n"
            + _doc_full[-_t:]
        )
        _trunc_meta = f" (~{_expert_doc_limit} симв. из {_nchars_full})"
    if len(_doc_txt) > 0:
        _storage_note = ""
        if _nchars_full >= 275_000:
            _storage_note = " (в БД мог храниться усечённый фрагмент)"
        doc_block = f"""
=== ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ (всего символов в системе: ~{_nchars_full}{_storage_note}{_trunc_meta}) ===
Внимание: ниже — содержимое загруженных файлов. Используй для проверки полноты пакета и фактов.

{_doc_txt}
=== КОНЕЦ ТЕКСТА PDF ===
"""
    else:
        doc_block = """
=== ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ ===
(отсутствует: заявка без загрузки PDF на эндпоинт с документами, либо из текста PDF ничего не извлеклось)
"""

    user_message = f"""{rules_summary}

{app_summary}

{doc_block}

Напиши экспертное заключение по данной заявке, строго опираясь на Правила субсидирования МСХ РК.
Если выше есть текст PDF — обязательно включи в аргументы отсылки к фактам из этого текста (что именно видно в документах)."""

    provider = expert_opinion_provider()

    if provider in ("openai", "gpt", "chatgpt"):
        try:
            return _expert_opinion_via_openai(system_prompt, user_message)
        except Exception as e:
            return f"Экспертное заключение (OpenAI) недоступно: {e}"

    if provider == "groq":
        try:
            return _expert_opinion_via_groq(system_prompt, user_message)
        except Exception as e:
            return f"Экспертное заключение (Groq) недоступно: {e}"

    gkey = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not gkey:
        return (
            "Экспертное заключение (Gemini) недоступно: задайте GEMINI_API_KEY в .env "
            "(корень проекта или frontend/.env) и перезапустите Streamlit."
        )

    import google.generativeai as genai
    genai.configure(api_key=gkey)

    model_name = (os.getenv("GEMINI_EXPERT_MODEL") or "gemini-2.0-flash").strip()
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_message)
            return response.text.strip()
        except Exception as e:
            last_exc = e
            if _is_gemini_quota_error(e) and attempt < 2:
                time.sleep(_gemini_429_retry_delay(e))
                continue
            break

    _tail = f"Экспертное заключение Gemini недоступно: {last_exc}"
    if last_exc is not None and _is_gemini_quota_error(last_exc):
        _tail += (
            "\n\nКвота Gemini исчерпана (429). Подождите или укажите другой GEMINI_API_KEY. "
            "Резерв Groq для экспертного заключения отключён: полный промпт не помещается в лимит TPM."
        )
    return _tail
