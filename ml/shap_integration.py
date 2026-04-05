
import json
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
        "veterinary_compliancяe":      "Ветеринарное соответствие",
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
    ) -> dict:

        X = self._prepare_feature_vector(raw_features)

        X_scaled = self._scale(X)

        raw_score = float(self.model.predict(X_scaled)[0])
        score = round(float(np.clip(raw_score, 1.0, 100.0)), 1)

        if include_shap:
            shap_values = self._compute_shap(X_scaled)
            shap_base = self._get_shap_base_value(X_scaled)
            top_positive, top_negative = self._build_explanations(
                shap_values, raw_features, lang=lang, n_factors=3
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
    ) -> tuple[list[dict], list[dict]]:
        factors = []
        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            raw_val = raw_features.get(name, 0.0)
            if raw_val is None:
                raw_val = 0.0
            label = FEATURE_LABELS[lang].get(name, name)

            explanation_text = self._explain_feature(name, raw_val, float(shap_val), lang=lang)

            factors.append({
                "feature": name,
                "label": label,
                "shap_value": round(float(shap_val), 2),
                "raw_value": round(float(raw_val), 4) if isinstance(raw_val, (int, float)) else raw_val,
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
                "default": lambda lbl, d: f"Показатель '{lbl}' влияет {d} на оценку",
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
                "default": lambda lbl, d: f"'{lbl}' көрсеткіші {d} әсер етеді",
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


def extract_features_from_documents(documents_text: str, api_key: str) -> dict:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    genai.configure(api_key=api_key)

    system_prompt = """Ты — аналитик Министерства сельского хозяйства РК.
Твоя задача: извлечь структурированные факты из документов сельхозпредприятия
для системы скоринга субсидий. Числа для ML должны совпадать по смыслу с полями модели.

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

    user_message = f"""Проанализируй следующие документы сельхозпредприятия и извлеки данные:

---НАЧАЛО ДОКУМЕНТОВ---
{documents_text}
---КОНЕЦ ДОКУМЕНТОВ---

Верни только JSON с извлечёнными данными. Если данных нет — ставь null."""

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
        response_text = _strip_markdown_json_fence((message.text or "").strip())
        if not response_text:
            return {
                "features": {},
                "llm_summary": None,
                "extraction_status": "empty_model_response",
            }

        try:
            extracted = json.loads(response_text)
        except json.JSONDecodeError:
            import re

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
            print(f"[extract_features_from_documents] Пропускаю неизвестные ключи Gemini: {unknown}")
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
    except Exception as e:
        return {
            "features": {},
            "llm_summary": None,
            "extraction_status": f"api_error: {e}",
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

def generate_gemini_expert_opinion(app_data: dict, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    rules_summary = """
=== ПРАВИЛА СУБСИДИРОВАНИЯ МСХ РК ===
Источник: Приказ Министра сельского хозяйства РК № 108 от 15.03.2019
Редакция: Приказ № 332 от 18.09.2023 (введён с 01.01.2024)

--- ПРИЛОЖЕНИЕ 1: НОРМАТИВЫ СУБСИДИЙ ---

МЯСНОЕ И МЯСО-МОЛОЧНОЕ СКОТОВОДСТВО:
- Племенной бык-производитель (отечественный): 260 000 ₸/голову
- Племенное маточное поголовье КРС (отечественное): 260 000 ₸/голову
- Племенное маточное поголовье КРС (из стран СНГ, Украины): 390 000 ₸/голову
- Племенное маточное поголовье КРС (из Австралии, Америки, Европы): 525 000 ₸/голову
- Выращивание племенного молодняка КРС мясного направления: 15 000 ₸/голову
- Удешевление КРС мужской особи на откорм/убой: 300 ₸/кг живого веса
- Удешевление реализованной говядины: 175 ₸/кг

МОЛОЧНОЕ И МОЛОЧНО-МЯСНОЕ СКОТОВОДСТВО:
- Племенное маточное поголовье (отечественное): 350 000 ₸/голову
- Племенное маточное поголовье (из СНГ, Украины): 390 000 ₸/голову
- Племенное маточное поголовье (из Австралии, Америки, Европы): 700 000 ₸/голову
- Удешевление производства молока, от 600 голов: 45 ₸/кг реализованного молока
- Удешевление производства молока, от 400 голов: 30 ₸/кг
- Удешевление производства молока, от 50 голов: 20 ₸/кг
- СПК (сельхозкооператив): 20 ₸/кг
- Эмбрионы КРС: 80 000 ₸/штуку

СКОТОВОДСТВО (общее):
- Искусственное осеменение КРС: 5 000 ₸/осеменённую голову
- Семя племенного быка (однополое): 10 000 ₸/дозу
- Семя племенного быка (двуполое): 5 000 ₸/дозу

ОВЦЕВОДСТВО:
- Отечественные племенные овцы: 26 000 ₸/голову
- Импортные племенные маточные овцы: 52 000 ₸/голову
- Импортные племенные бараны-производители: 260 000 ₸/голову
- Выращивание племенного молодняка МРС: 4 000 ₸/голову
- МРС мужской особи на откорм/убой: 3 000 ₸/голову
- МРС мужской особи сезонные поставки: 7 000 ₸/голову
- Эмбрионы овец: 80 000 ₸/штуку
- Искусственное осеменение овец: 1 500 ₸/осеменённую голову
- Тонкая/полутонкая шерсть (60 качество): 200 ₸/кг
- Тонкая/полутонкая шерсть (50 качество): 150 ₸/кг
- Грубая/полугрубая шерсть: 25 ₸/кг

КОНЕВОДСТВО:
- Ведение селекционной и племенной работы: 20 000 ₸/голову в год
- Племенные жеребцы-производители продуктивного направления: 175 000 ₸/голову

ВЕРБЛЮДОВОДСТВО:
- Племенные верблюды-производители: 175 000 ₸/голову

СВИНОВОДСТВО:
- Племенные свиньи: 100 000 ₸/голову
- Удешевление свиней на убой: 2 000 ₸/голову

ПО РЕШЕНИЮ МИО (местных исполнительных органов):
- Удешевление затрат на корма: устанавливается МИО
- Племенное маточное поголовье коз: 70 000 ₸/голову
- Кобылье молоко: 60 ₸/кг
- Верблюжье молоко: 55 ₸/кг (возможно увеличение до 190 ₸/кг при доп. финансировании)
- Мёд: 200 ₸/кг

ОГРАНИЧЕНИЯ (п. Приложения 1):
- Субсидия не более 50% от стоимости приобретения племенных животных
- Субсидирование племенных производителей — только при наличии маточного поголовья у товаропроизводителя (кроме племцентров и откормплощадок, передающих быков в аренду)

--- ПРИЛОЖЕНИЕ 2: КРИТЕРИИ К ТОВАРОПРОИЗВОДИТЕЛЯМ ---

КРС МЯСНОЕ/МЯСО-МОЛОЧНОЕ — Приобретение быка-производителя:
- Критерий 1: Наличие учётного номера хозяйства (кроме с/х кооперативов)
- Критерий 2: Наличие земель сельскохозяйственного назначения (кроме СПК из личных подсобных хозяйств)
- Критерий 3: Регистрация маточного поголовья в ИСЖ и ИБСПР на момент подачи заявки
- Критерий 4: Регистрация приобретённого поголовья в ИСЖ и ИБСПР на момент подачи заявки
- Критерий 5: Возраст быка на дату продажи (по племенному свидетельству) — 8–26 месяцев включительно
- Критерий 6: Соотношение быков к маткам: вольная случка — 1 бык на 20–30 маток; докрытие при ИО — 1 бык на 100 маток
- Критерий 7: Обязательство по целевому использованию — не более двух случных сезонов подряд (не менее 18 месяцев)
- Срок подачи заявки: с 20 января до 20 декабря текущего года (в течение 12 месяцев с момента приобретения)

КРС МЯСНОЕ/МОЛОЧНОЕ — Приобретение племенного маточного поголовья:
- Критерий 1: Наличие учётного номера хозяйства
- Критерий 2: Наличие земель сельскохозяйственного назначения
- Критерий 3: Регистрация приобретённого поголовья в ИБСПР и ИСЖ на момент подачи заявки
- Критерий 4: Возраст при приобретении внутри страны (по племенному свидетельству); при импорте — на момент карантина у экспортера: телки — 6–18 месяцев включительно; нетели — 13–26 месяцев включительно
- Критерий 5: Обязательство по целевому использованию — не менее 2 лет (кроме норм естественного падежа)
- Срок подачи: с 20 января до 20 декабря (в течение 12 месяцев с момента приобретения)

МОЛОКО — Удешевление стоимости производства:
- Критерий 1: Наличие учётного номера
- Критерий 2: Наличие земель сельскохозяйственного назначения
- Критерий 3: Регистрация в ИСЖ не менее 50 голов фуражного маточного поголовья в возрасте от 23 месяцев (племенные) / от 28 месяцев (товарные)
- Критерий 4: Реализация молока на молокоперерабатывающее предприятие или цех, имеющий учётный номер
- Критерий 5: Ежемесячные исследования проб молока на соматические клетки в аккредитованных лабораториях (результаты в ИБСПР)
- Критерий 6: Положительное заключение специальной комиссии на молочно-товарную ферму
- Срок подачи: с 20 января до 20 декабря (в течение 6 месяцев с момента оплаты за молоко)

ОВЦЕВОДСТВО — Приобретение племенных овец и баранов-производителей:
- Критерий 1: Наличие учётного номера (кроме СПК при приобретении баранов)
- Критерий 2: Наличие земель сельскохозяйственного назначения
- Критерий 3: Регистрация приобретённого поголовья в ИСЖ и ИБСПР
- Критерий 4: Возраст при приобретении (внутри страны по племенному свидетельству; при импорте — на момент карантина): бараны и матки — 4–18 месяцев включительно
- Критерий 5: Обязательство: маточное поголовье — не менее 2 лет; бараны — не менее 2 случных сезонов подряд (≥18 месяцев)
- Критерий 6: Соотношение баранов к маткам: вольная/ручная случка — 1 баран на 20–30 маток; ИО — 1 баран на 300 маток; докрытие при ИО — 1 баран на 100 маток

--- ОБЩИЕ УСЛОВИЯ (Глава 3, пп. 11–14) ---
- Получатели: физические и юридические лица, занимающиеся производством с/х продукции; племенные и дистрибьютерные центры; техники-осеменаторы
- Регистрация в ГИСС с использованием ЭЦП обязательна (п. 12–13)
- Обязательные сведения для регистрации: БИН/ИИН, наименование, ФИО руководителя, контактные данные, реквизиты текущего счёта банка второго уровня (ИИК, БИК)
- Подтверждение приобретения — через ЭСФ (ИС ЭСФ), при импорте — таможенная декларация или заявление о ввозе (п. 16)
- Встречные обязательства: рост/сохранение объёма валовой продукции АПК (п. 14-1), обязательны при субсидиях от 100 млн ₸ и более в текущем году
- Мониторинг сохранности просубсидированного поголовья — ежеквартально РОСХ (п. 25)
- При нарушении обязательств — возврат субсидий в местный бюджет в течение 90 рабочих дней

--- ОСНОВАНИЯ ДЛЯ ОТКАЗА (п. 23, Перечень п. 9) ---
- Несоответствие заявки критериям Приложения 2
- Неполнота или некорректность документов
- Отсутствие регистрации в ГИСС
- Наличие задолженности по возврату ранее полученных субсидий

--- ОБЛАСТИ И ГОРОДА РЕСПУБЛИКАНСКОГО ЗНАЧЕНИЯ РК (14 областей + 3 города) ---
Акмолинская, Актюбинская, Алматинская, Атырауская, Восточно-Казахстанская,
Жамбылская, Западно-Казахстанская, Карагандинская, Костанайская, Кызылординская,
Мангистауская, Павлодарская, Северо-Казахстанская, Туркестанская,
область Абай, Жетысуская область, Улытауская область,
г. Алматы, г. Астана, г. Шымкент
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
        _cf = _wrn = _dq = "— (анализ документов не выполнялся)"

    app_summary = f"""
=== КРИТИЧЕСКИ ВАЖНО: ИСТОЧНИКИ ДАННЫХ (НЕ ПРОТИВОРЕЧЬ ЭТОМУ) ===
1) Числовые признаки (блок «ПОКАЗАТЕЛИ ПРЕДПРИЯТИЯ») — фактический вход XGBoost после подстановки чисел из PDF (JSON + разбор текста) поверх анкеты.
   При явном противоречии с дословной цитатой из PDF сначала считай возможной устаревшей записью в интерфейсе и запроси пересчёт заявки.
2) SHAP (+/- баллы) — математическое объяснение вклада каждого признака в итоговый балл модели (SHAP TreeExplainer).
   SHAP описывает модель, а не дословную цитату из PDF. Не пиши, что «все показатели отсутствуют», если блок ПОКАЗАТЕЛИ заполнен числами.
3) Документы PDF: {doc_status_line}
4) Если compliance-пустой или статус «документы не загружались» — это значит только отсутствие успешного анализа вложений, 
   а НЕ отсутствие числовых данных для ML-скоринга. Не смешивай эти два утверждения в одном абзаце.

=== ДАННЫЕ АНАЛИЗИРУЕМОЙ ЗАЯВКИ ===
Предприятие: {app_data.get('company_name', '—')}
БИН/ИИН: {app_data.get('bin_iin', '—')}
Регион: {app_data.get('region', '—')}
Вид субсидии: {app_data.get('subsidy_type', '—')}
Направление: {app_data.get('direction', '—')}
Запрошено: {app_data.get('requested_amount', 0):,.0f} ₸
Источник заявки: {app_data.get('source_system', '—')}

ПОКАЗАТЕЛИ ПРЕДПРИЯТИЯ (вход модели XGBoost):
- Лет в работе: {raw.get('years_in_operation', '—')}
- Рост валовой продукции г/г: {raw.get('gross_output_growth_yoy', '—')}
- Долговая нагрузка (Долг/EBITDA): {raw.get('debt_load_ratio', '—')}
- Ветеринарное соответствие (0–1): {raw.get('veterinary_compliance', '—')}
- Сохранность поголовья (0–1): {raw.get('historical_survival_rate', '—')}
- Доля племенного поголовья (0–1): {raw.get('pedigree_ratio', '—')}
- Зависимость от субсидий (0–1): {raw.get('subsidy_dependence_index', '—')}
- Обеспеченность землёй (га/голову): {raw.get('land_to_livestock_ratio', '—')}
- Количество предыдущих субсидий: {raw.get('previous_subsidies_count', '—')}
- Количество голов (расч.): {raw.get('livestock_count', '—')}

РЕЗУЛЬТАТЫ СКОРИНГОВОЙ МОДЕЛИ:
- Итоговый балл (итог): {app_data.get('score', '—')} / 100
- Балл XGBoost (ML): {app_data.get('score_ml', '—')} / 100
- Балл документов (Gemini): {app_data.get('score_doc', 'не загружались')}
- Веса: ML {app_data.get('ml_weight_used', 1.0):.0%} / Документы {app_data.get('doc_weight_used', 0.0):.0%}
- Зона: {app_data.get('zone', '—').upper() if app_data.get('zone') else '—'}
- Требуется ручная проверка: {'да' if app_data.get('manual_review_required') else 'нет'}

SHAP — положительные факторы: {'; '.join([f"{f['label']} ({f.get('shap_value',0):+.1f})" for f in shap_pos]) if shap_pos else 'не определены'}
SHAP — отрицательные факторы: {'; '.join([f"{f['label']} ({f.get('shap_value',0):+.1f})" for f in shap_neg]) if shap_neg else 'не определены'}

COMPLIANCE-ПРОВЕРКА ДОКУМЕНТОВ (только если был извлечён текст из PDF):
- Статус: {compliance.get('overall_status') if compliance else 'не проводилась — нет текста из PDF'}
- Выполнено требований: {compliance.get('overall_score_pct', '—') if compliance else '—'}%
- Полнота документов: {compliance.get('doc_completeness', '—') if compliance else '—'}
- Критические нарушения: {_cf}
- Предупреждения: {_wrn}
- Дисквалифицирующие условия: {_dq}
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
6. Структура ответа (строго JSON, БЕЗ markdown-блоков):
   {{
     "conclusion": "<текст заключения: 250-400 слов, обычный текст без markdown>",
     "citations": [
       {{
         "point_number": 1,
         "quote": "<точная цитата из PDF документа, подтверждающая или опровергающая пункт>",
         "line_number": <номер строки в PDF тексте, откуда взята цитата>,
         "explanation": "<краткое пояснение, почему это важно>"
       }}
     ]
   }}
   
   КРИТИЧНО ДЛЯ CITATIONS:
   — В поле "conclusion" после каждого пункта/аргумента вставь маркер вида [CITATION:N] где N - номер пункта
   — Пример: "1. Заявка превышает норматив субсидии. [CITATION:1] Сумма завышена на 25%."
   — Для КАЖДОГО из 5-8 пунктов заключения ДОЛЖЕН быть маркер [CITATION:N]
   — В массив "citations" для каждого point_number добавь соответствующую цитату из PDF
   — Если PDF нет, citations может быть пустым, но conclusion всё равно должен быть написан
   — quote: точная выдержка из текста PDF (1-3 предложения)
   — line_number: номер строки в блоке "ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ", где начинается цитата (считая от 1)
   — Максимум 3 цитаты на один пункт (выбирай самые важные)
   
7. Длина conclusion: 250–400 слов
8. В conclusion НЕ используй символы **, ##, *, — в начале строк и другой markdown-разметки
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
    _doc_txt = (_raw_doc or "").strip()
    if len(_doc_txt) > 0:
        _nchars = len(_doc_txt)
        _trunc_note = ""
        if _nchars >= 275_000:
            _trunc_note = " (показан фрагмент до лимита хранения)"
        
        # Добавляем номера строк к тексту PDF для точного цитирования
        lines = _doc_txt.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"[{i:4d}] {line}")
        numbered_text = '\n'.join(numbered_lines)
        
        doc_block = f"""
=== ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ (извлечён при скоринге, {_nchars} симв.{_trunc_note}) ===
Внимание: ниже — содержимое загруженных файлов. Используй для проверки полноты пакета и фактов.
ВАЖНО: Каждая строка имеет номер в квадратных скобках [  1], [  2] и т.д.
При цитировании указывай номер строки, где начинается цитата.

{numbered_text}
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
Если выше есть текст PDF — обязательно включи в аргументы отсылки к фактам из этого текста (что именно видно в документах).

КРИТИЧНО: Верни ответ ТОЛЬКО в формате JSON с полями "conclusion" и "citations".
Каждый пункт заключения должен иметь минимум одну цитату из PDF с точным указанием номера строки."""

    try:
        from google.generativeai.types import GenerationConfig
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
        )
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        response = model.generate_content(
            user_message,
            generation_config=generation_config,
        )
        
        # Парсим JSON ответ
        response_text = _strip_markdown_json_fence((response.text or "").strip())
        if not response_text:
            return "Экспертное заключение: пустой ответ от модели."
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return f"Экспертное заключение (ошибка парсинга JSON): {response_text[:500]}"
            result = json.loads(json_match.group())
        
        if not isinstance(result, dict):
            return "Экспертное заключение: неверный формат ответа (не объект)."
        
        conclusion = result.get("conclusion", "")
        citations = result.get("citations", [])
        
        # Формируем ответ с маркерами для цитат
        if not citations:
            return conclusion or "Экспертное заключение: нет данных для анализа."
        
        # Встраиваем маркеры цитат в текст заключения
        # Формат: [CITATION:point_number] будет заменён на иконку скрепки на фронтенде
        enriched_conclusion = conclusion
        
        # Добавляем информацию о цитатах в конец для фронтенда
        result["enriched_conclusion"] = enriched_conclusion
        result["citations_list"] = citations
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Экспертное заключение Gemini недоступно: {e}"
