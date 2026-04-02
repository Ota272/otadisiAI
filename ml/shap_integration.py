
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path("models")

FEATURE_LABELS = {
    "gross_output_growth_yoy":    "Рост валовой продукции (г/г)",
    "land_to_livestock_ratio":    "Обеспеченность пастбищами (Га/голова)",
    "historical_survival_rate":   "Сохранность поголовья (%)",
    "subsidy_dependence_index":   "Индекс зависимости от субсидий",
    "veterinary_compliance":      "Ветеринарное соответствие",
    "years_in_operation":         "Стаж работы предприятия (лет)",
    "pedigree_ratio":             "Доля племенного поголовья",
    "previous_subsidies_count":   "Количество предыдущих субсидий",
    "debt_load_ratio":            "Долговая нагрузка (Долг/EBITDA)",
    "log_amount":                 "Масштаб заявки (log суммы)",
    "livestock_count":            "Количество голов скота",
    "direction_code":             "Направление животноводства",
    "is_pedigree":                "Племенное направление",
    "is_producer":                "Субсидия на производителей",
    "hour_submitted":             "Час подачи заявки",
    "month_submitted":            "Месяц подачи заявки",
    "region_encoded":             "Регион хозяйства",
}

DIRECTION_NAMES = {
    0: "скотоводство (КРС)",
    1: "овцеводство",
    2: "коневодство",
    3: "птицеводство",
    4: "верблюдоводство",
    5: "свиноводство",
    6: "прочее",
}

class ScoringEngine:

    def __init__(self, models_dir: Path = MODELS_DIR):
        print("🚀 Загружаю SmartAgro Scoring Engine...")

        self.model = joblib.load(models_dir / "xgb_scorer.joblib")

        self.scaler = joblib.load(models_dir / "scaler.joblib")

        self.explainer = joblib.load(models_dir / "shap_explainer.joblib")

        with open(models_dir / "feature_names.json", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        print(f"✅ Движок готов: {len(self.feature_names)} фичей, модель загружена")

    def score_farmer(
        self,
        raw_features: dict,
        llm_context: Optional[str] = None,
        *,
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
                shap_values, raw_features, n_factors=3
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

        zone, zone_label, recommendation = self._get_zone(score)

        verdict_text = self._generate_verdict(
            score, zone, top_positive, top_negative, llm_context
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

    def _prepare_feature_vector(self, raw: dict) -> pd.DataFrame:
        row = {}
        for feat in self.feature_names:
            val = raw.get(feat, 0.0)                                        
            row[feat] = float(val) if val is not None else 0.0

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
        n_factors: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        factors = []
        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            raw_val = raw_features.get(name, 0.0)
            label = FEATURE_LABELS.get(name, name)

            explanation_text = self._explain_feature(name, raw_val, float(shap_val))

            factors.append({
                "feature": name,
                "label": label,
                "shap_value": round(float(shap_val), 2),
                "raw_value": round(float(raw_val), 4) if isinstance(raw_val, (int, float)) else raw_val,
                "direction": "positive" if shap_val > 0 else "negative",
                "explanation": explanation_text,
                "impact_text": f"{'+'if shap_val>0 else ''}{shap_val:.1f} баллов: {explanation_text}",
            })

        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        positive = [f for f in factors if f["shap_value"] > 0][:n_factors]
        negative = [f for f in factors if f["shap_value"] < 0][:n_factors]

        return positive, negative

    def _explain_feature(self, name: str, value: float, shap_val: float) -> str:
        positive = shap_val > 0

        if name == "gross_output_growth_yoy":
            pct = value * 100
            if positive:
                return f"Рост валовой продукции {pct:+.1f}% — предприятие наращивает объёмы"
            else:
                return f"Спад валовой продукции {pct:+.1f}% — отрицательная динамика за год"

        elif name == "pedigree_ratio":
            pct = value * 100
            if positive:
                return f"Высокая доля племенного поголовья {pct:.0f}% — значительный племенной потенциал"
            else:
                return f"Низкая доля племенного поголовья {pct:.0f}% — слабая племенная база"

        elif name == "historical_survival_rate":
            pct = value * 100
            if positive:
                return f"Высокая сохранность стада {pct:.1f}% — хорошее ветеринарное управление"
            else:
                return f"Низкая сохранность стада {pct:.1f}% — повышенный падёж животных"

        elif name == "subsidy_dependence_index":
            pct = value * 100
            if positive:
                return f"Умеренная зависимость от субсидий {pct:.0f}% — экономически самостоятельно"
            else:
                return f"Высокая зависимость от субсидий {pct:.0f}% — бизнес держится на дотациях"

        elif name == "veterinary_compliance":
            pct = value * 100
            if positive:
                return f"Высокое ветеринарное соответствие {pct:.0f}% — все нормы соблюдены"
            else:
                return f"Нарушения ветеринарных норм (соответствие {pct:.0f}%)"

        elif name == "debt_load_ratio":
            if positive:
                return f"Низкая долговая нагрузка (Долг/EBITDA = {value:.1f}) — финансово устойчиво"
            else:
                return f"Высокая долговая нагрузка (Долг/EBITDA = {value:.1f}) — риск платёжеспособности"

        elif name == "years_in_operation":
            if positive:
                return f"Опытное предприятие ({value:.0f} лет работы) — проверенная история"
            else:
                return f"Молодое предприятие ({value:.0f} лет) — ограниченная история деятельности"

        elif name == "land_to_livestock_ratio":
            if positive:
                return f"Хорошая обеспеченность пастбищами ({value:.1f} Га/голову)"
            else:
                return f"Низкая обеспеченность землёй ({value:.1f} Га/голову) — перегруженность пастбищ"

        elif name == "previous_subsidies_count":
            if positive:
                return f"Успешная история субсидирования ({value:.0f} предыдущих субсидий)"
            else:
                return f"Нет истории получения субсидий — новый участник программы"

        elif name == "livestock_count":
            if positive:
                return f"Крупное хозяйство ({value:.0f} голов) — значимый масштаб производства"
            else:
                return f"Небольшое хозяйство ({value:.0f} голов) — ограниченный масштаб"

        elif name == "is_pedigree":
            if value == 1:
                return "Субсидия на племенное поголовье — стратегическое направление"
            else:
                return "Субсидия на товарное производство"

        else:
            direction_word = "положительно" if positive else "отрицательно"
            return f"Показатель '{FEATURE_LABELS.get(name, name)}' влияет {direction_word} на оценку"

    def _get_zone(self, score: float) -> tuple[str, str, str]:
        if score >= 80:
            return (
                "green",
                "Зелёная зона (80–100)",
                "Строго рекомендовано к включению в шорт-лист"
            )
        elif score >= 50:
            return (
                "yellow",
                "Жёлтая зона (50–79)",
                "Рекомендуется дополнительное рассмотрение комиссией"
            )
        else:
            return (
                "red",
                "Красная зона (0–49)",
                "Не рекомендовано — выявлены существенные риски"
            )

    def _generate_verdict(
        self,
        score: float,
        zone: str,
        top_positive: list,
        top_negative: list,
        llm_context: Optional[str],
    ) -> str:
        verdict_parts = []

        zone_intros = {
            "green": f"Предприятие получило высокий балл {score:.0f}/100 и рекомендуется к приоритетному рассмотрению.",
            "yellow": f"Предприятие получило балл {score:.0f}/100. Рекомендуется детальное рассмотрение комиссией.",
            "red":    f"Предприятие получило балл {score:.0f}/100. Система выявила существенные риски.",
        }
        verdict_parts.append(zone_intros[zone])

        if top_positive:
            verdict_parts.append("\n✅ Сильные стороны:")
            for factor in top_positive:
                verdict_parts.append(f"  • {factor['explanation']}")

        if top_negative:
            verdict_parts.append("\n⚠️ Факторы риска:")
            for factor in top_negative:
                verdict_parts.append(f"  • {factor['explanation']}")

        if llm_context:
            verdict_parts.append(f"\n📄 Данные из документов:\n  {llm_context}")

        verdict_parts.append(
            "\n⚖️ Данная оценка является рекомендацией ИИ-системы. "
            "Окончательное решение принимается уполномоченной комиссией "
            "Министерства сельского хозяйства РК."
        )

        return "\n".join(verdict_parts)

def extract_features_from_documents(documents_text: str, api_key: str) -> dict:
    import google.generativeai as genai

    client = genai.configure(api_key=api_key)

    system_prompt = """Ты — аналитик Министерства сельского хозяйства РК.
Твоя задача: извлечь структурированные факты из документов сельхозпредприятия
для системы скоринга субсидий.

Отвечай ТОЛЬКО валидным JSON без markdown-блоков, пояснений и вводных слов.

Формат ответа:
{
  "livestock_count": <число голов или null>,
  "breed_quality_score": <0.0-1.0: 1.0=элитная племенная, 0.5=товарная, 0.2=беспородная>,
  "has_vet_passport": <true/false>,
  "vaccination_current": <true/false: вакцинация в последние 6 месяцев>,
  "veterinary_compliance": <0.0-1.0 на основе документов>,
  "land_area_ha": <площадь в гектарах или null>,
  "has_lease_agreement": <true/false>,
  "pedigree_ratio": <0.0-1.0 доля племенных животных или null>,
  "years_in_operation": <число лет или null>,
  "has_tax_certificate": <true/false>,
  "company_type": <"ТОО"/"ИП"/"КХ"/"КФХ"/"другое">,
  "llm_summary": "<1-2 предложения: ключевые выводы по документам>",
  "document_completeness": <0.0-1.0: насколько полный пакет документов>
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
        message = model.generate_content(user_message)

        response_text = message.text.strip()

        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = json.loads(response_text)

        llm_summary = extracted.pop("llm_summary", None)

        for key in ["has_vet_passport", "vaccination_current", "has_lease_agreement",
                    "has_tax_certificate"]:
            if key in extracted and isinstance(extracted[key], bool):
                extracted[key] = 1.0 if extracted[key] else 0.0

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
1) Числовые признаки (блок «ПОКАЗАТЕЛИ ПРЕДПРИЯТИЯ») — это входные данные в модель XGBoost: из анкеты веб-формы, опциональных полей, 
   дефолтов при «не знаю», и при наличии — извлечённые из PDF числа. Это НЕ «воздух» и НЕ выдумка LLM.
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

    system_prompt = """Ты — старший эксперт-аналитик Министерства сельского хозяйства РК.
Твоя задача: написать профессиональное экспертное заключение по заявке на субсидию.

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
10. Если блока текста PDF нет или он пуст — честно укажи, что заключение только по анкетным данным и скорингу, без разбора вложений."""

    _raw_doc = app_data.get("documents_extracted_text")
    _doc_txt = (_raw_doc or "").strip()
    if len(_doc_txt) > 0:
        _nchars = len(_doc_txt)
        _trunc_note = ""
        if _nchars >= 275_000:
            _trunc_note = " (показан фрагмент до лимита хранения)"
        doc_block = f"""
=== ТЕКСТ ИЗ PDF-ДОКУМЕНТОВ (извлечён при скоринге, {_nchars} симв.{_trunc_note}) ===
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

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_message)
        return response.text.strip()
    except Exception as e:
        return f"Экспертное заключение Gemini недоступно: {e}"
