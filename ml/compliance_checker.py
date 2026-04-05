
import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL_PATH = _REPO_ROOT / "models" / "embed_model"


_OBLIGATION_REGEX = re.compile(
    r"(?:обязу[юесмьь]+|обязательств[оа]|обязуемся|сохранность)"
    r".{0,200}?"
    r"(?:[23]\s*год[а-я]*|два\s+года|трёх?\s+лет|двух\s+лет|не\s+менее\s+(?:двух?|2|3)\s+лет|\d+\s*\([а-яА-Я]+\)\s*лет)",
    re.IGNORECASE | re.DOTALL,
)

_OBLIGATION_KW_REGEX = re.compile(
    r"целевое\s+использование|воспроизводств[аео]|не\s+менее\s+(?:двух?|2|3)\s+(?:лет|года)|"
    r"обязуюсь|обязательств|обязуемся|сохранность",
    re.IGNORECASE,
)

SUBSIDY_RULES = {

    "КРС_маточное": {
        "name": "Приобретение племенного маточного поголовья КРС",
        "normative_tenge": {"domestic": 260_000, "cis": 390_000, "foreign": 700_000},
        "max_subsidy_pct": 50,                             
        "requirements": [
            {
                "id": "R-01",
                "text": "Наличие учётного номера хозяйства (за исключением с/х кооператива)",
                "keywords": ["учетный номер", "учётный номер", "номер хозяйства"],
                "critical": True,
                "source": "Приложение 2, п.1",
            },
            {
                "id": "R-02",
                "text": "Наличие земель сельскохозяйственного назначения",
                "keywords": ["земельн", "кадастр", "гектар", "сельскохозяйственного назначения",
                             "пастбищ", "сенокос"],
                "critical": True,
                "source": "Приложение 2, п.2",
            },
            {
                "id": "R-03",
                "text": "Наличие регистрации приобретённого поголовья в ИСЖ и ИБСПР",
                "keywords": ["ИСЖ", "ИБСПР", "регистрация поголовья", "идентификация животных",
                             "ушная бирка", "бирка"],
                "critical": True,
                "source": "Приложение 2, п.3",
            },
            {
                "id": "R-04",
                "text": "Возраст приобретённого поголовья: телки 6–18 мес., нетели 13–26 мес.",
                "keywords": ["возраст", "месяц", "телк", "нетел", "племенное свидетельство"],
                "critical": True,
                "source": "Приложение 2, п.4",
            },
            {
                "id": "R-05",
                "text": "Принятие обязательства по целевому использованию не менее 2 лет",
                "keywords": ["обязательств", "два года", "2 года", "целевое использование",
                             "воспроизводств", "не менее двух лет", "сохранность"],
                "critical": True,
                "source": "Приложение 2, п.5; п.14-1 Правил",
            },
            {
                "id": "R-06",
                "text": "Акт или ЭСФ на приобретение племенного скота с указанием породы",
                "keywords": ["акт", "ЭСФ", "счет-фактура", "приобретение", "порода",
                             "племенное свидетельство", "договор купли"],
                "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-07",
                "text": "Справка о ветеринарном благополучии хозяйства",
                "keywords": ["ветеринар", "благополучи", "ветсправка", "карантин",
                             "эпизоотич", "ветеринарного"],
                "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-08",
                "text": "Банковские реквизиты (ИИК, БИК, КБе)",
                "keywords": ["ИИК", "БИК", "КБе", "банковские реквизиты", "расчётный счёт",
                             "счет в банке", "банк"],
                "critical": True,
                "source": "Заявка, п.4",
            },
            {
                "id": "R-09",
                "text": "Сведения о земельных участках (кадастровый номер, площадь)",
                "keywords": ["кадастровый номер", "площадь", "гектар", "Га", "земельный участок"],
                "critical": False,
                "source": "Заявка, п.6",
            },
            {
                "id": "R-10",
                "text": "Справка о налоговом учёте (БИН/ИИН)",
                "keywords": ["БИН", "ИИН", "налоговый учёт", "налоговая", "регистрация",
                             "свидетельство о постановке"],
                "critical": False,
                "source": "Учредительные документы",
            },
        ],
        "disqualifiers": [
            "приобретены по бартеру",
            "использованы не для воспроизводства",
            "ранее просубсидированы на удешевление",
            "реализованы за пределы РК",
        ],
    },

    "КРС_быки": {
        "name": "Приобретение племенных быков-производителей",
        "normative_tenge": {"domestic": 260_000},
        "max_subsidy_pct": 50,
        "requirements": [
            {
                "id": "R-01", "text": "Наличие учётного номера хозяйства",
                "keywords": ["учетный номер", "учётный номер"], "critical": True,
                "source": "Приложение 2, п.1",
            },
            {
                "id": "R-02", "text": "Наличие земель сельскохозяйственного назначения",
                "keywords": ["земельн", "гектар", "пастбищ"], "critical": True,
                "source": "Приложение 2, п.2",
            },
            {
                "id": "R-03", "text": "Наличие маточного поголовья у товаропроизводителя",
                "keywords": ["маточное поголовье", "коров", "тёлок", "коровы"], "critical": True,
                "source": "Приложение 1, примечание",
            },
            {
                "id": "R-04",
                "text": "Возраст приобретённых быков: 8–26 месяцев на дату продажи",
                "keywords": ["возраст", "месяц", "племенное свидетельство", "бык"], "critical": True,
                "source": "Приложение 2, п.5",
            },
            {
                "id": "R-05",
                "text": "Соотношение: 1 бык на 20–25 маток (не более 5% от маточного стада)",
                "keywords": ["соотношение", "маточное поголовье", "голов"], "critical": False,
                "source": "Приложение 2, п.6",
            },
            {
                "id": "R-06", "text": "Акт или ЭСФ на приобретение с указанием породы и цены",
                "keywords": ["акт", "ЭСФ", "счет-фактура", "порода", "цена"], "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-07", "text": "Справка о ветеринарном благополучии хозяйства",
                "keywords": ["ветеринар", "благополучи", "эпизоотич"], "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-08",
                "text": "Обязательство по целевому использованию быка не менее 2 лет",
                "keywords": ["обязательств", "два года", "2 года", "целевое использование"],
                "critical": True, "source": "Приложение 2, п.7",
            },
        ],
        "disqualifiers": [
            "приобретены по бартеру",
            "ранее просубсидированы",
        ],
    },

    "овцы_бараны": {
        "name": "Приобретение племенных баранов-производителей",
        "normative_tenge": {"per_head": 100_000},
        "max_subsidy_pct": 50,
        "requirements": [
            {
                "id": "R-01", "text": "Наличие учётного номера хозяйства",
                "keywords": ["учетный номер"], "critical": True,
                "source": "Приложение 2",
            },
            {
                "id": "R-02", "text": "Наличие маточного поголовья овец у заявителя",
                "keywords": ["матки", "маточное поголовье", "овцематки", "овец"], "critical": True,
                "source": "Приложение 1, примечание",
            },
            {
                "id": "R-03",
                "text": "Регистрация поголовья в ИСЖ и ИБСПР",
                "keywords": ["ИСЖ", "ИБСПР", "регистрация", "бирка"], "critical": True,
                "source": "Приложение 2",
            },
            {
                "id": "R-04", "text": "Акт или ЭСФ на приобретение",
                "keywords": ["акт", "ЭСФ", "счет-фактура"], "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-05", "text": "Справка о ветеринарном благополучии",
                "keywords": ["ветеринар", "благополучи"], "critical": True,
                "source": "Приложение 3",
            },
            {
                "id": "R-06",
                "text": "Обязательство по использованию баранов не менее 2 лет",
                "keywords": ["обязательств", "два года", "2 года"], "critical": True,
                "source": "Приложение 2",
            },
        ],
        "disqualifiers": ["приобретены по бартеру"],
    },

    "КРС_молоко": {
        "name": "Удешевление стоимости производства молока",
        "normative_tenge": {"600+": 45, "400+": 30, "50+": 20, "кооператив": 15},
        "requirements": [
            {
                "id": "R-01", "text": "Наличие учётного номера",
                "keywords": ["учетный номер"], "critical": True, "source": "Приложение 2",
            },
            {
                "id": "R-02", "text": "Фуражное поголовье коров (минимум 50 голов для субсидии)",
                "keywords": ["фуражн", "коров", "поголовье", "голов"], "critical": True,
                "source": "Приложение 1",
            },
            {
                "id": "R-03",
                "text": "Документы о реализации молока на молокоперерабатывающее предприятие",
                "keywords": ["молоко", "реализация", "молокозавод", "переработк", "ЭСФ", "счет"],
                "critical": True, "source": "Приложение 3",
            },
            {
                "id": "R-04", "text": "Справка о ветеринарном благополучии",
                "keywords": ["ветеринар", "благополучи"], "critical": True,
                "source": "Приложение 3",
            },
        ],
        "disqualifiers": [
            "молоко реализовано за пределы РК",
            "молоко реализовано на предприятие другого района области",
        ],
    },
}

UNIVERSAL_DOCUMENT_CHECKLIST = [
    {
        "id": "U-01",
        "text": "Копия свидетельства о регистрации (ТОО, ИП, КФХ)",
        "keywords": ["свидетельство", "регистрация", "ТОО", "ИП", "КФХ", "устав", "юридическое лицо"],
        "critical": False,
    },
    {
        "id": "U-02",
        "text": "Справка о постановке на налоговый учёт (БИН/ИИН)",
        "keywords": ["БИН", "ИИН", "налоговый учет", "КНП"],
        "critical": False,
    },
    {
        "id": "U-03",
        "text": "Банковские реквизиты (справка с банка или реквизиты счёта)",
        "keywords": ["ИИК", "БИК", "банк", "расчетный счет", "счёт"],
        "critical": True,
    },
]

@dataclass
class CheckResult:
    requirement_id: str
    requirement_text: str
    status: str                                                                      
    status_emoji: str                
    found_evidence: str                                   
    is_critical: bool
    source: str                                  

@dataclass
class ComplianceReport:
    subsidy_type: str
    subsidy_name: str
    overall_status: str                                                          
    overall_score: float                                               
    compliance_bonus: float                                     
    checks: list[CheckResult]
    critical_failures: list[str]
    warnings: list[str]
    disqualifiers_found: list[str]
    summary_text: str
    recommendation: str

class ComplianceChecker:

    def __init__(self, gemini_api_key: Optional[str] = None, use_embeddings: bool = True):

        self.api_key = gemini_api_key
        self.use_llm = bool(gemini_api_key)
        self.use_embeddings = use_embeddings
        self._embed_model = None

        mode = "Embeddings + Negation" if self.use_embeddings else ("LLM (Gemini)" if self.use_llm else "Keyword search")
        print(f"ComplianceChecker инициализирован. Режим: {mode}")

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            if EMBED_MODEL_PATH.exists():
                print(f"📦 Загружаю embed-модель локально из: {EMBED_MODEL_PATH}")
                self._embed_model = SentenceTransformer(str(EMBED_MODEL_PATH))
            else:
                print(f"⚠️ Локальная модель не найдена, скачиваю из HF...")
                self._embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._embed_model

    NEGATION_WORDS = [
        "не выдана", "не выдано", "не получена", "не получено",
        "не проведена", "не проведено", "не зарегистрирован",
        "не зарегистрирована", "не оформлен", "не оформлена",
        "не действует", "не предоставлен", "не предоставлена",
        "отсутствует", "отсутствуют",
        "снята", "снят", "снято", "изъят", "изъята",
        "аннулирован", "расторгнут", "отказано", "запрещен",
        "без права", "лишен",
    ]

    def _check_with_embeddings(self, documents_text: str, rules: dict, threshold: float = 0.45) -> list[CheckResult]:
        from sentence_transformers import util

        embed_model = self._get_embed_model()
        sentences = [s.strip() for s in re.split(r'[.!?]+', documents_text) if len(s.strip()) > 10]
        results = []

        for req in rules["requirements"]:
            query = " ".join(req["keywords"])
            query_emb = embed_model.encode(query, convert_to_tensor=True)

            best_score = 0.0
            negation_found = False

            for sentence in sentences:
                s_lower = sentence.lower()
                if any(neg in s_lower for neg in self.NEGATION_WORDS):
                    negation_found = True
                    continue
                sent_emb = embed_model.encode(sentence, convert_to_tensor=True)
                score = util.cos_sim(query_emb, sent_emb).item()
                if score > best_score:
                    best_score = score

            if negation_found:
                status = "НЕ НАЙДЕНО"
                evidence = "Обнаружено отрицание в тексте"
            elif best_score >= threshold:
                status = "ВЫПОЛНЕНО"
                evidence = f"cosine={best_score:.3f}"
            else:
                status = "НЕ НАЙДЕНО"
                evidence = f"cosine={best_score:.3f}"

            results.append(CheckResult(
                requirement_id=req["id"],
                requirement_text=req["text"],
                status=status,
                status_emoji=self._status_emoji(status),
                found_evidence=evidence,
                is_critical=req["critical"],
                source=req["source"],
            ))

        return results

    def check(
        self,
        documents_text: str,
        subsidy_type_key: str = "КРС_маточное",
    ) -> ComplianceReport:
        rules = SUBSIDY_RULES.get(subsidy_type_key, SUBSIDY_RULES["КРС_маточное"])

        # Приоритет: embeddings → LLM (если есть) → keywords fallback
        if self.use_embeddings:
            try:
                checks = self._check_with_embeddings(documents_text, rules)
            except Exception as e:
                print(f"⚠️ Embeddings ошибка ({e}), переключаюсь на keywords")
                checks = self._check_with_keywords(documents_text, rules)
        elif self.use_llm:
            checks = self._check_with_llm(documents_text, rules)
        else:
            checks = self._check_with_keywords(documents_text, rules)

        # Если есть LLM — улучшаем результаты embeddings через LLM
        if self.use_llm and self.use_embeddings:
            try:
                llm_checks = self._check_with_llm(documents_text, rules)
                # Берём LLM результат если embeddings дали НЕ НАЙДЕНО
                llm_map = {c.requirement_id: c for c in llm_checks}
                for i, c in enumerate(checks):
                    llm_c = llm_map.get(c.requirement_id)
                    if llm_c and c.status == "НЕ НАЙДЕНО" and llm_c.status == "ВЫПОЛНЕНО":
                        checks[i] = llm_c
            except Exception as e:
                print(f"⚠️ LLM улучшение не удалось ({e})")

        universal_checks = self._check_universal(documents_text)
        checks = universal_checks + checks

        disqualifiers = self._find_disqualifiers(documents_text, rules)

        return self._build_report(rules, checks, disqualifiers)

    def _check_with_llm(self, documents_text: str, rules: dict) -> list[CheckResult]:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)

        requirements_text = "\n".join([
            f"{r['id']}. [{r['source']}] {r['text']} (КРИТИЧНО: {'ДА' if r['critical'] else 'НЕТ'})"
            for r in rules["requirements"]
        ])

        system_prompt = """Ты — юридический аналитик Министерства сельского хозяйства РК.
Твоя задача: проверить пакет документов сельхозпредприятия на соответствие
Правилам субсидирования (Приказ МСХ РК № 108 от 15.03.2019, ред. 2023).

ИНСТРУКЦИИ:
- Проверяй каждое требование независимо
- "ВЫПОЛНЕНО" — только если есть ЯВНОЕ подтверждение в документах
- "ЧАСТИЧНО" — если есть упоминание но не полная информация
- "НЕ НАЙДЕНО" — если в документах нет никаких следов этого требования
- "ПРЕДУПРЕЖДЕНИЕ" — есть документ, но с потенциальной проблемой (просроченный, неполный)
- Цитируй конкретный фрагмент из документов как доказательство

Отвечай ТОЛЬКО валидным JSON без markdown, преамбул и постамбул."""

        user_message = f"""Проверь документы на соответствие требованиям для субсидии:
"{rules['name']}"

ТРЕБОВАНИЯ ДЛЯ ПРОВЕРКИ:
{requirements_text}

ТЕКСТ ДОКУМЕНТОВ:
---
{documents_text[:30000]}
---

Верни JSON в ТОЧНО таком формате:
{{
  "checks": [
    {{
      "id": "R-01",
      "status": "ВЫПОЛНЕНО",
      "evidence": "Найдено: учетный номер хозяйства 12345678 в справке от акимата"
    }},
    {{
      "id": "R-02",
      "status": "НЕ НАЙДЕНО",
      "evidence": "В представленных документах отсутствуют сведения о земельных участках"
    }}
  ]
}}"""

        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt,
            )
            message = model.generate_content(user_message)
            response_text = message.text.strip()

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group())
            else:
                llm_result = json.loads(response_text)

            llm_checks = {c["id"]: c for c in llm_result.get("checks", [])}

            results = []
            for req in rules["requirements"]:
                llm_check = llm_checks.get(req["id"], {})
                status = llm_check.get("status", "НЕ НАЙДЕНО")
                evidence = llm_check.get("evidence", "Данные не извлечены LLM")

                results.append(CheckResult(
                    requirement_id=req["id"],
                    requirement_text=req["text"],
                    status=status,
                    status_emoji=self._status_emoji(status),
                    found_evidence=evidence,
                    is_critical=req["critical"],
                    source=req["source"],
                ))

            return results

        except Exception as e:
            print(f"⚠️ LLM недоступен ({e}), переключаюсь на keyword-режим")
            return self._check_with_keywords(documents_text, rules)

    def _check_with_keywords(self, documents_text: str, rules: dict) -> list[CheckResult]:
        text_lower = documents_text.lower()
        results = []

        for req in rules["requirements"]:
            req_id = req.get("id", "")
            found_keywords = [kw for kw in req["keywords"] if kw.lower() in text_lower]

            # Проверяем отрицания в контексте найденных ключевых слов
            negation_found = False
            if found_keywords:
                negation_found = self._check_negation_in_context(documents_text, req["keywords"])

            # For obligation requirements (R-05 / R-08), also run regex checks
            # that catch proximity of obligation words near timeframe words.
            regex_match = None
            if "обязательств" in req["text"].lower() or req_id in ("R-05", "R-06", "R-08"):
                regex_match = _OBLIGATION_REGEX.search(documents_text)
                if regex_match is None:
                    # Softer check: obligation keyword exists anywhere
                    kw_match = _OBLIGATION_KW_REGEX.search(documents_text)
                    if kw_match:
                        regex_match = kw_match  # flag as soft hit

            if negation_found:
                # Ключевые слова найдены, но в контексте есть отрицание
                status = "НЕ НАЙДЕНО"
                evidence = f"Обнаружено отрицание: ключевые слова найдены, но документ подтверждает отсутствие"
            elif len(found_keywords) >= 2 or regex_match:
                status = "ВЫПОЛНЕНО"
                if regex_match and not found_keywords:
                    evidence = (
                        f"Regex: найдена формулировка обязательства → "
                        f"«{regex_match.group(0)[:120].strip()}…»"
                    )
                else:
                    evidence = f"Найдены ключевые слова: {', '.join(found_keywords[:3])}"
                    if regex_match and found_keywords:
                        evidence += " + подтверждение обязательств через regex"
            elif len(found_keywords) == 1:
                status = "ЧАСТИЧНО"
                evidence = f"Частично найдено: '{found_keywords[0]}' (ожидается больше подтверждений)"
            else:
                status = "НЕ НАЙДЕНО"
                evidence = f"Не найдены ключевые слова: {', '.join(req['keywords'][:4])}"

            results.append(CheckResult(
                requirement_id=req_id,
                requirement_text=req["text"],
                status=status,
                status_emoji=self._status_emoji(status),
                found_evidence=evidence,
                is_critical=req["critical"],
                source=req["source"],
            ))

        return results

    def _check_negation_in_context(self, text: str, keywords: list[str], window: int = 150) -> bool:
        """Проверяет есть ли отрицание рядом с ключевыми словами."""
        text_lower = text.lower()
        for kw in keywords:
            kw_lower = kw.lower()
            start = 0
            while True:
                pos = text_lower.find(kw_lower, start)
                if pos == -1:
                    break
                # Берём контекст ±window символов
                ctx_start = max(0, pos - window)
                ctx_end = min(len(text), pos + len(kw) + window)
                context = text_lower[ctx_start:ctx_end]
                # Проверяем отрицание в контексте
                if any(neg in context for neg in self.NEGATION_WORDS):
                    return True
                start = pos + len(kw)
        return False

    def _check_universal(self, documents_text: str) -> list[CheckResult]:
        text_lower = documents_text.lower()
        results = []
        for req in UNIVERSAL_DOCUMENT_CHECKLIST:
            found = [kw for kw in req["keywords"] if kw.lower() in text_lower]
            status = "ВЫПОЛНЕНО" if len(found) >= 1 else "НЕ НАЙДЕНО"
            evidence = f"Найдено: {', '.join(found)}" if found else "Не найдено в документах"
            results.append(CheckResult(
                requirement_id=req["id"],
                requirement_text=req["text"],
                status=status,
                status_emoji=self._status_emoji(status),
                found_evidence=evidence,
                is_critical=req["critical"],
                source="Общие требования",
            ))
        return results

    def _find_disqualifiers(self, documents_text: str, rules: dict) -> list[str]:
        text_lower = documents_text.lower()
        found = []
        for disq in rules.get("disqualifiers", []):
            disq_lower = disq.lower()
            if disq_lower in text_lower:
                # Проверяем нет ли отрицания рядом с дисквалификатором
                pos = text_lower.find(disq_lower)
                ctx_start = max(0, pos - 100)
                ctx_end = min(len(text_lower), pos + len(disq_lower) + 50)
                context = text_lower[ctx_start:ctx_end]
                # Если рядом отрицание — это НЕ дисквалификатор
                if any(neg in context for neg in self.NEGATION_WORDS):
                    continue
                found.append(disq)
        return found

    def _build_report(
        self,
        rules: dict,
        checks: list[CheckResult],
        disqualifiers: list[str],
    ) -> ComplianceReport:

        critical_failures = [
            c.requirement_text for c in checks
            if c.is_critical and c.status == "НЕ НАЙДЕНО"
        ]
        warnings = [
            c.requirement_text for c in checks
            if c.status in ("ЧАСТИЧНО", "ПРЕДУПРЕЖДЕНИЕ")
        ]

        total = len(checks)
        done = sum(1 for c in checks if c.status == "ВЫПОЛНЕНО")
        partial = sum(1 for c in checks if c.status == "ЧАСТИЧНО")
        overall_score = (done + partial * 0.5) / total if total > 0 else 0.0

        if disqualifiers:
            overall_status = "ДИСКВАЛИФИКАЦИЯ"
        elif critical_failures:
            overall_status = "НЕ СООТВЕТСТВУЕТ"
        elif overall_score >= 0.85:
            overall_status = "СООТВЕТСТВУЕТ"
        elif overall_score >= 0.60:
            overall_status = "ЧАСТИЧНО"
        else:
            overall_status = "НЕ СООТВЕТСТВУЕТ"

        if overall_status == "ДИСКВАЛИФИКАЦИЯ":
            compliance_bonus = -20.0
        elif overall_status == "НЕ СООТВЕТСТВУЕТ":
            bonus_base = -15.0
            compliance_bonus = bonus_base + (overall_score * 5)
        elif overall_status == "ЧАСТИЧНО":
            compliance_bonus = (overall_score - 0.6) / 0.25 * 10 - 5            
        else:                 
            compliance_bonus = (overall_score - 0.85) / 0.15 * 8           

        compliance_bonus = round(max(-20.0, min(10.0, compliance_bonus)), 1)

        if overall_status == "ДИСКВАЛИФИКАЦИЯ":
            recommendation = (
                "❌ ДИСКВАЛИФИКАЦИЯ: Обнаружены условия, исключающие право на субсидию. "
                f"Причина: {', '.join(disqualifiers)}"
            )
        elif overall_status == "НЕ СООТВЕТСТВУЕТ":
            recommendation = (
                f"🔴 НЕ СООТВЕТСТВУЕТ: {len(critical_failures)} критических требований не выполнены. "
                "Подача заявки без устранения нарушений нецелесообразна."
            )
        elif overall_status == "ЧАСТИЧНО":
            recommendation = (
                f"🟡 ЧАСТИЧНОЕ СООТВЕТСТВИЕ: {len(warnings)} требований выполнены не полностью. "
                "Необходимо дополнить пакет документов."
            )
        else:
            recommendation = (
                f"🟢 СООТВЕТСТВУЕТ: Пакет документов в основном соответствует требованиям. "
                f"Выполнено {done}/{total} требований."
            )

        summary_lines = [
            f"📋 Проверка соответствия: {rules['name']}",
            f"Статус: {overall_status} | Выполнено: {done}/{total} требований ({overall_score*100:.0f}%)",
            f"Влияние на балл: {'+' if compliance_bonus >= 0 else ''}{compliance_bonus:.1f} баллов",
        ]
        if critical_failures:
            summary_lines.append(f"\n❌ Критические нарушения ({len(critical_failures)}):")
            for f in critical_failures[:3]:
                summary_lines.append(f"  • {f}")
        if warnings:
            summary_lines.append(f"\n⚠️ Предупреждения ({len(warnings)}):")
            for w in warnings[:3]:
                summary_lines.append(f"  • {w}")

        return ComplianceReport(
            subsidy_type=list(SUBSIDY_RULES.keys())[0],
            subsidy_name=rules["name"],
            overall_status=overall_status,
            overall_score=round(overall_score, 3),
            compliance_bonus=compliance_bonus,
            checks=checks,
            critical_failures=critical_failures,
            warnings=warnings,
            disqualifiers_found=disqualifiers,
            summary_text="\n".join(summary_lines),
            recommendation=recommendation,
        )

    @staticmethod
    def _status_emoji(status: str) -> str:
        return {
            "ВЫПОЛНЕНО":      "✅",
            "ЧАСТИЧНО":       "⚠️",
            "НЕ НАЙДЕНО":     "❌",
            "ПРЕДУПРЕЖДЕНИЕ": "🔶",
            "ДИСКВАЛИФИКАЦИЯ":"🚫",
        }.get(status, "❓")

def detect_subsidy_type(subsidy_name: str, direction: str) -> str:
    text = (subsidy_name + " " + direction).lower()

    if "быка" in text or "быков" in text or "бык-производитель" in text:
        return "КРС_быки"
    elif "маточн" in text and ("крупного рогатого" in text or "КРС" in text):
        return "КРС_маточное"
    elif "молок" in text:
        return "КРС_молоко"
    elif "баран" in text or "баранов-производит" in text:
        return "овцы_бараны"
    else:
        return "КРС_маточное"          

def run_compliance_check(
    documents_text: str,
    subsidy_name: str,
    direction: str,
    gemini_api_key: Optional[str] = None,
    use_embeddings: bool = True,
) -> dict:
    subsidy_type_key = detect_subsidy_type(subsidy_name, direction)
    checker = ComplianceChecker(gemini_api_key=gemini_api_key, use_embeddings=use_embeddings)
    report = checker.check(documents_text, subsidy_type_key)

    total_checks   = len(report.checks)
    found_checks   = sum(1 for c in report.checks if c.status in ("found", "partial", "ВЫПОЛНЕНО", "ЧАСТИЧНО"))
    doc_completeness = round(found_checks / total_checks, 3) if total_checks > 0 else 0.0

    return {
        "subsidy_type_checked": subsidy_type_key,
        "subsidy_name":         report.subsidy_name,
        "overall_status":       report.overall_status,
        "overall_score_pct":    round(report.overall_score * 100, 1),
        "doc_completeness":     doc_completeness,
        "compliance_bonus":     report.compliance_bonus,
        "recommendation":       report.recommendation,
        "summary":              report.summary_text,
        "checks": [
            {
                "id":           c.requirement_id,
                "text":         c.requirement_text,
                "status":       c.status,
                "emoji":        c.status_emoji,
                "evidence":     c.found_evidence,
                "is_critical":  c.is_critical,
                "source":       c.source,
            }
            for c in report.checks
        ],
        "critical_failures":    report.critical_failures,
        "warnings":             report.warnings,
        "disqualifiers_found":  report.disqualifiers_found,
    }
