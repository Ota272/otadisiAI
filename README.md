# SmartAgro Score

**Интеллектуальный скоринг заявок на субсидии в сельском хозяйстве** — REST API на FastAPI, дашборд на Streamlit, модель **XGBoost** с объяснениями **SHAP**, опциональный разбор PDF и проверка соответствия правилам через **Google Gemini**.

Проект в контексте хакатона **Decentrathon 5.0** и темы **AI for Government**. ИИ выдаёт рекомендацию; итоговое решение по заявке остаётся за комиссией МСХ РК.

---

## Возможности

| Компонент | Описание |
|-----------|----------|
| **Скоринг** | Балл **0–100**, зона **green / yellow / red**, текстовая рекомендация |
| **Объяснимость** | **SHAP** — вклад 17 признаков в итоговый балл |
| **Документы** | Загрузка PDF, извлечение текста, уточнение признаков и compliance через **Gemini** |
| **Хранение** | **SQLite** — заявки и решения комиссии |
| **Дашборд** | Streamlit: очередь заявок, шорт-лист, профиль с графиками и SHAP, интеграция API |
| **Human-in-the-Loop** | Фиксация решения комиссии (approved/rejected/review) через API |

---

## Стек технологий

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic
- **ML:** XGBoost, scikit-learn, SHAP, joblib, pandas, numpy
- **LLM / PDF:** google-generativeai, pdfplumber, pypdf, PyMuPDF
- **Frontend:** Streamlit, Plotly, requests
- **Данные:** SQLite

---

## Структура репозитория

```
otadisiAI/
├── src/
│   ├── main.py              # FastAPI: скоринг, документы, решения, аналитика
│   └── store.py             # SQLite: заявки и решения
├── frontend/
│   └── frontend.py          # Streamlit-дашборд
├── ml/
│   ├── shap_integration.py  # ScoringEngine, SHAP, Gemini (текст PDF, извлечение фич)
│   └── compliance_checker.py # Проверка соответствия правилам
├── data_prep.py             # Подготовка данных → CSV фич
├── train_model.py           # Обучение XGBoost и артефакты в models/
├── models/                  # xgb_scorer.joblib, scaler, SHAP explainer, feature_names.json
├── data/                    # data.csv, data_cleaned.csv, data_features.csv
├── requirements.txt
└── .env                     # локально: ключи API (не коммитить)
```

---

## Быстрый старт

### 1. Клонирование и окружение

```bash
git clone https://github.com/Ota272/otadisiAI.git
cd otadisiAI
python -m venv .venv
```

**Windows (PowerShell):**
`.venv\Scripts\activate`

**Linux / macOS:**
`source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

### 2. Переменные окружения

Создайте файл **`.env`** в корне проекта (файл в `.gitignore`):

```env
GEMINI_API_KEY=ваш_ключ_Google_AI_Studio
```
https://aistudio.google.com/api-keys

модель бесплатная, просто можно будет менять ключ, 
ошибка "429 You exceeded your current quota, please check your plan and billing details." - абсолютно нормальна, модель бесплатная, просто можно будет менять ключ, 
это не критично.
Без ключа часть функций (разбор PDF, заключения LLM, compliance) будет недоступна; базовый скоринг по числовым полям работает при наличии обученной модели.

### 3. Модель (если артефактов ещё нет)

Из корня проекта, при необходимости после подготовки данных:

```bash
python data_prep.py
python train_model.py
```

В папке **`models/`** должны появиться: `xgb_scorer.joblib`, `scaler.joblib`, `shap_explainer.joblib`, `feature_names.json`.

### 4. Запуск backend

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

- Документация API: `http://localhost:8003/docs`
- Проверка: `http://localhost:8003/health`

### 5. Запуск дашборда (второй терминал)

```bash
streamlit run frontend/frontend.py 
```

---

## API (кратко)

Заголовок авторизации: **`X-API-Key`** (демо-ключ: `sk-msgov-2025-demo-key-abc123`).

| Метод | Путь | Назначение |
|-------|------|------------|
| `POST` | `/api/v1/score` | Скоринг по JSON-полям заявки (17 признаков) |
| `POST` | `/api/v1/score-with-documents` | Скоринг + PDF (multipart/form-data) |
| `GET` | `/api/v1/applications` | Список заявок (фильтры: zone, min_score) |
| `POST` | `/api/v1/decision` | Фиксация решения комиссии (decision: approved/rejected/review) |
| `POST` | `/api/v1/giss/sync` | **Demo:** генерация тестовых заявок (в памяти) |

---

## Примеры запросов

### Скоринг заявки

```bash
curl -X POST http://localhost:8003/api/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \
  -d '{
    "bin_iin": "123456789012",
    "company_name": "ТОО Агро-Нур",
    "region": "Алматинская область",
    "subsidy_type": "Приобретение племенного КРС",
    "requested_amount": 15000000,
    "pedigree_ratio": 0.85,
    "veterinary_compliance": 0.98,
    "years_in_operation": 7
  }'
```

### Решение комиссии

```bash
curl -X POST http://localhost:8003/api/v1/decision \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123" \
  -d '{
    "application_id": "A7F2B1C0",
    "decision": "approved",
    "comment": "Соответствует критериям"
  }'
```

### Список заявок (фильтр по зоне)

```bash
curl "http://localhost:8003/api/v1/applications?zone=green&min_score=80" \
  -H "X-API-Key: sk-msgov-2025-demo-key-abc123"
```

---

## Зоны скоринга

| Зона | Диапазон | Рекомендация |
|------|----------|--------------|
| 🟢 green | 80–100 | Строго рекомендовано |
| 🟡 yellow | 50–79 | Рассмотрение комиссией |
| 🔴 red | 0–49 | Не рекомендовано |

---

## Важно

- Решение системы — **рекомендательное**. Финальное решение принимает комиссия МСХ РК.
- **Тестовые заявки** (`POST /api/v1/giss/sync`) не сохраняются в SQLite.
- Не публикуйте **`.env`** и ключи API в репозиторий.
