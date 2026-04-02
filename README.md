# SmartAgro Score

**Интеллектуальный скоринг заявок на субсидии в сельском хозяйстве** — REST API на FastAPI, дашборд на Streamlit, модель **XGBoost** с объяснениями **SHAP**, опциональный разбор PDF и проверка соответствия правилам через **Google Gemini**.

Проект в контексте хакатона **Decentrathon** и темы **AI for Government**. ИИ выдаёт рекомендацию; итоговое решение по заявке остаётся за комиссией.

---

## Возможности

| Компонент | Описание |
|-----------|----------|
| **Скоринг** | Балл **0–100**, зона **green / yellow / red**, текстовая рекомендация |
| **Объяснимость** | **SHAP** — вклад признаков в итоговый балл |
| **Документы** | Загрузка PDF, извлечение текста, уточнение признаков и compliance через **Gemini** |
| **Хранение** | **SQLite** (`data/applications.sqlite`) для реальных заявок; тестовые заявки только в памяти |
| **Дашборд** | Streamlit: очередь заявок, шорт-лист, профиль с графиками и SHAP, интеграция API |

---

## Стек технологий

- **Backend:** Python 3.10+, FastAPI, Uvicorn  
- **ML:** XGBoost, scikit-learn, SHAP, joblib  
- **LLM / PDF:** `google-generativeai`, pdfplumber, pypdf, PyMuPDF  
- **Frontend:** Streamlit, Plotly, requests  
- **Данные:** pandas, SQLite  

---

## Структура репозитория

```
├── main.py              # FastAPI: скоринг, документы, решения, аналитика
├── app.py               # Streamlit-дашборд
├── shap_integration.py  # ScoringEngine, SHAP, Gemini (текст PDF, извлечение фич)
├── compliance_checker.py
├── applications_store.py # SQLite: заявки (без демо)
├── data_prep.py         # Подготовка данных → CSV фич
├── train_model.py       # Обучение XGBoost и артефакты в models/
├── models/              # xgb_scorer.joblib, scaler, SHAP explainer, feature_names.json
├── data/                # БД заявок (sqlite; не коммитить секреты)
├── requirements.txt
└── .env                 # локально: ключи API (не коммитить)
```

В каталоге **`app/`** может лежать альтернативная/историческая версия приложения — основной сценарий из корня: **`main.py` + `app.py`**.

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

Без ключа часть функций (разбор PDF, заключения LLM, compliance) будет недоступна; базовый скоринг по числовым полям может работать при наличии обученной модели.

### 3. Модель (если артефактов ещё нет)

Из корня проекта, при необходимости после подготовки данных:

```bash
python data_prep.py
python train_model.py
```

В папке **`models/`** должны появиться, в частности, `xgb_scorer.joblib`, `scaler.joblib`, `shap_explainer.joblib`, `feature_names.json`.

### 4. Запуск backend

```bash
uvicorn main:app --reload --port 8000
```

- Документация API: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Проверка: [http://localhost:8000/health](http://localhost:8000/health)

### 5. Запуск дашборда (второй терминал)

```bash
streamlit run app.py
```

В настройках дашборда должен быть указан тот же хост API (по умолчанию часто `http://127.0.0.1:8000`).

---

## API (кратко)

Заголовок авторизации: **`X-API-Key`** (демо-ключи заданы в `main.py`, для продакшена их нужно заменить на безопасную схему).

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/` | Информация о сервисе |
| `GET` | `/health` | Статус и загрузка модели |
| `POST` | `/api/v1/score` | Скоринг по JSON-полям заявки |
| `POST` | `/api/v1/score-with-documents` | Скоринг + PDF (multipart) |
| `GET` | `/api/v1/applications` | Список заявок |
| `GET` | `/api/v1/applications/{id}` | Одна заявка |
| `POST` | `/api/v1/decision` | Фиксация решения комиссии |
| `POST` | `/api/v1/giss/sync` | **Демо:** тестовые заявки (не пишутся в SQLite) |
| `GET` | `/api/v1/analytics/summary` | Сводная аналитика по нетестовым заявкам |

Признаков в модели **17** (см. `models/feature_names.json`).

---

## Важно

- Решение системы — **рекомендательное**.  
- **Тестовые заявки** (кнопка в UI / `POST .../giss/sync`) помечаются как демо и **не сохраняются** в SQLite.  
- Не публикуйте **`.env`** и ключи API в репозиторий.

---

## Лицензия

См. файл `LICENSE` в репозитории (если добавлен).

---

<p align="center">
  <b>SmartAgro Score</b> · ML + SHAP + LLM для прозрачного скоринга субсидий АПК
</p>
