#!/usr/bin/env python3
"""
Тест функционала цитирования из PDF документов.
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.shap_integration import generate_gemini_expert_opinion

# Тестовые данные заявки с PDF текстом
test_app_data = {
    "company_name": "ТОО «Агро-Элита»",
    "bin_iin": "123456789012",
    "region": "Алматинская область",
    "subsidy_type": "Приобретение племенного маточного поголовья КРС",
    "direction": "Субсидирование в скотоводстве",
    "requested_amount": 35000000,
    "score": 65,
    "score_ml": 65,
    "score_doc": None,
    "ml_weight_used": 1.0,
    "doc_weight_used": 0.0,
    "zone": "yellow",
    "manual_review_required": True,
    "source_system": "manual",
    "raw_features_used": {
        "years_in_operation": 8.0,
        "gross_output_growth_yoy": 0.05,
        "debt_load_ratio": 1.2,
        "veterinary_compliance": 0.85,
        "historical_survival_rate": 0.92,
        "pedigree_ratio": 0.65,
        "subsidy_dependence_index": 0.28,
        "land_to_livestock_ratio": 7.5,
        "previous_subsidies_count": 4.0,
        "livestock_count": 120.0,
    },
    "top_positive_factors": [
        {"label": "Сохранность поголовья", "shap_value": 3.2},
        {"label": "Ветеринарное соответствие", "shap_value": 2.8},
        {"label": "Доля племенного поголовья", "shap_value": 2.1},
    ],
    "top_negative_factors": [
        {"label": "Долговая нагрузка", "shap_value": -2.5},
        {"label": "Зависимость от субсидий", "shap_value": -1.8},
    ],
    "compliance": {
        "overall_status": "warning",
        "overall_score_pct": 75,
        "doc_completeness": "mostly_full",
        "critical_failures": [],
        "warnings": ["Не найдено обязательство по целевому использованию"],
        "disqualifiers_found": [],
    },
    # Имитация текста из PDF (как будто извлечено из реальных документов)
    "documents_extracted_text": """
Договор купли-продажи №45/2026 от 15 января 2026 года
Продавец: ТОО «Племзавод Алматинский», БИН 987654321098
Покупатель: ТОО «Агро-Элита», БИН 123456789012

Предмет договора: Племенные нетели черно-пестрой породы
Количество: 120 голов
Цена за голову: 580 000 тенге
Общая сумма: 69 600 000 тенге

Срок поставки: до 28 февраля 2026 года
Гарантийные обязательства: Продавец гарантирует племенную ценность

Приложение 1: Ветеринарное свидетельство №КЗ-2026-001234
Дата выдачи: 10 января 2026 года
Ветеринарный врач: Иванов А.С.
Статус: Благополучно по инфекционным заболеваниям
Результат анализов: Бруцеллез - отрицательно
Туберкулез - отрицательно
Лейкоз - отрицательно

Приложение 2: Племенное свидетельство
Порода: Черно-пестрая
Класс: Элита
Возраст: 18-24 месяца
Продуктивность матерей: 6500 кг молока за лактацию

Земельные документы:
Аренда пастбища: 900 гектар
Срок договора: до 31 декабря 2030 года
Кадастровый номер: 01-234-567-890
Обеспеченность: 7.5 га на голову

Финансовые показатели:
Выручка 2025: 125 000 000 тенге
Рост к 2024: +5%
Долговая нагрузка: 1.2 EBITDA
Субсидии составляют менее 30% дохода

Обязательства:
Покупатель обязуется использовать поголовье по целевому назначению
в течение 2 лет с момента приобретения.
""",
    "documents_pdf_count": 3,
    "documents_text_chars": 1234,
    "documents_extracted_ok": True,
}

def test_citation_generation():
    """Тестирует генерацию экспертного заключения с цитатами."""
    print("=" * 80)
    print("ТЕСТ: Генерация экспертного заключения с цитатами из PDF")
    print("=" * 80)
    
    # Получаем API ключ
    import os
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    if not gemini_key:
        print("\n⚠️ GEMINI_API_KEY не найден в окружении.")
        print("Создаю .env файл с примером (замените на реальный ключ):")
        env_path = ROOT_DIR / ".env"
        if not env_path.exists():
            env_path.write_text("GEMINI_API_KEY=your_api_key_here\n", encoding="utf-8")
            print(f"✅ Создан {env_path}")
        print("\nДля запуска теста:")
        print("1. Получите ключ: https://aistudio.google.com/app/apikey")
        print("2. Добавьте в .env: GEMINI_API_KEY=ваш_ключ")
        print("3. Запустите: python test_citations.py")
        return False
    
    print("\n🚀 Генерирую экспертное заключение...")
    print("(это может занять 10-30 секунд)\n")
    
    try:
        result = generate_gemini_expert_opinion(test_app_data, gemini_key)
        
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ:")
        print("=" * 80)
        
        # Пытаемся распарсить JSON
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "conclusion" in parsed:
                print("\n✅ JSON успешно распарсен!\n")
                
                # Показываем заключение
                conclusion = parsed.get("enriched_conclusion", parsed.get("conclusion", ""))
                print("ЗАКЛЮЧЕНИЕ:")
                print("-" * 80)
                print(conclusion)
                print("-" * 80)
                
                # Показываем цитаты
                citations = parsed.get("citations", parsed.get("citations_list", []))
                if citations:
                    print(f"\n📎 НАЙДЕНО ЦИТАТ: {len(citations)}\n")
                    for i, citation in enumerate(citations, 1):
                        print(f"Цитата #{i}:")
                        print(f"  Пункт: {citation.get('point_number', 'N/A')}")
                        print(f"  Строка: {citation.get('line_number', 'N/A')}")
                        print(f"  Цитата: {citation.get('quote', 'N/A')[:100]}...")
                        print(f"  Пояснение: {citation.get('explanation', 'N/A')[:100]}")
                        print()
                else:
                    print("\n⚠️ Цитаты не найдены в ответе")
                
                return True
            else:
                print("\n❌ Ответ не содержит поля 'conclusion'")
                print(result[:500])
                return False
                
        except json.JSONDecodeError:
            print("\n⚠️ Ответ не в JSON формате (старый формат или ошибка):")
            print(result[:1000])
            return False
            
    except Exception as e:
        print(f"\n❌ Ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_citation_generation()
    sys.exit(0 if success else 1)
