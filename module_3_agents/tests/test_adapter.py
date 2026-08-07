import os
import pytest
import config
from adapter import generate_search_query

@pytest.fixture(autouse=True)
def check_env():
    if "GOOGLE_API_KEY" not in os.environ:
        pytest.fail("Для запуска тестов необходимо задать GOOGLE_API_KEY в переменных окружения")

def test_adapter_initial_query():
    """Тест 1: Обычная генерация запроса без ошибок"""
    task = "Сравнение алгоритмов балансировки в AVL-дереве"
    
    query = generate_search_query(task)
    
    assert isinstance(query, str)
    assert len(query) > 5

def test_adapter_reflection():
    """Тест 2: Работа над ошибками (Смещение фокуса по требованию Валидатора)"""
    task = "Описание градиентного бустинга в ML"
    previous_queries = ["градиентный бустинг машинное обучение"]
    rejection_reason = "Текст описывает случайный лес. Нужна математика XGBoost или LightGBM."
    
    new_query = generate_search_query(task, previous_queries, rejection_reason)
    
    assert new_query not in previous_queries
    
    new_query_lower = new_query.lower()
    assert "xgboost" in new_query_lower or "lightgbm" in new_query_lower or "математика" in new_query_lower