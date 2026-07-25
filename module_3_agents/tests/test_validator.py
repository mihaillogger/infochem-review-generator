import sys
import os

# 1. Вычисляем путь к папке, где лежит этот файл, и поднимаемся на один уровень вверх
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Жестко добавляем родительскую папку в начало поиска Питона
sys.path.insert(0, parent_dir)

import pytest
import src.agents.config
from validator import check_relevance

# Проверяем, что ключ есть в системе, иначе тесты упадут до старта
@pytest.fixture(autouse=True)
def check_env():
    if "GOOGLE_API_KEY" not in os.environ:
        pytest.fail("Для запуска тестов необходимо задать GOOGLE_API_KEY в переменных окружения")

def test_validator_perfect_match():
    """Тест 1: Идеальное попадание (Технический текст с фактами)"""
    query = "Как работает балансировка в AVL-дереве при построении инвертированного индекса?"
    good_chunk = "В AVL-дереве балансировка достигается за счет малых и больших вращений узлов. Разность высот левого и правого поддеревьев не должна превышать 1. Это гарантирует логарифмическое время поиска при обращении к индексу."
    
    result = check_relevance(query, good_chunk)
    
    assert result["is_relevant"] is True
    assert result["reason"] == "", "Для релевантного текста причина должна быть пустой"

def test_validator_water_text():
    """Тест 2: Откровенная вода (По теме, но без конкретики)"""
    query = "Формула остаточного члена в формуле Тейлора"
    bad_chunk = "Ряды Тейлора играют важнейшую роль в математическом анализе. Они позволяют приближать сложные функции многочленами, что активно используется в дифференциальном исчислении и физике."
    
    result = check_relevance(query, bad_chunk)
    
    assert result["is_relevant"] is False
    assert len(result["reason"]) > 0, "Модель должна была объяснить, почему забраковала текст (нет самой формулы)"

def test_validator_partial_match_but_correct():
    """Тест 3: Специфичный ML-контекст (Проверка на понимание терминов)"""
    query = "Архитектура роста деревьев в LightGBM"
    ml_chunk = "LightGBM использует алгоритм построения на основе гистограмм (histogram-based) и стратегию роста дерева по листьям (leaf-wise), что значительно ускоряет обучение на больших датасетах по сравнению с CatBoost."
    
    result = check_relevance(query, ml_chunk)
    
    assert result["is_relevant"] is True