#!/bin/bash
set -e

echo "Начинаем полный цикл проверок (Ruff, Mypy, Pytest)..."

MODULES=("module_1_downloader" "module_2_parser_db" "module_3_agents")

for MODULE in "${MODULES[@]}"; do
    echo "======================================"
    echo "Проверка модуля: $MODULE"
    echo "======================================"

    cd "$MODULE"

    echo "1) Запуск Ruff (Линтер и форматирование)..."
    uv run ruff check .
    uv run ruff format --check .

    echo "2) Запуск Mypy (Тайпчекинг)..."
    if [ -d "src" ]; then
        uv run mypy src/
    else
        uv run mypy .
    fi

    echo "3) Запуск Pytest с расчетом покрытия..."
    if [ -d "tests" ]; then
        uv run pytest tests/ -v --cov=src --cov-report=term-missing
    else
        echo "[SKIP] Папка tests/ не найдена. Пропускаем..."
    fi

    cd ..
done

echo "======================================"
echo "Все модули успешно прошли проверки!"