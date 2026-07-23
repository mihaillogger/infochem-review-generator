"""Асинхронные воркеры для тяжелых задач парсинга и векторизации."""

from pathlib import Path
from typing import Any

from parser_db.broker import broker
from parser_db.chunker import chunk_document
from parser_db.store import get_store
from parser_db.schemas import ParsedDocument


@broker.task(task_name="parse_pdf_task")
async def parse_pdf_task(file_paths: list[str]) -> dict[str, Any]:
    """
    Асинхронная задача для векторизации обработанных документов.

    Находит готовые JSON-файлы от парсера MinerU, десериализует их,
    нарезает текст на чанки и сохраняет в векторную базу данных.

    Args:
        file_paths: Список абсолютных путей к сырым PDF.

    Returns:
        Словарь со статусом выполнения и количеством обработанных файлов.
    """
    if not file_paths:
        return {"status": "success", "processed_files": 0}

    processed = 0
    first_pdf = Path(file_paths[0])
    parsed_data_dir = first_pdf.parent.parent / "processed"

    for path_str in file_paths:
        pdf_path = Path(path_str)
        mineru_json_path = parsed_data_dir / f"{pdf_path.stem}_parsed.json"

        if not mineru_json_path.exists():
            print(f"[-] Результаты MinerU не найдены для {pdf_path.name}. Ждем парсер.")
            continue

        try:
            json_text = mineru_json_path.read_text(encoding="utf-8")
            doc = ParsedDocument.model_validate_json(json_text)
        except Exception as e:
            print(f"[-] Ошибка валидации JSON от MinerU для {pdf_path.name}: {e}")
            continue

        chunks = chunk_document(doc)

        if chunks:
            # Инициализируем БД только тогда, когда есть что сохранять
            store = get_store()
            store.insert_chunks(chunks)

        processed += 1

    return {"status": "success", "processed_files": processed}
