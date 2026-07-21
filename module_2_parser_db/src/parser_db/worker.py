"""Асинхронные воркеры для тяжелых задач парсинга и векторизации."""

import json
from pathlib import Path
from typing import Any

from parser_db.broker import broker
from parser_db.chunker import chunk_document
from parser_db.extractor import build_parsed_document
from parser_db.store import QdrantStore

store = QdrantStore()


@broker.task(task_name="parse_pdf_task")
async def parse_pdf_task(file_paths: list[str]) -> dict[str, Any]:
    """
    Асинхронная задача для обработки скачанных PDF-файлов.

    Читает единый манифест metadata.json от Модуля 1, извлекает DOI
    для каждого файла по его пути, запускает парсер MinerU,
    нарезает чанки и сохраняет в БД.

    Args:
        file_paths: Список путей к сырым PDF.

    Returns:
        Словарь со статусом выполнения.
    """
    processed = 0

    if not file_paths:
        return {"status": "success", "processed_files": 0}

    first_pdf = Path(file_paths[0])
    metadata_path = first_pdf.parent.parent / "metadata.json"

    manifest = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            manifest = json.load(f)

    path_to_doi = {}
    for doi, record in manifest.items():
        if record.get("pdf_path"):
            path_to_doi[Path(record["pdf_path"]).name] = doi

    parsed_data_dir = first_pdf.parent.parent / "processed"

    for path_str in file_paths:
        pdf_path = Path(path_str)
        doi = path_to_doi.get(pdf_path.name, f"unknown-doi-{processed}")

        mineru_json_path = parsed_data_dir / f"{pdf_path.stem}_parsed.json"

        if not mineru_json_path.exists():
            print(f"[-] Результаты MinerU не найдены для {pdf_path.name}. Ждем парсер.")
            continue

        try:
            with open(mineru_json_path, encoding="utf-8") as f:
                mineru_data = json.load(f)

            if isinstance(mineru_data, dict) and "content" in mineru_data:
                mineru_data = mineru_data["content"]

        except Exception as e:
            print(f"[-] Ошибка чтения JSON от MinerU для {pdf_path.name}: {e}")
            continue

        doc = build_parsed_document(
            mineru_data=mineru_data, doi=doi, title=f"Article {pdf_path.name}"
        )

        chunks = chunk_document(doc)

        if chunks:
            store.insert_chunks(chunks)

        processed += 1

    return {"status": "success", "processed_files": processed}
