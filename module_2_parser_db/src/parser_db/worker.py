"""Асинхронные воркеры для тяжелых задач парсинга и векторизации."""

import asyncio
from pathlib import Path
from typing import Any

import structlog

from parser_db.broker import broker
from parser_db.chunker import chunk_document
from parser_db.enricher import enrich_document
from parser_db.profiler import profile_time
from parser_db.schemas import ParsedDocument
from parser_db.store import get_store

logger = structlog.get_logger(__name__)


@broker.task(task_name="parse_pdf_task")
@profile_time
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
        log = logger.bind(file_name=pdf_path.name)

        mineru_json_path = parsed_data_dir / f"{pdf_path.stem}_parsed.json"

        if not mineru_json_path.exists():
            log.warning("mineru_json_not_found", expected_path=str(mineru_json_path))
            continue

        try:
            json_text = await asyncio.to_thread(mineru_json_path.read_text, encoding="utf-8")
            doc = ParsedDocument.model_validate_json(json_text)
        except Exception as e:
            log.exception("mineru_validation_failed", error=str(e))
            continue

        doc = await enrich_document(doc)

        chunks = await asyncio.to_thread(chunk_document, doc)

        if chunks:
            # Инициализируем БД только тогда, когда есть что сохранять
            store = await get_store()
            await store.insert_chunks(chunks)

        processed += 1

    logger.info("parse_pdf_task_finished", processed_files=processed, total_files=len(file_paths))
    return {"status": "success", "processed_files": processed}
