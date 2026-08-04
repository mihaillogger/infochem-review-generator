"""Асинхронные воркеры для тяжелых задач парсинга и векторизации."""

import asyncio
from pathlib import Path

import structlog
from pydantic import ValidationError
from structlog.contextvars import bind_contextvars, clear_contextvars
from taskiq import Context, TaskiqDepends, TaskiqEvents, TaskiqState

from rag_core.broker import broker
from rag_core.chunker import chunk_document
from rag_core.config import settings
from rag_core.embedder import NomicEmbedder
from rag_core.enricher import enrich_document
from rag_core.llm import AsyncGeminiClient
from rag_core.logger import setup_logging
from rag_core.preprocessor import warmup_tokenizer
from rag_core.profiler import profile_time
from rag_core.schemas import ParsedDocument, ParseTaskResult
from rag_core.store import get_store

setup_logging()
logger = structlog.get_logger(__name__)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
@profile_time
async def warmup_models(state: TaskiqState) -> None:
    """
    Прогрев нейросетей внутри изолированного процесса воркера
    до того, как он начнет тянуть задачи из очереди.

    Args:
        state (TaskiqState): Текущее состояние и контекст воркера.
    """
    logger.info("worker_warmup_started", message="Loading models into memory...")

    store = await get_store()

    await asyncio.to_thread(warmup_tokenizer)
    await asyncio.to_thread(store.dense_embedder.encode_batch, ["warmup_query"], is_document=False)

    logger.info("worker_warmup_finished", message="Models are hot and ready!")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown_models(state: TaskiqState) -> None:
    """
    Очищает ресурсы воркера при его остановке.

    Закрывает сетевые соединения с векторной базой данных, предотвращая
    утечку файловых дескрипторов.

    Args:
        state (TaskiqState): Текущее состояние и контекст воркера.
    """
    logger.info("worker_shutdown_started", message="Releasing resources...")
    store = await get_store()
    await store.close()
    logger.info("worker_shutdown_finished", message="Worker resources released!")


@broker.task(task_name="vectorize_document_task")
@profile_time
async def vectorize_document_task(
    file_path: str, context: Context = TaskiqDepends()
) -> ParseTaskResult:
    """
    Асинхронная задача для векторизации обработанного документа.

    Находит готовый JSON-файл от парсера, десериализует его,
    нарезает текст на чанки и сохраняет в векторную базу данных.

    Args:
        file_path (str): Абсолютный путь к сырому PDF.
        context (Context): Контекст выполнения задачи от TaskIQ.

    Returns:
        ParseTaskResult: Объект с результатами выполнения задачи.

    Raises:
        ValueError: Если передан пустой путь к файлу.
        FileNotFoundError: Если JSON-файл от парсера не найден.
        ValidationError: Если структура JSON-файла не прошла валидацию Pydantic.
        Exception: При возникновении прочих непредвиденных ошибок обработки.
    """
    clear_contextvars()
    bind_contextvars(task_id=context.message.task_id)

    logger.info("vectorize_document_task_started", file_path=file_path)

    if not file_path:
        error_msg = "Получен пустой путь к файлу"
        logger.error("vectorize_document_task_invalid_input", error=error_msg)
        raise ValueError(error_msg)

    try:
        pdf_path = Path(file_path)
        log = logger.bind(file_name=pdf_path.name)

        parsed_data_dir = Path(settings.PROCESSED_DATA_ROOT)
        mineru_json_path = parsed_data_dir / f"{pdf_path.stem}{settings.PARSED_FILE_SUFFIX}"

        if not mineru_json_path.exists():
            log.warning("mineru_json_not_found", expected_path=str(mineru_json_path))
            raise FileNotFoundError(f"JSON-файл парсера не найден по пути: {mineru_json_path}")

        try:
            json_text = await asyncio.to_thread(mineru_json_path.read_text, encoding="utf-8")
            doc = ParsedDocument.model_validate_json(json_text)
        except ValidationError as e:
            log.exception("mineru_validation_failed", error=str(e))
            raise
        except Exception as e:
            log.exception("mineru_read_failed", error=str(e))
            raise

        llm_client = AsyncGeminiClient()
        doc = await enrich_document(doc, llm_client)

        embedder = NomicEmbedder()
        chunks = await asyncio.to_thread(chunk_document, doc, embedder)

        if chunks:
            store = await get_store()
            await store.insert_chunks(chunks)

        logger.info("vectorize_document_task_finished", file_path=file_path)
        return ParseTaskResult(status="success", file_path=file_path)

    except Exception as e:
        logger.error(
            "vectorize_document_task_failed", file_path=file_path, error=str(e), exc_info=True
        )
        raise
