"""API-шлюз Модуля 2.

Реализует REST-интерфейс для поиска по базе знаний и
асинхронной векторизации обработанных документов.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from parser_db.broker import broker
from parser_db.config import settings
from parser_db.logger import setup_logging
from parser_db.schemas import IngestRequest, RFC9457Error, SearchRequest, SearchResponse
from parser_db.store import AsyncQdrantStore, get_store
from parser_db.worker import parse_pdf_task

setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управляет жизненным циклом приложения FastAPI.

    Открывает соединение брокера TaskIQ с Redis при старте сервера
    и корректно закрывает его при остановке.

    Args:
        app: Экземпляр приложения FastAPI.
    """
    logger.info("system_startup", message="Starting Infochem RAG Core...")
    await broker.startup()
    await get_store()
    logger.info("system_ready", message="Vector DB and Models are loaded.")
    yield
    await broker.shutdown()
    logger.info("system_shutdown", message="Shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)


# --- RFC 9457 Обработчики ошибок ---


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Перехватывает ошибки Pydantic и форматирует их по стандарту RFC 9457.

    Вшивает прямую инструкцию для LLM-агента в поле `detail`,
    чтобы он мог автономно исправить свой запрос.
    """
    errors = exc.errors()

    llm_instructions = "Твой JSON-запрос не прошел валидацию. "
    for err in errors:
        loc = " -> ".join(str(part) for part in err["loc"])
        msg = err["msg"]
        llm_instructions += f"Ошибка в поле '{loc}': {msg}. "

    llm_instructions += settings.LLM_INSTRUCTION_VALIDATION

    error_response = RFC9457Error(
        type=settings.RFC_TYPE_VALIDATION,
        title="Unprocessable Entity (Validation Error)",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=llm_instructions,
        instance=str(request.url),
        errors=cast(list[dict[str, Any]], errors),
    )

    logger.warning("agent_validation_error", url=str(request.url), errors=errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )


# --- API Эндпоинты ---


@app.post(
    "/api/v1/search",
    summary="Гибридный поиск по базе знаний",
    response_model=SearchResponse,
    responses={
        422: {"model": RFC9457Error, "description": "Ошибка валидации запроса"},
        500: {"model": RFC9457Error, "description": "Внутренняя ошибка сервера"},
    },
)
async def search_documents(
    request: SearchRequest, http_request: Request, store: AsyncQdrantStore = Depends(get_store)
) -> Any:
    """Точка входа для агентов. Выполняет поиск Dense+Sparse с алгоритмом RRF."""
    try:
        section_val = request.section_filter.value if request.section_filter else None

        results = await store.hybrid_search(
            query=request.query,
            limit=request.limit,
            doi_filter=request.doi_filter,
            section_filter=section_val,
            require_table=request.require_table,
            require_math=request.require_math,
        )

        logger.info("hybrid_search_success", query=request.query, returned_chunks=len(results))

        return {"status": "success", "count": len(results), "data": results}
    except Exception as e:
        logger.exception("hybrid_search_failed", query=request.query, error=str(e))

        error_response = RFC9457Error(
            type=settings.RFC_TYPE_INTERNAL,
            title="Internal Server Error",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{settings.LLM_INSTRUCTION_INTERNAL} Техническая деталь: {str(e)}",
            instance=str(http_request.url),
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(),
        )


@app.post(
    "/api/v1/documents",
    summary="Запуск чанкинга и векторизации обработанных документов",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_documents(request: IngestRequest) -> dict[str, str]:
    """Отправляет задачу векторизации в воркер TaskIQ."""
    task = await parse_pdf_task.kiq(request.file_paths)

    logger.info("ingest_task_created", task_id=task.task_id, files_count=len(request.file_paths))

    return {
        "status": "accepted",
        "message": "Задачи на векторизацию успешно добавлены в очередь.",
        "task_id": task.task_id,
    }
