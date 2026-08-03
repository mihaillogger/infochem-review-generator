"""API-шлюз Модуля 2.

Реализует REST-интерфейс для поиска по базе знаний и
асинхронной векторизации обработанных документов.
"""

import asyncio
import secrets
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from structlog.contextvars import bind_contextvars, clear_contextvars

from parser_db.broker import broker
from parser_db.config import settings
from parser_db.logger import setup_logging
from parser_db.preprocessor import warmup_tokenizer
from parser_db.schemas import (
    IngestRequest,
    IngestResponse,
    RFC9457Error,
    SearchRequest,
    SearchResponse,
)
from parser_db.store import AsyncQdrantStore, get_store
from parser_db.worker import vectorize_document_task

setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Управляет жизненным циклом приложения FastAPI.

    Открывает соединение брокера TaskIQ с Redis при старте сервера
    и корректно закрывает его при остановке.

    Args:
        app (FastAPI): Экземпляр приложения FastAPI.

    Returns:
        AsyncGenerator[None, None]: Асинхронный генератор жизненного цикла.
    """
    logger.info("system_startup", message="Starting Infochem RAG Core...")
    await broker.startup()
    await get_store()
    await asyncio.to_thread(warmup_tokenizer)
    logger.info("system_ready", message="Vector DB and Models are loaded.")

    yield

    store = await get_store()
    await store.close()
    await broker.shutdown()
    logger.info("system_shutdown", message="Shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER_NAME)


def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Проверяет токен авторизации клиента.

    Args:
        api_key (str): Заголовок X-API-Key.

    Returns:
        str: Валидный токен.

    Raises:
        HTTPException: Если токен не совпадает с системным.
    """
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный API ключ"
        )
    return api_key


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    """
    Middleware для сквозного логирования HTTP-запросов.

    Генерирует уникальный request_id и прокидывает его в контекст structlog,
    чтобы все логи в рамках одного запроса можно было связать (Traceability).

    Args:
        request (Request): Объект входящего HTTP-запроса FastAPI.
        call_next (Any): Следующий обработчик в цепочке вызовов.

    Returns:
        Any: Объект ответа на HTTP-запрос.
    """
    request_id = str(uuid.uuid4())
    clear_contextvars()
    bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# RFC 9457 Обработчик ошибок
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Перехватывает ошибки Pydantic и форматирует их по стандарту RFC 9457.

    Вшивает прямую инструкцию для LLM-агента в поле `detail`,
    чтобы он мог автономно исправить свой запрос.

    Args:
        request (Request): Исходный HTTP-запрос.
        exc (RequestValidationError): Исключение, выброшенное Pydantic.

    Returns:
        JSONResponse: Стандартизированный ответ с ошибкой 422.
    """
    errors = exc.errors()

    llm_instructions = "Твой JSON-запрос не прошел валидацию. "
    for err in errors:
        loc = " -> ".join(str(part) for part in err["loc"])
        msg = err["msg"]
        llm_instructions += f"Ошибка в поле '{loc}': {msg}. "

    llm_instructions += settings.LLM_INSTRUCTION_VALIDATION

    safe_errors = [
        {"loc": err.get("loc", []), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in errors
    ]

    error_response = RFC9457Error(
        type=settings.RFC_TYPE_VALIDATION,
        title="Unprocessable Entity (Validation Error)",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=llm_instructions,
        instance=str(request.url),
        errors=safe_errors,
    )

    logger.warning("agent_validation_error", url=str(request.url), errors=safe_errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(mode="json"),
    )


# API Эндпоинты
@app.post(
    "/api/v1/search",
    summary="Гибридный поиск по базе знаний",
    response_model=SearchResponse,
    dependencies=[Depends(get_api_key)],
    responses={
        401: {"description": "Неавторизованный доступ"},
        422: {"model": RFC9457Error, "description": "Ошибка валидации запроса"},
        500: {"model": RFC9457Error, "description": "Внутренняя ошибка сервера"},
    },
)
async def search_documents(
    request: SearchRequest, http_request: Request, store: AsyncQdrantStore = Depends(get_store)
) -> Any:
    """
    Точка входа для агентов. Выполняет гибридный поиск (Dense+Sparse) с алгоритмом RRF.

    Args:
        request (SearchRequest): Параметры поискового запроса.
        http_request (Request): Исходный HTTP-запрос для формирования инстанса ошибки.
        store (AsyncQdrantStore): Клиент для взаимодействия с векторной базой данных.

    Returns:
        SearchResponse: Найденные и отранжированные чанки.
    """
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

        return SearchResponse(
            status="success", query=request.query, count=len(results), data=results
        )
    except Exception as e:
        logger.exception("hybrid_search_failed", query=request.query, error=str(e), exc_info=True)

        error_response = RFC9457Error(
            type=settings.RFC_TYPE_INTERNAL,
            title="Internal Server Error",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.LLM_INSTRUCTION_INTERNAL,
            instance=str(http_request.url),
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode="json"),
        )


@app.post(
    "/api/v1/documents",
    summary="Запуск чанкинга и векторизации обработанных документов",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    dependencies=[Depends(get_api_key)],
    responses={
        401: {"description": "Неавторизованный доступ"},
    },
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Отправляет задачу векторизации в воркер TaskIQ.

    Args:
        request (IngestRequest): Запрос со списком путей к PDF-файлам.

    Returns:
        IngestResponse: Ответ с идентификаторами созданных задач.
    """
    task_ids = []

    for file_path in request.file_paths:
        task = await vectorize_document_task.kiq(file_path)
        task_ids.append(task.task_id)

    logger.info("ingest_tasks_created", task_ids=task_ids, tasks_count=len(task_ids))

    return IngestResponse(
        status="accepted",
        message=f"{len(task_ids)} задач на векторизацию успешно добавлены в очередь.",
        task_ids=task_ids,
    )
