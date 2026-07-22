"""API-шлюз Модуля 2.

Реализует REST-интерфейс для поиска по базе знаний и асинхронного парсинга PDF.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from parser_db.broker import broker
from parser_db.store import get_store
from parser_db.worker import parse_pdf_task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управляет жизненным циклом приложения FastAPI.

    Открывает соединение брокера TaskIQ с Redis при старте сервера
    и корректно закрывает его при остановке.

    Args:
        app: Экземпляр приложения FastAPI.
    """
    await broker.startup()
    yield
    await broker.shutdown()


app = FastAPI(
    title="Infochem RAG Core API",
    description="Ядро семантического поиска и парсинга научных статей.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Строгие контракты (Enum) ---


class StandardSection(StrEnum):
    """Стандартизированные разделы научных статей для фильтрации."""

    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    METHODOLOGY = "Methodology"
    RESULTS = "Results"
    CONCLUSION = "Conclusion"
    DISCUSSION = "Discussion"


# --- Схемы запросов (Pydantic Контракты) ---


class SearchRequest(BaseModel):
    """Схема запроса для поиска фактов LLM-агентом."""

    query: str = Field(
        ...,
        description="Поисковый запрос на естественном языке "
        "(например, 'методы синтеза перовскитов').",
    )
    limit: int = Field(default=5, ge=1, le=20, description="Количество возвращаемых чанков.")
    doi_filter: str | None = Field(default=None, description="Ограничить поиск конкретным DOI.")
    section_filter: StandardSection | None = Field(
        default=None,
        description="Искать только в определенном разделе. "
        "Используй строго одно из доступных значений.",
    )
    require_table: bool = Field(
        default=False, description="Вернуть только те чанки, которые содержат таблицы."
    )
    require_math: bool = Field(
        default=False, description="Вернуть только те чанки, которые содержат формулы."
    )


class IngestRequest(BaseModel):
    """Схема запроса от Модуля 1 на старт парсинга."""

    file_paths: list[str] = Field(
        ..., description="Список абсолютных путей к скачанным PDF в томе /data/pdfs/."
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

    llm_instructions += (
        "Изучи OpenAPI спецификацию этого метода, исправь тип данных и повтори вызов."
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "https://datatracker.ietf.org/doc/html/rfc9457#section-3",
            "title": "Unprocessable Entity (Validation Error)",
            "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "detail": llm_instructions,
            "instance": str(request.url),
            "errors": errors,
        },
    )


# --- API Эндпоинты ---


@app.post("/api/v1/search", summary="Гибридный поиск по базе знаний")
async def search_documents(request: SearchRequest, http_request: Request) -> Any:
    """Точка входа для агентов. Выполняет поиск Dense+Sparse с алгоритмом RRF."""
    try:
        # БД инициализируется только в момент реального запроса
        store = get_store()

        section_val = request.section_filter.value if request.section_filter else None

        results = store.hybrid_search(
            query=request.query,
            limit=request.limit,
            doi_filter=request.doi_filter,
            section_filter=section_val,
            require_table=request.require_table,
            require_math=request.require_math,
        )
        return {"status": "success", "count": len(results), "data": results}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": f"Внутренняя ошибка векторной БД: {str(e)}. "
                f"Попробуй изменить параметры запроса.",
                "instance": str(http_request.url),
            },
        )


@app.post(
    "/api/v1/documents",
    summary="Запуск индексации PDF (Асинхронно)",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_documents(request: IngestRequest) -> dict[str, str]:
    """Точка входа для Модуля 1. Отправляет задачу в воркер TaskIQ."""
    task = await parse_pdf_task.kiq(request.file_paths)

    return {
        "status": "accepted",
        "message": "Задачи на парсинг успешно добавлены в очередь.",
        "task_id": task.task_id,
    }
