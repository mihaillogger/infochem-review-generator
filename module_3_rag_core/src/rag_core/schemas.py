"""Pydantic-схемы для RAG Core."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from rag_core.config import settings


class VisualMeta(BaseModel):
    """Метаданные визуального элемента (картинки или таблицы)."""

    id: str = Field(..., description="Идентификатор в тексте, например 'Fig. 1'")
    path: str = Field(..., description="Абсолютный путь к файлу в /data/images/")
    caption: str | None = Field(default=None, description="Распознанная подпись")


class Paragraph(BaseModel):
    """Абзац документа."""

    type: str = Field(..., description="Тип контента: 'text', 'table' или 'equation'")
    content: str = Field(..., description="Сам текст, markdown или latex")
    is_broken: bool = Field(default=False, description="Флаг кривого парсинга для VLM-агента")
    image_fallback_path: str | None = Field(
        default=None, description="Путь к картинке для восстановления VLM-агентом"
    )
    enriched_summary: str | None = Field(
        default=None,
        description="Текстовое описание таблицы от LLM.",
    )


class StandardSection(StrEnum):
    """Стандартизированные разделы научных статей для фильтрации."""

    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    CONCEPTS_AND_MECHANISMS = "Concepts & Mechanisms"
    MATERIALS_AND_SYNTHESIS = "Materials & Synthesis"
    APPLICATIONS = "Applications & Devices"
    PERSPECTIVES_AND_CONCLUSIONS = "Perspectives & Conclusions"
    UNKNOWN = "Unknown"


class Section(BaseModel):
    """Логический раздел документа."""

    original_heading: str = Field(
        ..., description="Исходный текст заголовка из PDF (например, '3. Encapsulation')"
    )
    macro_category: StandardSection = Field(
        ..., description="Стандартизированный класс (например, 'Concepts & Mechanisms')"
    )
    level: int = Field(..., description="Уровень вложенности заголовка (1 - H1, 2 - H2)")
    paragraphs: list[Paragraph] = Field(..., description="Список абзацев раздела")


class ParsedDocument(BaseModel):
    """Финальный выходной объект парсера."""

    doi: str = Field(..., description="Идентификатор документа DOI")
    title: str | None = Field(default=None, description="Заголовок научной статьи")
    authors: list[str] = Field(default_factory=list, description="Список авторов статьи")
    year: int | None = Field(default=None, description="Год публикации статьи")
    journal: str | None = Field(default=None, description="Название журнала")
    abstract: str | None = Field(default=None, description="Абстракт статьи")
    sections: list[Section] = Field(..., description="Иерархическая структура текста")
    visuals: list[VisualMeta] = Field(default_factory=list, description="Все графические элементы")


class DBChunkMetadata(BaseModel):
    """Метаданные чанка для сохранения в БД."""

    doi: str
    title: str | None = Field(default=None, description="Заголовок научной статьи")
    authors: list[str] = Field(default_factory=list, description="Список авторов статьи")
    year: int | None = Field(default=None, description="Год публикации статьи")
    journal: str | None = Field(default=None, description="Название журнала")
    abstract: str | None = Field(default=None, description="Абстракт статьи")
    original_heading_path: str = Field(
        ..., description="Путь оригинальных заголовков из PDF (например, '1. Intro > 1.1 Goals')"
    )
    macro_category_path: str = Field(
        ..., description="Путь стандартизированных макро-категорий (для точной фильтрации)"
    )
    linked_images: dict[str, str] = Field(
        default_factory=dict,
        description="Словарь путей к картинкам "
        "(ключ - идентификатор картинки в тексте, например 'Fig. 1', "
        "значение - абсолютный путь к файлу в /data/images/)",
    )
    contains_table: bool = Field(default=False, description="Флаг: есть ли в этом чанке таблица")
    contains_math: bool = Field(
        default=False, description="Флаг: есть ли в этом чанке математика/формулы (LaTeX)"
    )
    raw_table_markup: str | None = Field(
        default=None, description="Оригинальный код таблицы (если contains_table=True)"
    )
    raw_math_markup: list[str] | None = Field(
        default=None,
        description="Список оригинальных LaTeX формул в чанке (если contains_math=True)",
    )
    has_broken_table: bool = Field(default=False, description="Флаг: есть ли в чанке битая таблица")
    has_broken_math: bool = Field(default=False, description="Флаг: есть ли в чанке битая формула")
    has_broken_text: bool = Field(
        default=False,
        description="Флаг: был ли текст аварийно разорван по токенам "
        "(возможен мусор/обрезанные слова)",
    )
    fallback_table_paths: list[str] = Field(
        default_factory=list,
        description="Пути к картинкам битых таблиц для восстановления VLM-агеном "
        "(если has_broken_table=True)",
    )
    fallback_math_paths: list[str] = Field(
        default_factory=list,
        description="Пути к картинкам битых формул для восстановления VLM-агентов "
        "(если has_broken_math=True)",
    )


class DBChunk(BaseModel):
    """Объект чанка, готовый к загрузке в БД."""

    chunk_id: str
    text: str = Field(..., description="Связный кусок текста (один или несколько абзацев)")
    metadata: DBChunkMetadata


class SearchRequest(BaseModel):
    """Схема запроса для поиска фактов LLM-агентом."""

    query: str = Field(
        ...,
        description="Поисковый запрос на естественном языке "
        "(например, 'методы синтеза перовскитов').",
    )
    limit: int = Field(
        default=settings.SEARCH_DEFAULT_LIMIT,
        ge=1,
        le=settings.SEARCH_MAX_LIMIT,
        description="Количество возвращаемых чанков.",
    )
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


class SearchResponse(BaseModel):
    """Схема ответа на поиск для LLM-агента."""

    status: str
    query: str
    count: int
    data: list[DBChunk]


class IngestRequest(BaseModel):
    """Схема запроса на старт чанкинга и векторизации."""

    file_paths: list[str] = Field(
        ..., description="Список абсолютных путей к скачанным PDF в томе /data/pdfs/."
    )

    @field_validator("file_paths")
    @classmethod
    def validate_paths(cls, v: list[str]) -> list[str]:
        """Проверяет пути на наличие Directory Traversal уязвимости.

        Args:
            v (list[str]): Исходный список путей.

        Returns:
            list[str]: Очищенный и проверенный список абсолютных путей.

        Raises:
            ValueError: Если путь пытается выйти за пределы разрешенной директории.
        """
        safe_root = Path(settings.SHARED_DATA_ROOT).resolve()
        validated = []
        for p in v:
            resolved = Path(p).resolve()
            if not resolved.is_relative_to(safe_root):
                raise ValueError(f"Путь {p} выходит за пределы защищенной директории.")
            validated.append(str(resolved))
        return validated


class IngestResponse(BaseModel):
    """Схема успешного ответа на запуск векторизации."""

    status: str = Field(default="accepted")
    message: str
    task_ids: list[str]


class ParseTaskResult(BaseModel):
    """
    Схема результата выполнения фоновой задачи парсинга PDF.
    """

    status: str = Field(..., description="Статус выполнения задачи (success/failed)")
    file_path: str = Field(..., description="Путь к обрабатываемому PDF-файлу")
    error: str | None = Field(default=None, description="Текст ошибки, если она возникла")


class RFC9457Error(BaseModel):
    """
    Схема ошибки по стандарту RFC 9457 (Problem Details for HTTP APIs).
    Обеспечивает агентам предсказуемый формат для обработки ошибок и Self-Healing'а.
    """

    type: str = Field(default="about:blank", description="URI, идентифицирующий тип проблемы.")
    title: str = Field(..., description="Краткое человекочитаемое название проблемы.")
    status: int = Field(..., description="HTTP статус-код ошибки.")
    detail: str = Field(
        ..., description="Подробное описание проблемы, включая прямые инструкции для LLM-агента."
    )
    instance: str | None = Field(default=None, description="URI эндпоинта, где произошла ошибка.")
    errors: list[dict[str, Any]] | None = Field(
        default=None, description="Массив с деталями сырых ошибок валидации Pydantic."
    )
