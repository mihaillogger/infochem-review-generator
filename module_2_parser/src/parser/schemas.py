"""Pydantic-схемы для парсера."""

from enum import StrEnum

from pydantic import BaseModel, Field


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
