from pydantic import BaseModel, Field


class VisualMeta(BaseModel):
    """Метаданные визуального элемента (картинки или таблицы)."""

    id: str = Field(..., description="Идентификатор в тексте, например 'Fig. 1' или 'Table 2'")
    path: str = Field(..., description="Абсолютный путь к сохраненному файлу в /data/images/")
    caption: str | None = Field(default=None, description="Распознанная подпись к элементу")


class Paragraph(BaseModel):
    """Абзац."""

    type: str = Field(..., description="Тип контента: 'text', 'table' или 'equation'")
    content: str = Field(..., description="Сам текст, markdown или latex")


class Section(BaseModel):
    """Логический раздел документа."""

    heading: str = Field(..., description="Заголовок раздела (например, '1. Introduction')")
    level: int = Field(..., description="Уровень вложенности заголовка (1 для H1, 2 для H2 и т.д.)")
    paragraphs: list[Paragraph] = Field(
        ..., description="Список абзацев с типом контента, принадлежащих разделу"
    )


class ParsedDocument(BaseModel):
    """Финальный выходной объект парсера."""

    doi: str
    title: str = Field(..., description="Название научной статьи")
    authors: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(..., description="Иерархическая структура текста")
    visuals: list[VisualMeta] = Field(
        default_factory=list, description="Все найденные графические элементы"
    )


class DBChunkMetadata(BaseModel):
    """Метаданные чанка для сохранения в БД."""

    doi: str
    section_path: str = Field(
        ..., description="Путь заголовков, например 'Introduction > Background'"
    )
    linked_images: list[str] = Field(default_factory=list, description="Список путей к картинкам")
    contains_table: bool = Field(default=False, description="Флаг: есть ли в этом чанке таблица")
    contains_math: bool = Field(
        default=False, description="Флаг: есть ли в этом чанке математика/формулы (LaTeX)"
    )
    raw_table_markup: str | None = Field(
        default=None, description="Оригинальный Markdown таблицы (если contains_table=True)"
    )
    raw_math_markup: list[str] | None = Field(
        default=None,
        description="Список оригинальных LaTeX формул в чанке (если contains_math=True)",
    )


class DBChunk(BaseModel):
    """Объект чанка, готовый к загрузке в БД."""

    chunk_id: str
    text: str = Field(..., description="Связный кусок текста (один или несколько абзацев)")
    metadata: DBChunkMetadata
