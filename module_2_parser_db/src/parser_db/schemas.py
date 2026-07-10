from pydantic import BaseModel, Field


class VisualMeta(BaseModel):
    """Метаданные визуального элемента (картинки или таблицы)."""

    id: str = Field(..., description="Идентификатор в тексте, например 'Fig. 1' или 'Table 2'")
    path: str = Field(..., description="Абсолютный путь к сохраненному файлу в /data/images/")
    caption: str | None = Field(default=None, description="Распознанная подпись к элементу")


class Section(BaseModel):
    """Логический раздел документа."""

    heading: str = Field(..., description="Заголовок раздела (например, '1. Introduction')")
    level: int = Field(..., description="Уровень вложенности заголовка (1 для H1, 2 для H2 и т.д.)")
    paragraphs: list[str] = Field(
        ..., description="Список сырых абзацев текста, принадлежащих разделу"
    )


class ParsedDocument(BaseModel):
    """
    Финальный выходной объект парсера.
    """

    doi: str
    title: str = Field(..., description="Название научной статьи")
    authors: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(..., description="Иерархическая структура текста")
    visuals: list[VisualMeta] = Field(
        default_factory=list, description="Все найденные графические элементы"
    )


class DBChunkMetadata(BaseModel):
    """
    Метаданные чанка для сохранения в ChromaDB.
    """

    doi: str
    section_path: str = Field(
        ..., description="Путь заголовков, например 'Introduction > Background'"
    )
    linked_images: str = Field(
        default="none", description="Строка с путями к картинкам через запятую"
    )


class DBChunk(BaseModel):
    """
    Объект чанка, готовый к загрузке в БД.
    """

    chunk_id: str
    text: str = Field(..., description="Связный кусок текста (один или несколько абзацев)")
    metadata: DBChunkMetadata
