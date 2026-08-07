from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict, Dict


class SubSectionPlan(BaseModel):
    section_number: str = Field(description="Номер подглавы (например, '2.1', '3.1.2')")
    title: str = Field(description="Название подглавы")
    target_paths: List[str] = Field(description="Точные пути из Дерева Контента.")
    instructions: str = Field(description="Инструкции для writer_node с лимитом генерации 800-1000 слов")


class ChapterPlan(BaseModel):
    chapter_number: str = Field(description="Номер главы (например, '1', '2')")
    title: str = Field(description="Название корневой главы")
    instructions: str = Field(
        default="",
        description="Инструкции для генерации, ЕСЛИ глава НЕ имеет подглав (например, Abstract или Introduction)."
    )
    target_paths: List[str] = Field(
        default_factory=list,
        description="Пути из Дерева Контента, ЕСЛИ глава НЕ имеет подглав."
    )
    subsections: List[SubSectionPlan] = Field(
        default_factory=list,
        description="Вложенные подглавы. Оставь пустым для монолитных глав (Введение, Заключение)."
    )


class PlanOutput(BaseModel):
    chapters: List[ChapterPlan] = Field(description="Список глав обзора")


class WriterOutput(BaseModel):
    section_title: str = Field(description="Название секции")
    content: str = Field(description="Сгенерированный текст с тегами [ID: xxx]")
    used_chunk_ids: List[str] = Field(
        description="Список ID чанков, реально использованных в тексте"
    )


class SectionState(TypedDict):
    section_id: str
    title: str
    instructions: str
    target_paths: List[str]

    search_queries: List[str]  # История попыток поиска
    raw_chunks: List[Dict]  # То, что пришло из БД
    memory_bank: List[Dict]  # Валидированные чанки (Pass)
    retriever_rejection: str  # Критика от Eval_Retriever
    retriever_retries: int  # Счетчик попыток Adapter

    draft_content: str  # Черновик до проверки
    draft_used_ids: List[str]  # ID в черновике
    citation_errors: str  # Критика от Eval_Citation
    writer_retries: int  # Счетчик попыток Writer

    content: str
    used_chunk_ids: List[str]


class GraphState(TypedDict):
    global_topic: str
    pending_sections: List[SectionState]
    current_section: Optional[SectionState]
    completed_sections: List[SectionState]
    final_document: str
    previous_sections_summary: str