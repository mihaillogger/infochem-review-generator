from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict, Dict


class SubSectionPlan(BaseModel):
    section_number: str = Field(description="Номер секции")
    title: str = Field(description="Название секции")
    target_paths: List[str] = Field(description="Точные пути из Дерева Контента.")
    instructions: str = Field(description="Инструкции для writer_node")


class PlanOutput(BaseModel):
    sections: List[SubSectionPlan] = Field(description="Список секций обзора")


class WriterOutput(BaseModel):
    section_title: str = Field(description="Название секции")
    content: str = Field(description="Сгенерированный текст с тегами [ID: xxx]")
    used_chunk_ids: List[str] = Field(description="Список ID чанков, реально использованных в тексте")


class SectionState(TypedDict):
    section_id: str
    title: str
    instructions: str
    target_paths: List[str]

    # Adapter & Memory Bank
    search_queries: List[str]  # История попыток поиска
    raw_chunks: List[Dict]  # То, что пришло из БД
    memory_bank: List[Dict]  # Валидированные чанки (Pass)
    retriever_rejection: str  # Критика от Eval_Retriever
    retriever_retries: int  # Счетчик попыток Adapter

    # Writer & Citation
    draft_content: str  # Черновик до проверки
    draft_used_ids: List[str]  # ID в черновике
    citation_errors: str  # Критика от Eval_Citation
    writer_retries: int  # Счетчик попыток Writer

    # Final Approved Content
    content: str
    used_chunk_ids: List[str]


class GraphState(TypedDict):
    global_topic: str
    pending_sections: List[SectionState]
    current_section: Optional[SectionState]
    completed_sections: List[SectionState]
    final_document: str
    previous_sections_summary: str