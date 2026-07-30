from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict

# КОНТРАКТЫ (API)

class SubSectionPlan(BaseModel):
    section_number: str = Field(description="Номер секции (например, '1', '2.1')")
    title: str = Field(description="Название секции в академическом стиле")
    target_paths: List[str] = Field(description="ТОЧНЫЕ пути из Дерева Контента.")
    instructions: str = Field(description="Инструкции для writer_node")

class PlanOutput(BaseModel):
    sections: List[SubSectionPlan] = Field(description="Список секций обзора")

class WriterOutput(BaseModel):
    """
    ЖЕСТКИЙ КОНТРАКТ ДЛЯ МАТВЕЯ.
    Его API обязано возвращать JSON, строго соответствующий этой схеме.
    """
    section_title: str = Field(description="Название секции")
    content: str = Field(description="Сгенерированный текст с тегами [ID: xxx]")
    used_chunk_ids: List[str] = Field(description="Список ID чанков, реально использованных в тексте")


# СОСТОЯНИЕ ГРАФА (STATE)

class SectionState(TypedDict):
    section_id: str
    title: str
    instructions: str
    target_paths: List[str]  # Передаем для Retrieval
    content: str             # Заполняется из WriterOutput.content
    used_chunk_ids: List[str] # Заполняется из WriterOutput.used_chunk_ids

class GraphState(TypedDict):
    global_topic: str
    pending_sections: List[SectionState]
    current_section: Optional[SectionState]
    completed_sections: List[SectionState]
    final_document: str