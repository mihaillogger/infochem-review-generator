from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
import config
import db_connector

class WriterOutput(BaseModel):
    section_title: str = Field(description="Название секции")
    content: str = Field(description="Сгенерированный текст с тегами [ID: xxx]")
    used_chunk_ids: List[str] = Field(description="Список ID чанков, реально использованных в тексте")

def generate_section(title: str, instructions: str, target_paths: List[str]) -> WriterOutput:
    raw_chunks = db_connector.fetch_chunks_by_paths(target_paths)
    
    if not raw_chunks:
        return WriterOutput(
            section_title=title,
            content=f"Error: Could not retrieve data for paths: {target_paths}",
            used_chunk_ids=[]
        )

    context_str = ""
    valid_ids = []
    
    for chunk in raw_chunks:
        chunk_id = chunk.get("id", "unknown")
        text = chunk.get("text", "")
        context_str += f"--- SOURCE FACT [ID: {chunk_id}] ---\n{text}\n\n"
        valid_ids.append(chunk_id)

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, 
        temperature=0.1 
    )
    structured_llm = llm.with_structured_output(WriterOutput)

    prompt = f"""
    Ты — академический писатель элитного журнала Chemical Reviews.
    Твоя задача — написать раздел "{title}" для монументального научного обзора.

    ИНСТРУКЦИИ ПЛАНИРОВЩИКА:
    {instructions}

    ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ (ФАКТЫ):
    {context_str}

    СТРОГИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ И СТИЛЯ:
    1. Тон: Authoritative, in-depth, balanced. Избегай банальных вводных слов. Никакого жаргона без предварительного объяснения.
    2. Фокус: Пиши густой научный текст. Описывай механизмы, цифры, результаты.
    3. ЦИТИРОВАНИЕ (КРИТИЧЕСКИ ВАЖНО): Ты ОБЯЗАН подтверждать каждое научное утверждение ссылкой на предоставленные факты. 
       Используй ТОЛЬКО теги формата [ID: xxx].
       Пример правильного цитирования: "Эти катализаторы демонстрируют высокую селективность [ID: a1b2][ID: 9f8c]."
    4. Запрет галлюцинаций: Используй только те знания и те ID, которые есть в блоке ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ.
    """

    result = structured_llm.invoke(prompt)
     
    actual_used_ids = [cid for cid in result.used_chunk_ids if cid in valid_ids]
    result.used_chunk_ids = actual_used_ids

    return result