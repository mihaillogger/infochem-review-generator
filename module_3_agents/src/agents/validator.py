import config
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI


class ChunkEval(BaseModel):
    chunk_id: str = Field(description="ID проверяемого фрагмента")
    is_relevant: bool = Field(
        description="True, если текст содержит точный ответ. False, если вода."
    )
    reason: str = Field(description="Причина отказа, если False. Иначе пустая строка.")


class BatchRelevanceScore(BaseModel):
    evaluations: List[ChunkEval] = Field(
        description="Оценки для каждого переданного фрагмента текста"
    )


def check_relevance_batch(query: str, chunks_batch: List[Dict]) -> Dict[str, Dict]:
    """
    Принимает массив чанков, возвращает словарь: { "chunk_id": {"is_relevant": bool, "reason": str} }
    """
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)

    structured_llm = llm.with_structured_output(BatchRelevanceScore)

    chunks_text = ""
    for chunk in chunks_batch:
        chunks_text += f"\n--- CHUNK ID: {chunk['id']} ---\n{chunk['text']}\n"

    prompt = f"""
        Ты — академический фильтр данных (Validator). Твоя задача — оценить релевантность фрагментов текста для ответа на запрос.

        ЗАПРОС (Что мы ищем):
        {query}

        ФРАГМЕНТЫ ТЕКСТА:
        {chunks_text}

        ИНСТРУКЦИЯ:
        Проанализируй КАЖДЫЙ фрагмент. Верни список оценок, строго привязанный к CHUNK ID.
        КРИТЕРИЙ ПРОПУСКА (is_relevant: True): Пропускай текст, если он содержит фактическую базу, методологию, важный научный бэкграунд или результаты экспериментов, хотя бы косвенно связанные с запросом. Ищи семантическое совпадение.
        КРИТЕРИЙ ОТКАЗА (is_relevant: False): Отбраковывай ТОЛЬКО абсолютный мусор (оглавления, списки литературы, битые куски PDF, текст совершенно из другой области химии/физики).
        """
    try:
        result = structured_llm.invoke(prompt)
        return {
            item.chunk_id: {"is_relevant": item.is_relevant, "reason": item.reason}
            for item in result.evaluations
        }
    except Exception as e:
        print(f"[ERROR] Ошибка валидатора (батч): {e}")
        return {}
