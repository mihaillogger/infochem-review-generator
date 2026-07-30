import os
import config
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class RelevanceScore(BaseModel):
    is_relevant: bool = Field(
        description="True, если текст содержит прямой и точный ответ на запрос. False, если это вода или не по теме."
    )
    reason: str = Field(
        description="Если is_relevant=False, напиши кратко, почему текст забракован (например: 'В тексте только история вопроса, нет математических формул'). Если True, оставь строку пустой."
    )

def check_relevance(query: str, chunk: str) -> dict:
    #температура 0 для строгости
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.0
    )

    structured_llm = llm.with_structured_output(RelevanceScore)

    prompt = f"""
    Ты — строгий академический фильтр данных (Validator). Твоя задача — оценить, содержит ли предоставленный фрагмент текста конкретную информацию для ответа на запрос.
    
    ЗАПРОС (Что мы ищем):
    {query}

    ФРАГМЕНТ ТЕКСТА (Что выдала база данных):
    {chunk}

    ИНСТРУКЦИЯ:
    Проанализируй текст. Если в нем есть фактическая база (данные, алгоритмы, выводы), отвечающая на запрос — верни True.
    Если текст содержит только общие слова, введения, нерелевантную информацию или не отвечает на суть запроса — верни False и укажи причину отказа.
    """

    result = structured_llm.invoke(prompt)
    
    return {
        "is_relevant": result.is_relevant,
        "reason": result.reason
    }