import os
import config
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class SearchQuery(BaseModel):
    query: str = Field(
        description="Короткий и точный поисковый запрос (5-10 слов) для векторной базы данных."
    )

def generate_search_query(task: str, previous_queries: list[str] = None, rejection_reason: str = None) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.3 
    )

    structured_llm = llm.with_structured_output(SearchQuery)
    
    # Защита от None
    previous_queries = previous_queries or []

    prompt = f"""
    Ты — исследователь-адаптер (Adapter). Твоя задача — составить точный поисковый запрос для семантического поиска в векторной базе данных.
    
    ГЛОБАЛЬНАЯ ЗАДАЧА:
    {task}
    """

    if rejection_reason and previous_queries:
        prompt += f"""
        ПРЕДЫДУЩИЕ ПОПЫТКИ: {', '.join(previous_queries)}
        
        ОТВЕТ ВАЛИДАТОРА (Критика): "{rejection_reason}"
        
        ИНСТРУКЦИЯ:
        Напиши НОВЫЙ поисковый запрос. Учти критику валидатора: используй другие синонимы, сузь контекст поиска. 
        Не повторяй старые запросы!
        """
    else:
        prompt += """
        ИНСТРУКЦИЯ:
        Напиши один емкий поисковый запрос, который лучше всего вытащит релевантную информацию.
        """

    result = structured_llm.invoke(prompt)
    
    return result.query