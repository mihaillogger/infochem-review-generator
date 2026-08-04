from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import config


class SearchQuery(BaseModel):
    query: str = Field(
        description="Короткий и точный поисковый запрос (5-7 слов) для векторной базы."
    )


def generate_search_query(
    task: str, previous_queries: list[str] = None, rejection_reason: str = None
) -> str:
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(SearchQuery)

    previous_queries = previous_queries or []

    prompt = f"Ты — исследователь-адаптер (Adapter). Твоя задача — составить точный запрос для БД с сематическим поиском(то есть запросы должны быть не слишком узкими, но по теме).\nГЛОБАЛЬНАЯ ЗАДАЧА:\n{task}\n"

    if rejection_reason and previous_queries:
        prompt += f"""
            ПРЕДЫДУЩИЕ ПОПЫТКИ: {", ".join(previous_queries)}

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
