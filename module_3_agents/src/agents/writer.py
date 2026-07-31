from langchain_google_genai import ChatGoogleGenerativeAI
import config
from state import WriterOutput
from typing import List, Dict

def generate_section_draft(title: str, instructions: str, memory_bank: List[Dict], previous_summary: str, citation_errors: str = "") -> WriterOutput:
    if not memory_bank:
        return WriterOutput(section_title=title, content="No valid context found.", used_chunk_ids=[])

    context_str = ""
    for chunk in memory_bank:
        context_str += f"--- SOURCE FACT [ID: {chunk['id']}] ---\n{chunk['text']}\n\n"

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(WriterOutput)

    prompt = f"""
    Ты — академический писатель элитного журнала Chemical Reviews.
    Твоя задача — написать раздел "{title}" для монументального научного обзора (целевой объем обзора ~80 страниц).

    ЗАДАЧА РАЗДЕЛА:
    {instructions}

    ЧТО УЖЕ НАПИСАНО В ПРЕДЫДУЩИХ РАЗДЕЛАХ (Не повторяй этот текст!):
    {previous_summary if previous_summary else "Это первый раздел обзора."}

    ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ (ФАКТЫ):
    {context_str}

    СТРОГИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ И СТИЛЯ:
    1. ПЕРЕИСПОЛЬЗОВАНИЕ ФАКТОВ: Если факт из Базы уже упоминался в предыдущих разделах, рассматривай его ЗДЕСЬ СТРОГО под углом текущей ЗАДАЧИ РАЗДЕЛА. Не дублируй вводную информацию.
    2. ТОН: Authoritative, in-depth, balanced. Уровень химических наук. Jargon-free для широкой научной аудитории, фундаментальные идеи объясняются четко.
    3. ЦИТИРОВАНИЕ (КРИТИЧЕСКИ ВАЖНО): Подтверждай КАЖДОЕ научное утверждение ссылкой на предоставленные факты. Используй ТОЛЬКО теги формата [ID: xxx].
    4. Запрет галлюцинаций: Используй только те знания и те ID, которые есть в блоке ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ.
    """

    if citation_errors:
        prompt += f"\nКРИТИКА ПРОШЛОЙ ПОПЫТКИ: {citation_errors}. ИСПРАВЬ ОШИБКИ ЦИТИРОВАНИЯ!"

    return structured_llm.invoke(prompt)