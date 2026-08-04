from langchain_google_genai import ChatGoogleGenerativeAI
import config
from state import WriterOutput
from typing import List, Dict


def format_chunks_for_writer(chunks: list[dict]) -> str:
    formatted_context = []

    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["id"]
        text = chunk["text"]
        meta = chunk.get("metadata", {})

        title = meta.get("title", "Unknown Title")
        authors = meta.get("authors", [])
        author_str = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "Unknown Author")
        year = meta.get("year", "Unknown Year")
        heading = meta.get("original_heading_path", "Unknown Section")

        chunk_str = f"--- ИСТОЧНИК {i} [ID: {chunk_id}] ---\n"
        chunk_str += f"Текст:\n{text}\n"

        if meta.get("contains_table") and not meta.get("has_broken_table"):
            raw_table = meta.get("raw_table_markup")
            if raw_table:
                chunk_str += f"\n[!!! ВАЖНО: В ЭТОМ ИСТОЧНИКЕ ЕСТЬ ТАБЛИЦА !!!]\nТы ОБЯЗАН использовать эту таблицу переводя ее в строгий Markdown, если ссылаешься на этот текст и она не использовалась до этого:\n{raw_table}\n"

        if meta.get("contains_math") and not meta.get("has_broken_math"):
            raw_math = meta.get("raw_math_markup", [])
            if raw_math:
                math_str = "\n".join(raw_math)
                chunk_str += f"\nМатематические формулы из источника:\n{math_str}\n"

        linked_images = meta.get("linked_images", {})
        if linked_images:
            chunk_str += "\n[!!! ВАЖНО: К ЭТОМУ ТЕКСТУ ЕСТЬ ИЛЛЮСТРАЦИИ !!!]\nТы ОБЯЗАН вставить эти иллюстрации в свой текст:\n"
            for fig_name, img_path in linked_images.items():
                chunk_str += f"- {fig_name}: Вставляй строго как |IMAGE: {img_path}|\n"

        formatted_context.append(chunk_str)

    return "\n\n".join(formatted_context)


def generate_section_draft(
    title: str,
    instructions: str,
    memory_bank: List[Dict],
    previous_summary: str,
    citation_errors: str = "",
) -> WriterOutput:
    if not memory_bank:
        return WriterOutput(
            section_title=title, content="No valid context found.", used_chunk_ids=[]
        )

    context_str = format_chunks_for_writer(memory_bank)

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(WriterOutput)

    prompt = f"""
Ты — академический писатель элитного журнала Chemical Reviews.
Твоя задача — написать глубокий аналитический раздел "{title}" для монументального научного обзора.

ЗАДАЧА РАЗДЕЛА:
{instructions}

ГЛОБАЛЬНЫЙ КОНТЕКСТ (Уже написано в прошлых разделах — СТРОГО ЗАПРЕЩЕНО ПОВТОРЯТЬ ЭТИ ТЕЗИСЫ И ВВОДНЫЕ ФРАЗЫ):
{previous_summary if previous_summary else "Это первый раздел обзора."}

====================
ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ (ФАКТЫ):
{context_str}
====================

ВНИМАНИЕ! СТРОГИЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ):
1. АНТИ-АМНЕЗИЯ И СТИЛЬ: Ты пишешь аналитику, а не каталог фактов. Выстраивай причинно-следственные связи ("Метод А имел недостаток Х, поэтому разработали метод В"). ЗАПРЕЩЕНО использовать банальные перечисления (Furthermore, Moreover, In addition, Similarly) чаще 1 раза на раздел.
2. ЦИТИРОВАНИЕ: Подтверждай каждое научное утверждение. ИСПОЛЬЗУЙ ТОЛЬКО ТЕГИ ФОРМАТА: [ID: УНИКАЛЬНЫЙ_ХЭШ]. Никаких [1] или надстрочных индексов.
3. ТАБЛИЦЫ: Если в Базе есть таблица, проанализируй её. Если это оглавление с номерами страниц (вида "1. Introduction | 12869") — ИГНОРИРУЙ ЕЁ. Вставляй только таблицы с реальными научными данными (концентрации, КПД и т.д.).
4. ИЛЛЮСТРАЦИИ: Картинки должны органично дополнять текст в строгом формате: |IMAGE: путь|, вставляй их в конце утверждения. АБСОЛЮТНО ЗАПРЕЩЕНЫ фразы вида "Рисунок 1 показывает...", "Table 2 summarizes...". Пиши так(пример): "...что подтверждается высокой эффективностью разделения зарядов |IMAGE: путь|".
5. ZERO HALLUCINATIONS: Не выдумывай числа, pH, температуры или ID чанков.
"""

    if citation_errors:
        prompt += f"\n\n!!! КРИТИКА ПРОШЛОЙ ПОПЫТКИ !!!\n{citation_errors}\nИСПРАВЬ ОШИБКИ И СГЕНЕРИРУЙ ТЕКСТ ЗАНОВО! ВАЖНО: Сохраняй изначальный объем, глубину анализа и детализацию. Не вздумай удалять целые абзацы только ради того, чтобы избавиться от плохой цитаты. Найди правильный ID в Базе Знаний или переформулируй предложение."

    return structured_llm.invoke(prompt)