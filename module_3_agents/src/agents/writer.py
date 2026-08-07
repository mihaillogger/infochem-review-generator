import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
import config
from state import WriterOutput
from typing import List, Dict


def format_chunks_for_writer(chunks: list[dict]) -> str:
    formatted_context = []

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        meta = chunk.get("metadata", {})

        chunk_str = f"--- ИСТОЧНИК {i} [ID: {i}] ---\n"
        chunk_str += f"Текст:\n{text}\n"

        if meta.get("contains_table") and not meta.get("has_broken_table"):
            raw_table = meta.get("raw_table_markup")
            if raw_table:
                chunk_str += f"\n[!!! ВАЖНО: В ЭТОМ ИСТОЧНИКЕ ЕСТЬ ТАБЛИЦА !!!]\nТы ОБЯЗАН использовать эту таблицу переводя ее в строгий Markdown, если ее содержание полезно для раскрытия темы и она не использовалась до этого:\n{raw_table}\n"

        if meta.get("contains_math") and not meta.get("has_broken_math"):
            raw_math = meta.get("raw_math_markup", [])
            if raw_math:
                math_str = "\n".join(raw_math)
                chunk_str += f"\nМатематические формулы из источника:\n{math_str}\n"

        linked_images = meta.get("linked_images", {})
        if linked_images:
            chunk_str += "\n[!!! ВАЖНО: К ЭТОМУ ТЕКСТУ ЕСТЬ ИЛЛЮСТРАЦИИ !!!]\nТы ОБЯЗАН вставить эти иллюстрации в свой текст если они по описанию соответствуют теме:\n"
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

    # СОЗДАЕМ ЛОКАЛЬНЫЙ СЛОВАРЬ (МАППИНГ): "1" -> "uuid-123..."
    id_map = {str(i): str(chunk["id"]) for i, chunk in enumerate(memory_bank, 1)}

    context_str = format_chunks_for_writer(memory_bank)

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(WriterOutput)

    prompt = f"""
Ты — академический писатель элитного журнала Chemical Reviews.
Твоя задача — написать глубокий аналитический раздел "{title}" для монументального научного обзора на английском языке.

ЗАДАЧА РАЗДЕЛА:
{instructions}

ГЛОБАЛЬНЫЙ КОНТЕКСТ (Уже написано в прошлых разделах — СТРОГО ЗАПРЕЩЕНО ПОВТОРЯТЬ ЭТИ ТЕЗИСЫ И ВВОДНЫЕ ФРАЗЫ):
{previous_summary if previous_summary else "Это первый раздел обзора."}

====================
ПРЕДОСТАВЛЕННАЯ БАЗА ЗНАНИЙ (ФАКТЫ):
{context_str}
====================

ВНИМАНИЕ! СТРОГИЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ):
0. НЕ ФАНТАЗИРОВАТЬ: Используй только факты предоставленные в базе знаний, никакой отсебятины.
1. АНТИ-АМНЕЗИЯ И СТИЛЬ: Ты пишешь аналитику, а не каталог фактов. Выстраивай причинно-следственные связи. ЗАПРЕЩЕНО использовать банальные перечисления (Furthermore, Moreover, In addition, Similarly) чаще 1 раза на раздел.
2. ЦИТИРОВАНИЕ: Подтверждай каждое научное утверждение. ИСПОЛЬЗУЙ ТОЛЬКО ТЕГИ ФОРМАТА: [ID: ЧИСЛО] (например, [ID: 1], [ID: 2]). Никаких [1] или надстрочных индексов.
3. ТАБЛИЦЫ: Если в Базе есть таблица, проанализируй её. Игнорируй оглавления.
4. ИЛЛЮСТРАЦИИ: Вставляй их строго как |IMAGE: путь| в конце утверждения (не пиши "Table 1 shows").
5. ZERO HALLUCINATIONS: Не выдумывай числа, pH или температуры, используй только если они прописаны в исходной базе.
6. АНТИ-СПАМ: СТРОГО ЗАПРЕЩЕНО генерировать пустые LaTeX-макросы (например, \\text{{ }}). Для пустых ячеек таблиц используй прочерк (-).
7. ФОРМАТ ТЕГОВ: ЗАПРЕЩЕНО группировать источники в одних скобках. Пиши строго раздельно: [ID: 1][ID: 2]. ЗАПРЕЩЕНО писать [ID: 1, 2].
8. ПОЗИЦИЯ АВТОРА: Пиши СТРОГО как сторонний обозреватель. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать фразы "in this work", "our results", "present study". Все упоминаемые исследования принадлежат другим ученым.
"""

    if citation_errors:
        prompt += f"\n\n!!! КРИТИКА ПРОШЛОЙ ПОПЫТКИ !!!\n{citation_errors}\nИСПРАВЬ ОШИБКИ И СГЕНЕРИРУЙ ТЕКСТ ЗАНОВО!"

    try:
        draft = structured_llm.invoke(prompt)
        def replacer(match):
            num = match.group(1).strip()
            if num in id_map:
                return f"[ID: {id_map[num]}]"
            return match.group(0)

        restored_content = re.sub(r"\[ID:\s*(\d+)\s*\]", replacer, draft.content)

        restored_ids = []
        for cid in draft.used_chunk_ids:
            cid_str = str(cid).strip()
            cid_str = cid_str.replace("ID:", "").strip()
            if cid_str in id_map:
                restored_ids.append(id_map[cid_str])
            else:
                restored_ids.append(cid_str)

        return WriterOutput(
            section_title=draft.section_title,
            content=restored_content,
            used_chunk_ids=restored_ids
        )

    except (OutputParserException, ValidationError, Exception) as e:
        print(f"[WRITER CATCH] Ошибка парсинга или лимит токенов: {e}")
        return WriterOutput(
            section_title=title,
            content="КРИТИЧЕСКАЯ ОШИБКА: Обрыв генерации по лимиту токенов или зацикливание.",
            used_chunk_ids=[]
        )