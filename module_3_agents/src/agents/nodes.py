import re
import requests
import time
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from state import GraphState, PlanOutput, SectionState
from adapter import generate_search_query
import db_connector
from writer import generate_section_draft
from validator import check_relevance_batch


def generate_short_summary(content: str) -> str:
    """Вытаскивает ключевые тезисы для запрета их повторения в будущих разделах."""
    if not content or "[Секция не сгенерирована" in content:
        return "Секция пропущена или пуста."

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    prompt = f"Выдели 2-3 главных концептуальных тезиса из текста ниже (например: 'Нитрид углерода имеет проблему быстрой рекомбинации', 'Z-схема решает проблему редокс-потенциала'). Сформулируй их как строгий список того, что УЖЕ БЫЛО СКАЗАНО, также упомяни какие таблицы были использованы.\n\nТЕКСТ:\n{content}"

    for attempt in range(3):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            print(f"[⚠️] Ошибка сети при генерации саммари: {e}. Ждем 5 сек (попытка {attempt + 1}/3)...")
            time.sleep(5)

    return "Саммари недоступно."


def get_database_topology(topic: str) -> Dict[str, Any]:
    payload = {"query": topic, "limit": 10000}

    headers = {"X-API-Key": config.CONNECT_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(config.DB_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        chunks = response.json().get("data", [])

        paths = set()
        abstracts = {}

        for c in chunks:
            meta = c.get("metadata", {})
            path = meta.get("original_heading_path")
            abstract = meta.get("abstract")

            if path:
                paths.add(path)
                root_doc = path.split(">")[0].strip()

                if abstract and root_doc not in abstracts:
                    abstracts[root_doc] = abstract.replace("\n", " ").strip()

        return {"paths": list(paths), "abstracts": abstracts}
    except Exception as e:
        print(f"[ERROR] Ошибка получения топологии БД: {e}")
        return {"paths": [], "abstracts": {}}


def build_markdown_tree(paths: list[str], abstracts: dict) -> str:
    tree = {}
    for path in paths:
        parts = [p.strip() for p in path.split(">")]
        curr = tree
        for part in parts:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]

    def render(node, d=0):
        res = ""
        for k, v in node.items():
            res += "  " * d + f"- {k}\n"

            if d == 0 and k in abstracts:
                res += "  " * (d + 1) + f"> АБСТРАКТ: {abstracts[k]}\n"

            res += render(v, d + 1)
        return res

    return render(tree)


def _group_vancouver_numbers(numbers: list[int]) -> str:
    if not numbers:
        return ""
    numbers = sorted(list(set(numbers)))
    ranges, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(
                f"{start}–{prev}"
                if prev - start >= 2
                else (f"{start}, {prev}" if prev != start else str(start))
            )
            start = prev = n
    ranges.append(
        f"{start}–{prev}"
        if prev - start >= 2
        else (f"{start}, {prev}" if prev != start else str(start))
    )
    return ", ".join(ranges)


def planner_node(state: GraphState) -> Dict[str, Any]:
    topic = state["global_topic"]
    topology = get_database_topology(topic)

    tree = build_markdown_tree(topology["paths"], topology["abstracts"])

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(PlanOutput)

    system_prompt = f"""
        Ты — ведущий научный редактор журнала Chemical Reviews. 
        Твоя задача: спроектировать структуру научного обзора (Review Article) на английском языке на тему "{topic}".

        ДЕРЕВО КОНТЕНТА (Доступные источники, их абстракты и секции):
        {tree}

        ОБЯЗАТЕЛЬНАЯ МИНИМАЛЬНАЯ АРХИТЕКТУРА CHEMICAL REVIEWS:
        1. Abstract.
        2. Introduction.
        3. Основные тематические разделы (Глубоко структурированные, с нумерацией 3, 4, 5... и подразделами 3.1, 3.2, 4.1 и т.д.). ЗАПРЕЩЕНО использовать стандартные заголовки вроде "Methods" или "Results". Названия должны отражать суть химических процессов/материалов в контексте темы "{topic}".
        4. Summary and Outlook.
        5. Data Availability Statement (если применимо).

        ПРАВИЛА ПРОЕКТИРОВАНИЯ (АГЕНТСКИЕ ОГРАНИЧЕНИЯ):
        - ZERO HALLUCINATION: Запрещено выдумывать тематические узлы, для которых нет данных. Опирайся ТОЛЬКО на предоставленное Дерево Контента. ИСКЛЮЧЕНИЕ: Для Abstract, Introduction, Summary и Data Availability поле `target_paths` оставляй пустым.
        - MAXIMAL COVERAGE: Интегрируй все релевантные(подходящие для данной темы) пути из Дерева Контента без остатка.
        - DYNAMIC SCALING (КРИТИЧЕСКОЕ ПРАВИЛО МАсштабирования): Оцени объем переданного Дерева Контента. 
          * Если материалов много: спроектируй монументальный обзор (5-8 тематических глав, разбитых на 2-4 подглавы каждая).
          * Если материалов мало: спроектируй компактный, но глубокий обзор (3-4 тематические главы, можно без жесткого дробления на подглавы).
        - ИЕРАРХИЯ JSON: Используй двухуровневую структуру: ГЛАВЫ (chapters) и вложенные ПОДГЛАВЫ (subsections).
        - ГИБКОСТЬ ГЛАВ: 
          Вводные и заключительные главы (Abstract, Introduction, Summary) делай МОНОЛИТНЫМИ (массив subsections пустой, заполняй instructions).
          Тематические главы дроби на подглавы ТОЛЬКО если для этого есть достаточно материала в Дереве Контента. Если тема узкая — делай монолитную тематическую главу.
        - GRANULARITY: Каждый генерируемый текстовый узел (монолитная глава или подглава) должен иметь лимит генерации 800-1000 слов. Прописывай это в instructions.
        - MAPPING: Каждая тематическая секция/подглава ДОЛЖНА содержать список `target_paths` из Дерева Контента.

        АЛГОРИТМ РАБОТЫ (CHAIN OF THOUGHT):
        1. Обязательно создай монолитные главы Abstract и Introduction.
        2. Изучи объем и разнообразие узлов дерева. Выбери масштаб обзора (монументальный или компактный).
        3. Сформируй тематические главы и, если материала достаточно, разбей их на подглавы.
        4. Обязательно добавь монолитные Summary и Data Availability.
        5. Сгенерируй финальный JSON, строго следуя Pydantic-схеме.
        """
    for attempt in range(3):
        try:
            result = structured_llm.invoke(system_prompt)
            if result and result.chapters:
                break
        except Exception as e:
            print(f"[PLANNER] Ошибка генерации плана: {e}. Повтор...")
            time.sleep(5)
    else:
        return {"current_section": None, "pending_sections": []}

    pending = []
    global_section_id = 1

    for chapter in result.chapters:
        if not chapter.subsections:
            pending.append(
                SectionState(
                    section_id=str(global_section_id),
                    title=f"{chapter.chapter_number}. {chapter.title}",
                    instructions=chapter.instructions,
                    target_paths=chapter.target_paths,
                    search_queries=[],
                    raw_chunks=[],
                    memory_bank=[],
                    retriever_rejection="",
                    retriever_retries=0,
                    draft_content="",
                    draft_used_ids=[],
                    citation_errors="",
                    writer_retries=0,
                    content="",
                    used_chunk_ids=[],
                )
            )
            global_section_id += 1
        else:
            for sub in chapter.subsections:
                combined_title = f"{chapter.chapter_number}. {chapter.title} — {sub.section_number}. {sub.title}"
                pending.append(
                    SectionState(
                        section_id=str(global_section_id),
                        title=combined_title,
                        instructions=sub.instructions,
                        target_paths=sub.target_paths,
                        search_queries=[],
                        raw_chunks=[],
                        memory_bank=[],
                        retriever_rejection="",
                        retriever_retries=0,
                        draft_content="",
                        draft_used_ids=[],
                        citation_errors="",
                        writer_retries=0,
                        content="",
                        used_chunk_ids=[],
                    )
                )
                global_section_id += 1

    current = pending.pop(0) if pending else None
    return {"current_section": current, "pending_sections": pending}


def adapter_node(state: GraphState) -> Dict[str, Any]:
    """Генерирует запрос и стягивает сырые данные из БД."""
    current = state["current_section"]
    current["retriever_retries"] += 1

    query = generate_search_query(
        task=current["instructions"],
        previous_queries=current["search_queries"],
        rejection_reason=current["retriever_rejection"],
    )
    current["search_queries"].append(query)

    raw_chunks = db_connector.fetch_chunks(query=query, target_paths=current["target_paths"])
    current["raw_chunks"] = raw_chunks
    return {"current_section": current}


def eval_retriever_node(state: GraphState) -> Dict[str, Any]:
    current = state["current_section"]
    query = current["search_queries"][-1]

    raw_chunks = current["raw_chunks"]

    if not raw_chunks:
        current["retriever_rejection"] = "Отказ: База данных ничего не вернула по этому запросу."
        return {"current_section": current}

    BATCH_SIZE = 10
    batches = [raw_chunks[i : i + BATCH_SIZE] for i in range(0, len(raw_chunks), BATCH_SIZE)]

    results = {}

    for batch in batches:
        batch_result = check_relevance_batch(query, batch)
        results.update(batch_result)

        time.sleep(4.5)

    valid_chunks = []
    rejections = []

    for chunk in raw_chunks:
        chunk_id = chunk["id"]
        eval_res = results.get(
            chunk_id, {"is_relevant": False, "reason": "LLM пропустила этот чанк при валидации"}
        )

        if eval_res["is_relevant"]:
            valid_chunks.append(chunk)
        else:
            rejections.append(eval_res["reason"])

    if valid_chunks:
        current["memory_bank"] = valid_chunks
        current["retriever_rejection"] = ""
    else:
        unique_reasons = list(set([r for r in rejections if r]))
        current["retriever_rejection"] = f"Отказ: {', '.join(unique_reasons[:3])}"

    return {"current_section": current}


def writer_node(state: GraphState) -> Dict[str, Any]:
    current = state["current_section"]
    if current["writer_retries"] > 0:
        print("Ошибка цитирования. Пауза 20 сек для сброса лимитов API...")
        time.sleep(20)
    current["writer_retries"] += 1

    prev_summary = state.get("previous_sections_summary", "")

    draft = generate_section_draft(
        title=current["title"],
        instructions=current["instructions"],
        memory_bank=current["memory_bank"],
        previous_summary=prev_summary,
        citation_errors=current["citation_errors"],
    )

    current["draft_content"] = draft.content
    current["draft_used_ids"] = draft.used_chunk_ids
    return {"current_section": current}


import re
from typing import Dict, Any


def eval_citation_node(state: GraphState) -> Dict[str, Any]:
    """Детерминированная AST-проверка галлюцинаций цитат, форматов и зацикливаний."""
    current = state["current_section"]
    draft_text = current["draft_content"]

    if "Обрыв генерации по лимиту токенов" in draft_text:
        current["citation_errors"] = (
            "КРИТИЧЕСКАЯ ОШИБКА: Твой предыдущий ответ был обрезан из-за бесконечного зацикливания. "
            "Сгенерируй текст строго по делу, не используй спам из макросов LaTeX и следи за структурой."
        )
        return {"current_section": current}

    draft_text = draft_text.replace('\x08', '\\b').replace('\x09', '\\t')

    if re.search(r'(.{15,})\1{10,}', draft_text):
        current["citation_errors"] = (
            "КРИТИЧЕСКАЯ ОШИБКА: Обнаружено жесткое зацикливание генерации (бесконечный повтор токенов). "
            "Сгенерируй текст заново, измени структуру предложения и не используй проблемный макрос."
        )
        return {"current_section": current}

    illegal_tags = re.findall(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", draft_text)
    if illegal_tags:
        current["citation_errors"] = (
            f"КРИТИЧЕСКАЯ ОШИБКА: Использован запрещенный формат цитирования {illegal_tags[0]}. "
            f"ТЫ ОБЯЗАН ИСПОЛЬЗОВАТЬ ТОЛЬКО ФОРМАТ [ID: УНИКАЛЬНЫЙ_НОМЕР]."
        )
        return {"current_section": current}

    raw_tags = re.findall(r"\[ID:\s*([^\]]+)\]", draft_text)

    found_ids = []
    for tag in raw_tags:
        uuids = [
            u.strip()
            for u in re.findall(r"([a-zA-Z0-9\-]+)", tag)
            if u.strip().upper() != 'ID'
        ]
        found_ids.extend(uuids)

    found_ids = list(set(found_ids))
    valid_bank_ids = [str(c["id"]) for c in current["memory_bank"]]
    hallucinated_ids = [fid for fid in found_ids if fid not in valid_bank_ids]

    if hallucinated_ids:
        current["citation_errors"] = (
            f"В тексте использованы несуществующие ID: {hallucinated_ids}. Используй ТОЛЬКО ID из ПАМЯТИ."
        )
    else:
        current["citation_errors"] = ""
        current["content"] = draft_text
        current["used_chunk_ids"] = found_ids

    return {"current_section": current}


def advance_section_node(state: GraphState) -> Dict[str, Any]:
    current = state["current_section"]

    if not current["content"]:
        current["content"] = (
            current["draft_content"] or "[Секция не сгенерирована из-за нехватки данных]"
        )

    completed = state.get("completed_sections", [])[:] + [current]
    pending = state["pending_sections"][:]

    next_current = pending.pop(0) if pending else None

    current_summary = generate_short_summary(current["content"])
    new_summary = (
        state.get("previous_sections_summary", "")
        + f"\nРаздел '{current['title']}': {current_summary}"
    )

    return {
        "current_section": next_current,
        "completed_sections": completed,
        "pending_sections": pending,
        "previous_sections_summary": new_summary,
    }


def compiler_node(state: GraphState) -> Dict[str, Any]:
    text_blocks = []
    last_chapter = None

    for sec in state["completed_sections"]:
        raw_title = sec.get("title", "Untitled Section")
        content = sec.get("content", "")

        if " — " in raw_title:
            chapter_title, sub_title = raw_title.split(" — ", 1)

            if chapter_title != last_chapter:
                text_blocks.append(f"## {chapter_title}")
                last_chapter = chapter_title

            text_blocks.append(f"### {sub_title}\n\n{content}")
        else:
            text_blocks.append(f"## {raw_title}\n\n{content}")
            last_chapter = raw_title

    full_text = "\n\n".join(text_blocks)

    meta_map = {}
    for sec in state["completed_sections"]:
        for chunk in sec.get("memory_bank", []):
            meta_map[str(chunk["id"])] = chunk.get("metadata", {})

    paper_ref_map = {}
    paper_meta_map = {}
    ref_counter = 1

    def process_cluster(match):
        nonlocal ref_counter
        cluster_text = match.group(0)

        raw_ids = re.findall(r"([a-zA-Z0-9\-]+)", cluster_text)
        ids = [i.strip() for i in raw_ids if i.strip().upper() != 'ID']

        if not ids:
            return cluster_text

        nums = []
        for cid in ids:
            if cid in meta_map:
                meta = meta_map[cid]

                # Идентифицируем уникальную статью по DOI. Если его нет — по названию.
                doi = meta.get("doi")
                if doi and str(doi).strip().lower() != "doi отсутствует":
                    paper_key = str(doi).strip().lower()
                else:
                    paper_key = str(meta.get("title", f"unknown_{cid}")).strip().lower()

                if paper_key not in paper_ref_map:
                    paper_ref_map[paper_key] = ref_counter
                    paper_meta_map[paper_key] = meta
                    ref_counter += 1

                nums.append(paper_ref_map[paper_key])

        if not nums:
            return ""

        return f"<sup>{_group_vancouver_numbers(nums)}</sup>"

    full_text = full_text.replace('\\\\', '\\')

    final_doc = re.sub(
        r"(?:\[ID:\s*[a-zA-Z0-9\-]+\]\s*)+",
        process_cluster,
        full_text,
        flags=re.IGNORECASE,
    )

    final_doc += "\n\n## References\n"
    for paper_key, num in sorted(paper_ref_map.items(), key=lambda i: i[1]):
        meta = paper_meta_map[paper_key]

        authors = meta.get("authors", [])
        author_str = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "Unknown Author")
        title = meta.get("title", "Unknown Title")
        year = meta.get("year", "Unknown Year")
        doi = meta.get("doi", "DOI отсутствует")

        final_doc += f"{num}. {author_str}, \"{title}\", {year}. DOI: {doi}\n"

    return {"final_document": final_doc}