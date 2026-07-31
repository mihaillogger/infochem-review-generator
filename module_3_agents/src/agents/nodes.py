import re
import requests
import time
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor

import config
from state import GraphState, PlanOutput, SectionState
from adapter import generate_search_query
import db_connector
from writer import generate_section_draft
from validator import check_relevance_batch


def generate_short_summary(content: str) -> str:
    """Делает короткую выжимку текста для передачи в следующие узлы."""
    if not content or "[Секция не сгенерирована" in content:
        return "Секция пропущена или пуста."

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    prompt = f"Сделай ультра-короткую выжимку (3-4 предложения) главных научных фактов из этого текста. Без вводных слов.\n\nТЕКСТ:\n{content}"
    return llm.invoke(prompt).content


def get_database_topology(topic: str) -> list[str]:
    payload = {"query": topic, "limit": 10000}
    try:
        response = requests.post(config.DB_API_URL, json=payload, timeout=30)
        chunks = response.json().get("data", [])
        return list(
            {c.get("metadata", {}).get("section_path") for c in chunks if c.get("metadata", {}).get("section_path")})
    except:
        return []


def build_markdown_tree(paths: list[str]) -> str:
    tree = {}
    for path in paths:
        parts = [p.strip() for p in path.split('>')]
        curr = tree
        for part in parts:
            if part not in curr: curr[part] = {}
            curr = curr[part]

    def render(node, d=0):
        res = ""
        for k, v in node.items():
            res += "  " * d + f"- {k}\n" + render(v, d + 1)
        return res

    return render(tree)


def _group_vancouver_numbers(numbers: list[int]) -> str:
    if not numbers: return ""
    numbers = sorted(list(set(numbers)))
    ranges, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(
                f"{start}–{prev}" if prev - start >= 2 else (f"{start}, {prev}" if prev != start else str(start)))
            start = prev = n
    ranges.append(f"{start}–{prev}" if prev - start >= 2 else (f"{start}, {prev}" if prev != start else str(start)))
    return ", ".join(ranges)


# УЗЛЫ LANGGRAPH

def planner_node(state: GraphState) -> Dict[str, Any]:
    topic = state["global_topic"]
    raw_paths = get_database_topology(topic)
    tree = build_markdown_tree(raw_paths)

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    structured_llm = llm.with_structured_output(PlanOutput)

    system_prompt = f"""
    Ты — ведущий научный редактор журнала Chemical Reviews. 
    Твоя задача: спроектировать структуру монументального научного обзора (Review Article) на тему "{topic}" на основе предоставленного Дерева Контента.
    
    ОБЯЗАТЕЛЬНАЯ АРХИТЕКТУРА CHEMICAL REVIEWS:
    1. Abstract.
    2. Introduction.
    3. Основные тематические разделы (Глубоко структурированные, нумерация 2, 3, 4 с подразделами 2.1, 2.2, 2.2.1 и т.д.). ЗАПРЕЩЕНО использовать стандартные заголовки вроде "Methods" или "Results". Названия должны отражать суть химических процессов/материалов в контексте темы "{topic}".
    4. Summary and Outlook.
    5. Data Availability Statement (если применимо).
    
    ПРАВИЛА ПРОЕКТИРОВАНИЯ (АГЕНТСКИЕ ОГРАНИЧЕНИЯ):
    - ZERO HALLUCINATION: Запрещено выдумывать разделы, для которых нет данных в Дереве Контента.
    - MAXIMAL COVERAGE: Ты обязан интегрировать МАКСИМАЛЬНОЕ количество релевантных узлов из Дерева Контента. Не оставляй целые статьи или крупные ветки без внимания, если они косвенно или прямо связаны с темой "{topic}".
    - CROSS-REFERENCING: Каждая крупная секция плана в идеале должна объединять `target_paths` из РАЗНЫХ статей (источников), если они пересекаются по смыслу (например, механизмы инкапсуляции и методы доставки). Избегай создания секций, опирающихся только на одну статью.
    - GRANULARITY: Разбивай обширные темы на подразделы (1000-1500 слов на узел), чтобы writer_node мог глубоко раскрыть тему.
    - MAPPING: Каждая секция плана ДОЛЖНА содержать список `target_paths` — точных путей из Дерева Контента.
    - COPY-PASTE ONLY: Ты ДОЛЖЕН копировать target_paths символ в символ из предоставленного Дерева Контента. Категорически запрещено перефразировать, изменять регистр или выдумывать пути, которых нет во входных данных.
    
    АЛГОРИТМ РАБОТЫ (CHAIN OF THOUGHT):
    1. Проведи аудит Дерева Контента: перечисли для себя все предоставленные источники (статьи) и выдели их главные темы.
    2. Сформируй междисциплинарные кластеры, которые объединяют данные из разных источников (например, методы синтеза + их применение).
    3. Проверь "Слепые зоны": убедись, что ни один крупный раздел Дерева (например, "Self-Healing" или "LbL") не был забыт, если он связан с темой.
    4. Выстрой логическую последовательность разделов от базовых концепций к сложным применениям.
    5. Сгенерируй финальный JSON, строго следуя Pydantic-схеме.
    """
    result = structured_llm.invoke(system_prompt)

    pending = []
    for i, sec in enumerate(result.sections):
        pending.append(SectionState(
            section_id=str(i + 1), title=sec.title, instructions=sec.instructions, target_paths=sec.target_paths,
            search_queries=[], raw_chunks=[], memory_bank=[], retriever_rejection="", retriever_retries=0,
            draft_content="", draft_used_ids=[], citation_errors="", writer_retries=0, content="", used_chunk_ids=[]
        ))

    current = pending.pop(0) if pending else None
    return {"current_section": current, "pending_sections": pending}


def adapter_node(state: GraphState) -> Dict[str, Any]:
    """Генерирует запрос и стягивает сырые данные из БД."""
    current = state["current_section"]
    current["retriever_retries"] += 1

    query = generate_search_query(
        task=current["instructions"],
        previous_queries=current["search_queries"],
        rejection_reason=current["retriever_rejection"]
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

    # Настройки батчинга
    BATCH_SIZE = 10
    batches = [raw_chunks[i:i + BATCH_SIZE] for i in range(0, len(raw_chunks), BATCH_SIZE)]

    results = {}

    for batch in batches:
        batch_result = check_relevance_batch(query, batch)
        results.update(batch_result)

        time.sleep(4.5)

    valid_chunks = []
    rejections = []

    for chunk in raw_chunks:
        chunk_id = chunk["id"]
        eval_res = results.get(chunk_id,
                               {"is_relevant": False, "reason": "LLM пропустила этот чанк при валидации"})

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
    current["writer_retries"] += 1

    prev_summary = state.get("previous_sections_summary", "")

    draft = generate_section_draft(
        title=current["title"],
        instructions=current["instructions"],
        memory_bank=current["memory_bank"],
        previous_summary=prev_summary,
        citation_errors=current["citation_errors"]
    )

    current["draft_content"] = draft.content
    current["draft_used_ids"] = draft.used_chunk_ids
    return {"current_section": current}


def eval_citation_node(state: GraphState) -> Dict[str, Any]:
    """Детерминированная AST-проверка галлюцинаций цитат."""
    current = state["current_section"]
    draft_text = current["draft_content"]

    # Парсим теги [ID: xxx] из текста
    found_ids = re.findall(r'\[ID:\s*([^\]]+)\]', draft_text)
    found_ids = [fid.strip() for fid in found_ids]

    valid_bank_ids = [c["id"] for c in current["memory_bank"]]
    hallucinated_ids = [fid for fid in found_ids if fid not in valid_bank_ids]

    if hallucinated_ids:
        current[
            "citation_errors"] = f"В тексте использованы несуществующие ID: {hallucinated_ids}. Используй ТОЛЬКО ID из ПАМЯТИ."
    else:
        current["citation_errors"] = ""
        current["content"] = draft_text
        current["used_chunk_ids"] = found_ids

    return {"current_section": current}


def advance_section_node(state: GraphState) -> Dict[str, Any]:
    current = state["current_section"]

    if not current["content"]:
        current["content"] = current["draft_content"] or "[Секция не сгенерирована из-за нехватки данных]"

    completed = state.get("completed_sections", [])[:] + [current]
    pending = state["pending_sections"][:]

    next_current = pending.pop(0) if pending else None

    current_summary = generate_short_summary(current["content"])
    new_summary = state.get("previous_sections_summary", "") + f"\nРаздел '{current['title']}': {current_summary}"

    return {
        "current_section": next_current,
        "completed_sections": completed,
        "pending_sections": pending,
        "previous_sections_summary": new_summary
    }


def compiler_node(state: GraphState) -> Dict[str, Any]:
    full_text = "\n\n".join([sec["content"] for sec in state["completed_sections"]])
    ref_map = {}
    ref_counter = 1

    def process_cluster(match):
        nonlocal ref_counter
        cluster_text = match.group(0)
        # Внутри кластера парсинг ID отработает четко, он игнорирует запятые
        ids = re.findall(r'\[ID:\s*([^\]]+)\]', cluster_text)
        nums = []
        for cid in ids:
            if cid.strip() not in ref_map:
                ref_map[cid.strip()] = ref_counter
                ref_counter += 1
            nums.append(ref_map[cid.strip()])
        return f"<sup>{_group_vancouver_numbers(nums)}</sup>"

    final_doc = re.sub(r'(?:\[ID:\s*[^\]]+\](?:[,\s]*))+', process_cluster, full_text)

    final_doc += "\n\n## References\n"
    for cid, num in sorted(ref_map.items(), key=lambda i: i[1]):
        final_doc += f"{num}. [Database Node: {cid}]\n"

    return {"final_document": final_doc}