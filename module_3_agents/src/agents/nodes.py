import re
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import GraphState, PlanOutput, SectionState, WriterOutput
import requests
from typing import List


def get_database_topology(topic: str) -> List[str]:
    """
    Делает гибридный поиск по базе Матвея и собирает уникальные пути (section_path).
    """
    url = "LINK/api/v1/search"

    payload = {
        "query": topic,
        "limit": 10000
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        chunks = response_data.get("data", [])

        unique_paths = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            section_path = metadata.get("section_path")

            if section_path and isinstance(section_path, str):
                unique_paths.add(section_path)

        return list(unique_paths)

    except Exception as e:
        print(f"ВНИМАНИЕ: Ошибка при запросе к API Матвея: {e}")
        return []

def build_markdown_tree(paths: List[str]) -> str:
    """Превращает список путей section_path в Markdown-дерево."""
    tree = {}
    for path in paths:
        parts = [p.strip() for p in path.split('>')]
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

    def _render_tree(node, depth=0):
        result = ""
        for key, child in node.items():
            result += "  " * depth + f"- {key}\n"
            result += _render_tree(child, depth + 1)
        return result

    return _render_tree(tree)


def _group_vancouver_numbers(numbers: List[int]) -> str:
    """Схлопывает массив номеров в диапазоны ([14,15,16,19] -> '14–16, 19')."""
    if not numbers:
        return ""
    numbers = sorted(list(set(numbers)))
    ranges = []
    start = prev = numbers[0]

    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
        else:
            if prev - start >= 2:
                ranges.append(f"{start}–{prev}")
            elif prev != start:
                ranges.extend([str(start), str(prev)])
            else:
                ranges.append(str(start))
            start = prev = n

    if prev - start >= 2:
        ranges.append(f"{start}–{prev}")
    elif prev != start:
        ranges.extend([str(start), str(prev)])
    else:
        ranges.append(str(start))

    return ", ".join(ranges)


# УЗЛЫ LANGGRAPH

def planner_node(state: GraphState) -> Dict[str, Any]:
    """Узел 1: Строит план на основе метаданных базы."""

    topic = state["global_topic"]

    raw_paths_from_db = get_database_topology(topic)

    if not raw_paths_from_db:
        raise ValueError("База вернула пустой список путей. Проверьте API или топик.")

    content_tree = build_markdown_tree(raw_paths_from_db)

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
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

    user_prompt = f"Тема обзора: {topic}\n\nДЕРЕВО ДОСТУПНОГО КОНТЕНТА:\n{content_tree}"

    result = structured_llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    pending = []
    for i, sec in enumerate(result.sections):
        pending.append(SectionState(
            section_id=str(i + 1),
            title=sec.title,
            instructions=sec.instructions,
            target_paths=sec.target_paths,  # ПРОКИДЫВАЕМ ПУТИ ДЛЯ МАТВЕЯ
            content="",
            used_chunk_ids=[]
        ))

    current = pending.pop(0) if pending else None

    return {
        "current_section": current,
        "pending_sections": pending,
        "completed_sections": []
    }


def external_writer_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 2: Adapter + Writer.
    Граф отправляет запрос и валидирует ответ по контракту WriterOutput.
    """
    current = state["current_section"]

    # === ИМИТАЦИЯ ВЫЗОВА API ===

    raw_response = {
        "section_title": current["title"],
        "content": f"## {current['title']}\nТекст по путям {current['target_paths']}. Скорость реакции растет [ID: 8f2][ID: 3b1].",
        "used_chunk_ids": ["8f2", "3b1"]
    }

    validated_output = WriterOutput.model_validate(raw_response)

    current["content"] = validated_output.content
    current["used_chunk_ids"] = validated_output.used_chunk_ids

    completed = state.get("completed_sections", []) + [current]
    next_pending = state["pending_sections"]
    next_current = next_pending.pop(0) if next_pending else None

    return {
        "current_section": next_current,
        "pending_sections": next_pending,
        "completed_sections": completed
    }


def compiler_node(state: GraphState) -> Dict[str, Any]:
    """Узел 3: Сборка и форматирование ссылок (Vancouver)."""
    full_text = "\n\n".join([sec["content"] for sec in state["completed_sections"]])
    ref_map = {}
    ref_counter = 1

    cluster_pattern = re.compile(r'(?:\[ID:\s*[^\]]+\])+')
    single_id_pattern = re.compile(r'\[ID:\s*([^\]]+)\]')

    def process_cluster(match):
        nonlocal ref_counter
        cluster_text = match.group(0)
        ids = single_id_pattern.findall(cluster_text)

        numbers = []
        for chunk_id in ids:
            chunk_id = chunk_id.strip()
            if chunk_id not in ref_map:
                ref_map[chunk_id] = ref_counter
                ref_counter += 1
            numbers.append(ref_map[chunk_id])

        grouped_string = _group_vancouver_numbers(numbers)
        return f"<sup>{grouped_string}</sup>"

    final_document = cluster_pattern.sub(process_cluster, full_text)

    final_document += "\n\n## References\n"
    sorted_refs = sorted(ref_map.items(), key=lambda item: item[1])
    for chunk_id, num in sorted_refs:
        final_document += f"{num}. Source document DOI/Metadata for {chunk_id}\n"

    return {"final_document": final_document}