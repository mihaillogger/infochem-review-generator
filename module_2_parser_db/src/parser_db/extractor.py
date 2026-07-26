"""Модуль для извлечения данных MinerU и преобразования их в схемы."""

import re
from enum import StrEnum
from typing import Any

from markdownify import markdownify as md

from parser_db.equations import validate_latex
from parser_db.schemas import Paragraph, ParsedDocument, Section, VisualMeta


class SectionType(StrEnum):
    """Перечисление стандартизированных разделов для химических review-статей."""

    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    CONCEPTS_AND_MECHANISMS = "Concepts & Mechanisms"
    MATERIALS_AND_SYNTHESIS = "Materials & Synthesis"
    APPLICATIONS = "Applications & Devices"
    PERSPECTIVES_AND_CONCLUSIONS = "Perspectives & Conclusions"
    UNKNOWN = "Unknown"


def optimize_table_markup(html_markup: str) -> str:
    """Адаптивно сжимает таблицу: простую в Markdown, сложную — в HTML.

    Args:
        html_markup: Сырой HTML-код таблицы.

    Returns:
        Оптимизированный Markdown или минифицированный HTML.
    """
    is_complex = "colspan" in html_markup.lower() or "rowspan" in html_markup.lower()

    if not is_complex:
        return md(html_markup, strip=["a", "img"], heading_style="ATX").strip()

    clean_html = re.sub(
        r"</?(thead|tbody|tfoot|div|span)[^>]*>",
        "",
        html_markup,
        flags=re.IGNORECASE,
    )
    clean_html = re.sub(
        r'\s+(?!colspan|rowspan)[a-z\-]+="[^"]*"',
        "",
        clean_html,
        flags=re.IGNORECASE,
    )
    return re.sub(r">\s+<", "><", clean_html).strip()


def normalize_section_name(heading: str) -> SectionType:
    """Приводит сырой заголовок обзорной статьи к строгому Enum.

    Args:
        heading: Исходный текст заголовка.

    Returns:
        Стандартизированное название секции макро-уровня.
    """
    h_lower = heading.lower()

    if "abstract" in h_lower:
        return SectionType.ABSTRACT

    if "intro" in h_lower or "background" in h_lower:
        return SectionType.INTRODUCTION

    concepts_kw = ["concept", "mechanism", "principle", "theory", "interaction", "behavior"]
    if any(x in h_lower for x in concepts_kw):
        return SectionType.CONCEPTS_AND_MECHANISMS

    materials_kw = [
        "material",
        "synthesis",
        "fabrication",
        "preparation",
        "structure",
        "composite",
        "hybrid",
        "nanostructuring",
        "route",
    ]
    if any(x in h_lower for x in materials_kw):
        return SectionType.MATERIALS_AND_SYNTHESIS

    apps_kw = ["application", "device", "delivery", "therapy", "sensor", "patterning", "coating"]
    if any(x in h_lower for x in apps_kw):
        return SectionType.APPLICATIONS

    conclusions_kw = ["conclus", "summary", "prospect", "perspective", "future", "outlook"]
    if any(x in h_lower for x in conclusions_kw):
        return SectionType.PERSPECTIVES_AND_CONCLUSIONS

    return SectionType.UNKNOWN


def extract_exact_visual_id(caption: str, default_id: str) -> str:
    """Вытягивает точный ID из подписи для препроцессора.

    Args:
        caption: Подпись к графику или таблице.
        default_id: ID по умолчанию, если паттерн не найден.

    Returns:
        Точный идентификатор объекта.
    """
    if not caption:
        return default_id
    match = re.match(
        r"^((?:Fig\.|Figure|Table|Scheme)\s*\d+[a-zA-Z]?)",
        caption,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else default_id


def is_smiles(text: str) -> bool:
    """Проверяет, похожа ли строка на химическую нотацию SMILES.

    Args:
        text: Строка для проверки.

    Returns:
        True, если строка содержит только допустимые для SMILES символы.
    """
    smiles_pattern = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\=#/]+$")
    return bool(smiles_pattern.match(text))


def is_table_broken(html_markup: str) -> bool:
    """Определяет, сломана ли структура HTML-таблицы парсером.

    Отдельно проверяет длинные слова, игнорируя ссылки и SMILES.

    Args:
        html_markup: Сырой HTML-код таблицы.

    Returns:
        True, если найдены аномалии, иначе False.
    """
    if not html_markup or len(html_markup) < 30:
        return True

    clean_text = re.sub(r"<[^>]+>", " ", html_markup)
    words = clean_text.split()

    for w in words:
        if len(w) > 35 and "http" not in w and not is_smiles(w):
            return True

    empty_cells = html_markup.count("<td></td>") + html_markup.count("<td> </td>")
    total_cells = html_markup.count("<td")

    if total_cells > 0 and (empty_cells / total_cells) > 0.4:
        return True

    return html_markup.count("<tr") != html_markup.count("</tr")


def clean_text_lite(text: str) -> str:
    """Очищает текст от базовых артефактов MinerU.

    Args:
        text: Исходный сырой текст.

    Returns:
        Нормализованный текст без висячих дефисов и лишних тегов.
    """
    text = text.replace("\u0001", "°").replace("\u0003", "-")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def build_parsed_document(
    mineru_data: list[dict[str, Any]], doi: str, title: str
) -> ParsedDocument:
    """Собирает объект ParsedDocument из сырого JSON MinerU.

    Группирует абзацы по разделам, конвертирует HTML-таблицы в Markdown
    и извлекает метаданные графики. Невалидный LaTeX помечается как битый.

    Args:
        mineru_data: Список блоков из JSON.
        doi: Уникальный идентификатор статьи.
        title: Название статьи.

    Returns:
        Валидированный Pydantic-объект статьи.
    """
    sections: list[Section] = []
    visuals: list[VisualMeta] = []

    current_heading_str = "Metadata / Abstract"
    current_heading_enum = SectionType.ABSTRACT

    current_paragraphs: list[Paragraph] = []
    current_level = 1

    for block in mineru_data:
        block_type = block.get("type", "")
        raw_content = block.get("text", "").strip()

        content = clean_text_lite(raw_content) if block_type == "text" else raw_content

        if block_type == "text" and "text_level" in block:
            if content == current_heading_str:
                continue

            if current_paragraphs:
                sections.append(
                    Section(
                        heading=current_heading_enum.value,
                        level=current_level,
                        paragraphs=current_paragraphs,
                    )
                )
            current_heading_str = content
            current_heading_enum = normalize_section_name(content)
            current_level = block.get("text_level", 1)
            current_paragraphs = []
            continue

        if block_type in ["equation", "inline_equation"]:
            img_path = block.get("img_path", "")

            if validate_latex(content):
                current_paragraphs.append(Paragraph(type="equation", content=content))
            else:
                current_paragraphs.append(
                    Paragraph(
                        type="equation",
                        content=content,
                        is_broken=True,
                        image_fallback_path=img_path,
                    )
                )
            continue

        if block_type in ["image", "table", "chart"]:
            raw_id = block.get("id") or f"Vis_{len(visuals)}"
            raw_caption = (
                block.get("image_caption")
                or block.get("chart_caption")
                or block.get("table_caption")
                or []
            )
            caption = " ".join(raw_caption).strip() if raw_caption else ""

            exact_id = extract_exact_visual_id(caption, raw_id)
            img_path = block.get("img_path", "")

            visuals.append(VisualMeta(id=exact_id, path=img_path, caption=caption))

            if block_type == "table":
                table_html = block.get("table_body", "")
                if not table_html:
                    continue

                if is_table_broken(table_html):
                    current_paragraphs.append(
                        Paragraph(
                            type="table",
                            content=table_html,
                            is_broken=True,
                            image_fallback_path=img_path,
                        )
                    )
                else:
                    table_md = optimize_table_markup(table_html)
                    table_content = f"Caption: {caption}\n\n{table_md}"
                    current_paragraphs.append(Paragraph(type="table", content=table_content))
            else:
                current_paragraphs.append(
                    Paragraph(type="text", content=f"[{exact_id}: {caption}]")
                )
            continue

        if block_type == "text" and content:
            current_paragraphs.append(Paragraph(type="text", content=content))

    if current_paragraphs:
        sections.append(
            Section(
                heading=current_heading_enum.value,
                level=current_level,
                paragraphs=current_paragraphs,
            )
        )

    return ParsedDocument(doi=doi, title=title, sections=sections, visuals=visuals)
