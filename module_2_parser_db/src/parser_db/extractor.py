"""Модуль для извлечения данных MinerU и преобразования их в схемы."""

import re
from typing import Any

from markdownify import markdownify as md

from parser_db.equations import fix_latex_brackets, validate_latex
from parser_db.schemas import Paragraph, ParsedDocument, Section, VisualMeta


def optimize_table_markup(html_markup: str) -> str:
    """Адаптивно сжимает таблицу: простую в Markdown, сложную — в HTML.

    Args:
        html_markup (str): Сырой HTML-код таблицы.

    Returns:
        str: Оптимизированный Markdown или минифицированный HTML.
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
    clean_html = re.sub(r">\s+<", "><", clean_html)

    return clean_html.strip()


def normalize_section_name(heading: str) -> str:
    """Приводит сырой заголовок к Pydantic Enum для API-шлюза.

    Args:
        heading (str): Исходный текст заголовка.

    Returns:
        str: Стандартизированное название секции или исходный заголовок.
    """
    h_lower = heading.lower()
    if "abstract" in h_lower:
        return "Abstract"
    if "intro" in h_lower:
        return "Introduction"
    if any(x in h_lower for x in ["method", "experiment", "procedure", "material"]):
        return "Methodology"
    if "result" in h_lower:
        return "Results"
    if "discuss" in h_lower:
        return "Discussion"
    if "conclus" in h_lower or "summary" in h_lower:
        return "Conclusion"
    return heading


def extract_exact_visual_id(caption: str, default_id: str) -> str:
    """Вытягивает точный ID (например, 'Fig. 1') из подписи для препроцессора.

    Args:
        caption (str): Подпись к графику или таблице.
        default_id (str): ID по умолчанию, если паттерн не найден.

    Returns:
        str: Точный идентификатор объекта.
    """
    if not caption:
        return default_id
    match = re.match(
        r"^((?:Fig\.|Figure|Table|Scheme)\s*\d+[a-zA-Z]?)",
        caption,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else default_id


def is_table_broken(html_markup: str) -> bool:
    """Определяет, сломана ли структура HTML-таблицы парсером.

    Args:
        html_markup (str): Сырой HTML-код таблицы.

    Returns:
        bool: True, если найдены аномалии, иначе False.
    """
    if not html_markup or len(html_markup) < 30:
        return True

    clean_text = re.sub(r"<[^>]+>", " ", html_markup)
    words = clean_text.split()

    if any(len(w) > 35 and "http" not in w for w in words):
        return True

    empty_cells = html_markup.count("<td></td>") + html_markup.count("<td> </td>")
    total_cells = html_markup.count("<td")

    if total_cells > 0 and (empty_cells / total_cells) > 0.4:
        return True

    if html_markup.count("<tr") != html_markup.count("</tr"):
        return True

    return False


def clean_text_lite(text: str) -> str:
    """Очищает текст от базовых артефактов MinerU.

    Args:
        text (str): Исходный сырой текст.

    Returns:
        str: Нормализованный текст без висячих дефисов и лишних тегов.
    """
    text = text.replace("\u0001", "°").replace("\u0003", "-")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_parsed_document(
    mineru_data: list[dict[str, Any]], doi: str, title: str
) -> ParsedDocument:
    """Собирает объект ParsedDocument из сырого JSON MinerU.

    Группирует абзацы по разделам, конвертирует HTML-таблицы в Markdown,
    лечит битый LaTeX и извлекает метаданные графики.

    Args:
        mineru_data (list[dict[str, Any]]): Список блоков из JSON.
        doi (str): Уникальный идентификатор статьи.
        title (str): Название статьи.

    Returns:
        ParsedDocument: Валидированный Pydantic-объект статьи.
    """
    sections: list[Section] = []
    visuals: list[VisualMeta] = []

    current_heading = "Metadata / Abstract"
    current_paragraphs: list[Paragraph] = []
    current_level = 1

    for block in mineru_data:
        block_type = block.get("type", "")
        raw_content = block.get("text", "").strip()

        content = clean_text_lite(raw_content) if block_type == "text" else raw_content

        if block_type == "text" and "text_level" in block:
            if content == current_heading:
                continue

            if current_paragraphs:
                normalized_heading = normalize_section_name(current_heading)
                sections.append(
                    Section(
                        heading=normalized_heading,
                        level=current_level,
                        paragraphs=current_paragraphs,
                    )
                )
            current_heading = content
            current_level = block.get("text_level", 1)
            current_paragraphs = []
            continue

        if block_type in ["equation", "inline_equation"]:
            img_path = block.get("img_path", "")

            if validate_latex(content):
                current_paragraphs.append(Paragraph(type="equation", content=content))
            else:
                fixed_content = fix_latex_brackets(content)
                if validate_latex(fixed_content):
                    current_paragraphs.append(Paragraph(type="equation", content=fixed_content))
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
        normalized_heading = normalize_section_name(current_heading)
        sections.append(
            Section(
                heading=normalized_heading,
                level=current_level,
                paragraphs=current_paragraphs,
            )
        )

    return ParsedDocument(doi=doi, title=title, sections=sections, visuals=visuals)
