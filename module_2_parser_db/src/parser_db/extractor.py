"""
Модуль для извлечения данных из сырого JSON (MinerU) и преобразования
их в строгие Pydantic-схемы (ParsedDocument).
"""

from typing import Any

from parser_db.equations import fix_latex_brackets, validate_latex
from parser_db.schemas import Paragraph, ParsedDocument, Section, VisualMeta


def flatten_markdown_table(md_table: str) -> str:
    """
    Преобразует Markdown-таблицу в плоский текст и проверяет её целостность.
    Если таблица битая (едут колонки), ставит флаг для агентов.
    """
    lines = [line.strip() for line in md_table.strip().split("\n") if line.strip()]

    if len(lines) < 3:
        return f"[REQUIRES_FACTCHECK] Неполная таблица: {md_table}"

    # 1. Достаем заголовки колонок
    headers = [col.strip() for col in lines[0].split("|") if col.strip()]
    expected_cols = len(headers)

    flat_rows = []
    is_broken = False

    # 2. Проходим по строкам с данными
    for row_idx, line in enumerate(lines[2:], start=1):
        cells = [col.strip() for col in line.split("|") if col.strip()]

        # Проверяем целостность: совпадает ли число ячеек с числом заголовков
        if len(cells) != expected_cols:
            is_broken = True

        row_data = []
        for i in range(min(len(headers), len(cells))):
            row_data.append(f"{headers[i]}: {cells[i]}")

        flat_rows.append(f"Строка {row_idx} [{', '.join(row_data)}]")

    result_text = " Данные таблицы: " + " | ".join(flat_rows)

    # Если ряды поехали, маркируем всю таблицу
    if is_broken:
        return f"[REQUIRES_FACTCHECK] Нарушена структура колонок. {result_text}"

    return result_text


def build_parsed_document(
    mineru_data: list[dict[str, Any]], doi: str, title: str
) -> ParsedDocument:
    """
    Собирает объект ParsedDocument из сырого JSON, полученного от MinerU.

    Использует конечный автомат для группировки абзацев по разделам,
    извлекает метаданные графики и лечит битый LaTeX.

    Args:
        mineru_data (List[Dict[str, Any]]): Список блоков из _content_list.json.
        doi (str): Уникальный идентификатор статьи.
        title (str): Название статьи.

    Returns:
        ParsedDocument: Валидированный Pydantic-объект всей статьи.
    """
    sections: list[Section] = []
    visuals: list[VisualMeta] = []

    current_heading = "Metadata / Abstract"
    current_paragraphs: list[Paragraph] = []

    for block in mineru_data:
        block_type = block.get("type")
        content = block.get("text", "").strip()

        # 1. Обработка заголовков (Создание новых секций)
        # У MinerU заголовки обычно идут с типом 'text' и дополнительным полем разметки
        if block_type == "text" and block.get("layout_type") == "heading":
            if current_paragraphs:
                sections.append(
                    Section(heading=current_heading, level=1, paragraphs=current_paragraphs)
                )
            current_heading = content
            current_paragraphs = []
            continue

        # 2. Обработка математики (Валидация и автоисправление)
        if block_type in ["equation", "inline_equation"]:
            if validate_latex(content):
                current_paragraphs.append(Paragraph(type="equation", content=content))
            else:
                # Пытаемся вылечить баланс скобок
                fixed_content = fix_latex_brackets(content)
                if validate_latex(fixed_content):
                    current_paragraphs.append(Paragraph(type="equation", content=fixed_content))
                else:
                    # Если лечение не помогло, ставим флаг для агентов
                    flagged_content = f"[REQUIRES_FACTCHECK] {content}"
                    current_paragraphs.append(Paragraph(type="equation", content=flagged_content))
            continue

        # 3. Обработка графики и таблиц
        if block_type in ["image", "table"]:
            # Генерируем ID, если MinerU его не дал (например, Vis_0, Vis_1)
            visual_id = block.get("id") or f"Vis_{len(visuals)}"
            caption = block.get("caption", "")

            visuals.append(
                VisualMeta(id=visual_id, path=block.get("img_path", ""), caption=caption)
            )

            # Добавляем текстовую репрезентацию в параграфы
            if block_type == "table" and content:
                flat_text = flatten_markdown_table(content)
                current_paragraphs.append(Paragraph(type="table", content=flat_text))
            else:
                # Заглушка для обычной картинки, чтобы не терять контекст в тексте
                current_paragraphs.append(
                    Paragraph(type="text", content=f"[{visual_id}: {caption}]")
                )
            continue

        # 4. Обычный текст
        if block_type == "text" and content:
            current_paragraphs.append(Paragraph(type="text", content=content))

    # Сохраняем последний накопленный раздел после выхода из цикла
    if current_paragraphs:
        sections.append(Section(heading=current_heading, level=1, paragraphs=current_paragraphs))

    return ParsedDocument(doi=doi, title=title, sections=sections, visuals=visuals)
