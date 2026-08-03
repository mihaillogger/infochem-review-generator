"""Модуль для извлечения данных MinerU и преобразования их в схемы."""

import os
import re
from enum import StrEnum
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from PIL import Image
from thefuzz import fuzz  # type: ignore

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
    """Адаптивно сжимает таблицу: простую в Markdown, сложную — в HTML."""
    is_complex = "colspan" in html_markup.lower() or "rowspan" in html_markup.lower()
    if not is_complex:
        return md(html_markup, strip=["a", "img"], heading_style="ATX").strip()

    soup = BeautifulSoup(html_markup, "html.parser")
    for tag in soup(["a", "img", "div", "span"]):
        tag.unwrap()

    for tag in soup.find_all(True):
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr.lower() not in ["colspan", "rowspan"]:
                del tag[attr]

    return str(soup).replace("\n", "").strip()


def normalize_section_name(heading: str) -> SectionType:
    """Приводит заголовок к строгому Enum через fuzzy matching."""
    clean_heading = re.sub(r"^[\d\.\sIVX]+", "", heading).strip().lower()

    mapping = {
        SectionType.ABSTRACT: ["abstract", "background"],
        SectionType.INTRODUCTION: ["introduction", "intro"],
        SectionType.CONCEPTS_AND_MECHANISMS: [
            "concept",
            "mechanism",
            "principle",
            "theory",
            "interaction",
            "behavior",
        ],
        SectionType.MATERIALS_AND_SYNTHESIS: [
            "material",
            "synthesis",
            "fabrication",
            "preparation",
            "structure",
            "composite",
            "route",
            "experimental",
        ],
        SectionType.APPLICATIONS: [
            "application",
            "device",
            "delivery",
            "therapy",
            "sensor",
            "patterning",
        ],
        SectionType.PERSPECTIVES_AND_CONCLUSIONS: [
            "conclusion",
            "summary",
            "prospect",
            "perspective",
            "future",
            "outlook",
        ],
    }

    best_match = SectionType.UNKNOWN
    highest_score = 0.0

    for sec_type, keywords in mapping.items():
        for kw in keywords:
            score = fuzz.partial_ratio(kw, clean_heading)
            if score > highest_score:
                highest_score = score
                best_match = sec_type

    if highest_score >= 80:
        return best_match

    return SectionType.UNKNOWN


def extract_exact_visual_id(caption: str, default_id: str) -> str:
    """Вытягивает точный ID из подписи для препроцессора."""
    if not caption:
        return default_id
    match = re.match(r"^((?:Fig\.|Figure|Table|Scheme)\s*\d+[a-zA-Z]?)", caption, re.IGNORECASE)
    return match.group(1).strip() if match else default_id


def is_smiles(text: str) -> bool:
    """Проверяет, похожа ли строка на химическую нотацию SMILES."""
    smiles_pattern = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\=#/]+$")
    return bool(smiles_pattern.match(text))


def is_table_broken(html_markup: str) -> bool:
    """Определяет, сломана ли структура HTML-таблицы с помощью DOM-дерева."""
    if not html_markup or len(html_markup) < 30:
        return True

    soup = BeautifulSoup(html_markup, "html.parser")
    clean_text = soup.get_text(separator=" ")

    for w in clean_text.split():
        if len(w) > 35 and "http" not in w and not is_smiles(w):
            return True

    cells = soup.find_all(["td", "th"])
    total_cells = len(cells)
    empty_cells = sum(1 for c in cells if not c.get_text(strip=True))

    if total_cells > 0 and (empty_cells / total_cells) > 0.4:
        return True

    return len(soup.find_all("tr")) == 0


def clean_text_lite(text: str) -> str:
    """Очищает текст от базовых артефактов MinerU без потери химических формул."""
    text = text.replace("\u0001", "°").replace("\u0003", "-")

    if "<" in text and ">" in text:
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Обычный IoU для удаления только 100% дубликатов."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - inter_area)
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def stitch_visuals(
    paths: list[str], bboxes: list[list[float]], pages: list[int], out_path: str
) -> str:
    """Собирает фрагменты в визуальную сетку (строки и столбцы) без белых дыр."""
    valid_data: list[dict[str, Any]] = []
    for p, b, pg in zip(paths, bboxes, pages):
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and len(b) == 4:
            valid_data.append({"path": abs_p, "bbox": b, "page": pg})

    if not valid_data:
        return ""

    pages_dict: dict[int, list[dict[str, Any]]] = {}
    for item in valid_data:
        pages_dict.setdefault(item["page"], []).append(item)

    # Используем Any, чтобы mypy не ругался на конфликты типов Pillow
    page_canvases: list[Any] = []

    for pg, items in sorted(pages_dict.items()):
        # 1. Жесткая фильтрация только полных клонов
        filtered_items: list[dict[str, Any]] = []
        for curr_item in items:
            is_duplicate = False
            for prev_item in filtered_items:
                if calculate_iou(curr_item["bbox"], prev_item["bbox"]) > 0.95:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_items.append(curr_item)

        if not filtered_items:
            continue

        if len(filtered_items) == 1:
            with Image.open(filtered_items[0]["path"]) as img_single:
                page_canvases.append(img_single.convert("RGB"))
            continue

        # 2. Сортируем элементы сверху вниз
        filtered_items.sort(key=lambda x: x["bbox"][1])

        # 3. Разбиваем на горизонтальные ряды
        rows: list[list[dict[str, Any]]] = []
        for item in filtered_items:
            if not rows:
                rows.append([item])
            else:
                current_row_bottom = rows[-1][0]["bbox"][3]
                if item["bbox"][1] < current_row_bottom:
                    rows[-1].append(item)
                else:
                    rows.append([item])

        # 4. Собираем ряды и клеим их горизонтально
        row_images: list[Any] = []
        for row in rows:
            row.sort(key=lambda x: x["bbox"][0])

            imgs = [Image.open(x["path"]).convert("RGB") for x in row]
            max_h = max(im.height for im in imgs)

            resized_imgs: list[Any] = []
            for im in imgs:
                if im.height != max_h:
                    new_w = int(im.width * (max_h / im.height))
                    resized_imgs.append(im.resize((new_w, max_h), Image.Resampling.LANCZOS))
                else:
                    resized_imgs.append(im)

            row_w = sum(im.width for im in resized_imgs) + 20 * (len(resized_imgs) - 1)
            row_canvas = Image.new("RGB", (row_w, max_h), "white")

            curr_x = 0
            for r_img in resized_imgs:
                row_canvas.paste(r_img, (curr_x, 0))
                curr_x += r_img.width + 20

            row_images.append(row_canvas)

        # 5. Склеиваем готовые ряды вертикально
        max_w = max(im.width for im in row_images)
        total_h = sum(im.height for im in row_images) + 20 * (len(row_images) - 1)

        pg_canvas = Image.new("RGB", (max_w, total_h), "white")
        curr_y = 0
        for r_img_row in row_images:
            x_offset = (max_w - r_img_row.width) // 2
            pg_canvas.paste(r_img_row, (x_offset, curr_y))
            curr_y += r_img_row.height + 20

        page_canvases.append(pg_canvas)

    if not page_canvases:
        return ""

    # 6. Финальный аккорд: склеиваем страницы друг под другом
    if len(page_canvases) == 1:
        final_canvas = page_canvases[0]
    else:
        max_final_w = max(im.width for im in page_canvases)
        total_final_h = sum(im.height for im in page_canvases) + 20 * (len(page_canvases) - 1)

        final_canvas = Image.new("RGB", (max_final_w, total_final_h), "white")
        current_y = 0
        for pg_img in page_canvases:
            x_offset = (max_final_w - pg_img.width) // 2
            final_canvas.paste(pg_img, (x_offset, current_y))
            current_y += pg_img.height + 20

    abs_out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    final_canvas.save(abs_out)

    return abs_out


def build_parsed_document(
    mineru_data: list[dict[str, Any]], metadata: dict[str, Any], output_images_dir: str = "."
) -> ParsedDocument:
    """Собирает объект ParsedDocument с буферизацией и склейкой визуальных блоков."""
    sections: list[Section] = []
    visuals: list[VisualMeta] = []

    current_heading_str = "Metadata / Abstract"
    current_heading_enum = SectionType.ABSTRACT
    current_paragraphs: list[Paragraph] = []
    current_level = 1

    vis_buffer_paths: list[str] = []
    vis_buffer_bboxes: list[list[float]] = []
    vis_buffer_pages: list[int] = []
    current_main_caption = ""
    current_visual_id = ""

    def flush_visual_buffer() -> None:
        nonlocal \
            vis_buffer_paths, \
            vis_buffer_bboxes, \
            vis_buffer_pages, \
            current_main_caption, \
            current_visual_id
        if not vis_buffer_paths:
            return

        exact_id = current_visual_id or f"Vis_{len(visuals)}"
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", exact_id)
        out_path = os.path.join(output_images_dir, f"stitched_{safe_id}.png")

        final_path = stitch_visuals(vis_buffer_paths, vis_buffer_bboxes, vis_buffer_pages, out_path)
        if not final_path:
            final_path = os.path.abspath(vis_buffer_paths[0])

        visuals.append(VisualMeta(id=exact_id, path=final_path, caption=current_main_caption))
        current_paragraphs.append(
            Paragraph(type="image", content=f"[{exact_id}: {current_main_caption}]")
        )

        vis_buffer_paths.clear()
        vis_buffer_bboxes.clear()
        vis_buffer_pages.clear()
        current_main_caption = ""
        current_visual_id = ""

    for block in mineru_data:
        block_type = block.get("type", "")
        raw_content = block.get("text", "").strip()
        content = clean_text_lite(raw_content) if block_type == "text" else raw_content

        if block_type in ["text", "equation", "inline_equation", "table"]:
            if block_type == "text":
                if len(content) < 5 and re.match(r"^\(?[a-zA-Z]\)?$", content):
                    if vis_buffer_paths and not current_main_caption:
                        current_main_caption = content
                    continue

                if re.match(r"^(Fig\.|Figure|Table|Scheme)", content, re.IGNORECASE):
                    new_id = extract_exact_visual_id(content, "")
                    if new_id and current_visual_id and new_id != current_visual_id:
                        flush_visual_buffer()

                    current_main_caption = content
                    current_visual_id = new_id or f"Vis_{len(visuals)}"
                    continue

            flush_visual_buffer()

        if block_type == "text" and "text_level" in block:
            if content == current_heading_str:
                continue

            if current_paragraphs:
                sections.append(
                    Section(
                        original_heading=current_heading_str,
                        macro_category=current_heading_enum.value,
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
            if img_path:
                img_path = os.path.abspath(img_path)
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

        if block_type in ["image", "chart"]:
            img_path = block.get("img_path", "")
            bbox = block.get("bbox")
            page_idx = block.get("page_idx", 0)

            raw_caption = block.get("image_caption", [])
            if isinstance(raw_caption, list) and raw_caption:
                caption_text = " ".join(raw_caption).strip()
            elif isinstance(raw_caption, str):
                caption_text = raw_caption.strip()
            else:
                caption_text = ""

            new_id = extract_exact_visual_id(caption_text, "") if caption_text else ""

            if new_id and current_visual_id and new_id != current_visual_id:
                flush_visual_buffer()

            if caption_text and not current_main_caption:
                current_main_caption = caption_text

            if new_id:
                current_visual_id = new_id
            elif caption_text and not current_visual_id:
                current_visual_id = f"Vis_{len(visuals)}"

            if img_path and bbox:
                vis_buffer_paths.append(img_path)
                vis_buffer_bboxes.append(bbox)
                vis_buffer_pages.append(page_idx)
            continue

        if block_type == "table":
            table_html = block.get("table_body", "")
            if not table_html:
                continue

            raw_caption = block.get("table_caption", [])
            caption = " ".join(raw_caption).strip() if raw_caption else ""
            img_path = block.get("img_path", "")
            if img_path:
                img_path = os.path.abspath(img_path)

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
            continue

        if block_type == "text" and content:
            current_paragraphs.append(Paragraph(type="text", content=content))

    flush_visual_buffer()

    if current_paragraphs:
        sections.append(
            Section(
                original_heading=current_heading_str,
                macro_category=current_heading_enum.value,
                level=current_level,
                paragraphs=current_paragraphs,
            )
        )

    return ParsedDocument(
        doi=metadata.get("doi", ""),
        title=metadata.get("title"),
        authors=metadata.get("authors", []),
        year=metadata.get("year"),
        journal=metadata.get("journal"),
        abstract=metadata.get("abstract"),
        sections=sections,
        visuals=visuals,
    )