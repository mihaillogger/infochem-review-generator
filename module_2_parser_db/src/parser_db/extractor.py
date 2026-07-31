"""Модуль для извлечения данных MinerU и преобразования их в схемы."""

import os
import re
from enum import StrEnum
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from PIL import Image
from thefuzz import fuzz

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

    # Удаляем строго HTML-теги (начинаются с буквы или /), игнорируя математику типа A < B > C
    if "<" in text and ">" in text:
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Вычисляет метрику Intersection over Union для отсева дубликатов."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / float(box1_area + box2_area - inter_area)


def stitch_visuals(paths: list[str], bboxes: list[list[float]], out_path: str) -> str:
    """Склеивает фрагментированные изображения графиков по координатам с отсевом дублей."""
    valid_data = []
    for p, b in zip(paths, bboxes):
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            valid_data.append((abs_p, b))

    if not valid_data:
        return ""

    # Отсев пересекающихся боксов (дублей от MinerU)
    filtered_data = [valid_data[0]]
    for curr_path, curr_box in valid_data[1:]:
        is_duplicate = False
        for _, prev_box in filtered_data:
            if calculate_iou(curr_box, prev_box) > 0.15:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_data.append((curr_path, curr_box))

    min_x = min(b[0] for _, b in filtered_data)
    min_y = min(b[1] for _, b in filtered_data)
    max_x = max(b[2] for _, b in filtered_data)
    max_y = max(b[3] for _, b in filtered_data)

    width, height = int(max_x - min_x), int(max_y - min_y)
    if width <= 0 or height <= 0:
        return filtered_data[0][0]

    canvas = Image.new("RGB", (width, height), "white")
    for path, bbox in filtered_data:
        with Image.open(path) as img:
            canvas.paste(img, (int(bbox[0] - min_x), int(bbox[1] - min_y)))

    abs_out = os.path.abspath(out_path)
    canvas.save(abs_out)
    return abs_out


def build_parsed_document(
    mineru_data: list[dict[str, Any]], metadata: dict[str, Any]
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
    current_main_caption = ""
    current_visual_id = ""

    def flush_visual_buffer() -> None:
        """Сбрасывает накопленные картинки в один склеенный объект."""
        nonlocal vis_buffer_paths, vis_buffer_bboxes, current_main_caption, current_visual_id
        if not vis_buffer_paths:
            return

        exact_id = current_visual_id or f"Vis_{len(visuals)}"
        out_path = f"stitched_{exact_id.replace(' ', '_')}.png"

        final_path = stitch_visuals(vis_buffer_paths, vis_buffer_bboxes, out_path)
        if not final_path:
            final_path = os.path.abspath(vis_buffer_paths[0])

        visuals.append(VisualMeta(id=exact_id, path=final_path, caption=current_main_caption))
        current_paragraphs.append(
            Paragraph(type="image", content=f"[{exact_id}: {current_main_caption}]")
        )

        vis_buffer_paths.clear()
        vis_buffer_bboxes.clear()
        current_main_caption = ""
        current_visual_id = ""

    for block in mineru_data:
        block_type = block.get("type", "")
        raw_content = block.get("text", "").strip()
        content = clean_text_lite(raw_content) if block_type == "text" else raw_content

        if block_type in ["text", "equation", "inline_equation", "table"]:
            if block_type == "text":
                if len(content) < 5 and re.match(r"^\(?[a-zA-Z]\)?$", content):
                    # Если попался кусок подписи типа "a)", привязываем к буферу
                    if vis_buffer_paths and not current_main_caption:
                        current_main_caption = content
                    continue
                if re.match(r"^(Fig\.|Figure|Table|Scheme)", content, re.IGNORECASE):
                    current_main_caption = content
                    current_visual_id = extract_exact_visual_id(content, f"Vis_{len(visuals)}")
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
            if img_path and bbox:
                vis_buffer_paths.append(img_path)
                vis_buffer_bboxes.append(bbox)
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
