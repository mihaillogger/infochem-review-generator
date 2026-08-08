"""Модуль для извлечения данных MinerU и преобразования их в схемы."""

import os
import re
from enum import StrEnum
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from PIL import Image
from thefuzz import fuzz  # type: ignore

from parser.equations import validate_latex
from parser.schemas import Paragraph, ParsedDocument, Section, VisualMeta


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
        # escape_underscores=False: markdownify по умолчанию экранирует "_" —
        # это ломает LaTeX-формулы в ячейках химических таблиц (C_3N_4 -> C\_3N\_4).
        return md(
            html_markup, strip=["a", "img"], heading_style="ATX", escape_underscores=False
        ).strip()

    soup = BeautifulSoup(html_markup, "html.parser")
    for tag in soup(["a", "img", "div", "span"]):
        tag.unwrap()

    for tag in soup.find_all(True):
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr.lower() not in ["colspan", "rowspan"]:
                del tag[attr]

    return str(soup).replace("\n", "").strip()


#  Административные/служебные заголовки без содержательного текста для review —
# секции с такими заголовками целиком выбрасываются в build_parsed_document
# (не мусор в смысле is_boilerplate_text — это цельные разделы вроде
# "Acknowledgements"/"CRediT authorship..."/"Funding", а не отдельные абзацы).
NON_CONTENT_SECTION_RE = re.compile(
    r"^(acknowledge?ments?|references?|credit authorship contribution statement|"
    r"credit author statement|author contributions?|authors?'?\s*contributions?|"
    r"authors?|funding|data availability( statement)?|"
    r"declaration of competing interest|conflicts? of interest|"
    r"compliance with ethical standards|declarations?|additional information|"
    r"publisher'?s note|correspondence|orcid|abbreviations?|contents?|"
    r"article\s?info(rmation)?|accepted (manuscript|article)|journal pre-?proofs?|"
    r"just accepted|check for updates|article|open access|paper|review|"
    r"appendix [a-z]\.?\s*(supplementary|supporting)( data| information)?|"
    r"si supporting information|supporting information)[:.]?$",
    re.IGNORECASE,
)


def is_non_content_section(heading: str) -> bool:
    """Проверяет, что заголовок раздела — административная плашка
    (Acknowledgements/References/CRediT/Funding/...), а не содержательный
    текст статьи. Такие разделы в build_parsed_document не идут в вывод —
    для review-контента и векторной БД это чистый шум."""
    clean_heading = re.sub(r"^[\d.\sIVX]+", "", heading).strip()
    return bool(NON_CONTENT_SECTION_RE.match(clean_heading))


def normalize_section_name(heading: str) -> SectionType:
    """Приводит заголовок к строгому Enum через fuzzy matching."""
    clean_heading = re.sub(r"^[\d\.\sIVX]+", "", heading).strip().lower()

    # Некоторые PDF рендерят заголовки вразрядку ("A B S T R A C T") —
    # OCR/MinerU сохраняет пробелы между буквами, fuzzy-мэтчинг на таком
    # проигрывает по длине. Если весь заголовок — однобуквенные "слова",
    # схлопываем пробелы перед сравнением.
    if clean_heading and all(len(w) == 1 for w in clean_heading.split()):
        clean_heading = clean_heading.replace(" ", "")

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
            "result",
            "discussion",
            # "Charge separation"/"charge transfer" — устойчивая фраза про
            # механизм фотокатализа, а не про синтез; заодно точное
            # совпадение "separation" (100) перебивает случайную коллизию
            # "preparation" vs "separation" (86 по fuzz.ratio), из-за
            # которой "Charge separation" раньше уезжало в Materials & Synthesis.
            # ("migration"/"recombination" сюда сознательно не добавлены —
            # они сами коллидируют с "integration"/"combination", которые
            # реально встречаются в заголовках про синтез композитов.)
            "separation",
            "transfer",
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
            "characterization",
            "characterisation",
            "method",
            "measurement",
            "chemical",
            "instrumentation",
            "computational",
            "reaction",
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

    # Сравниваем по отдельным словам через fuzz.ratio (не partial_ratio!) —
    # partial_ratio ищет наиболее похожее ОКНО внутри всей строки и не
    # штрафует за разницу в длине, из-за чего короткое ключевое слово может
    # словить случайную (и по смыслу неверную) подстроку внутри совсем
    # другого слова: "concept" vs "concentration" = 86, "method" vs
    # "methanol" = 80 — оба выше порога 80, оба реально ловились на корпусе
    # ("Effect of catalyst concentration" уезжало в Concepts & Mechanisms).
    # fuzz.ratio сравнивает слова целиком и штрафует за разницу в длине:
    # те же пары дают 60 и 71 — уже ниже порога, а настоящие варианты формы
    # слова (concept/concepts, synthesis/synthesized, material/materials)
    # остаются на 80+.
    words = re.findall(r"[a-zà-ÿ]+", clean_heading)

    best_match = SectionType.UNKNOWN
    highest_score = 0.0

    for sec_type, keywords in mapping.items():
        for kw in keywords:
            for word in words:
                score = fuzz.ratio(kw, word)
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
    if not smiles_pattern.match(text):
        return False

    # Исключаем обычные слова-паразиты, состоящие только из строчных букв
    if text.isalpha() and text.islower():
        return False

    return True


def _is_garbage_word(word: str) -> bool:
    """Длинный токен в таблице — мусор, если это не SMILES, не URL и не LaTeX-формула.

    Составные названия материалов в химических таблицах (например,
    '$Ni_{1.5}Co_{1.5}S_{4}@g-C_{3}N_{4}$') легко превышают 35 символов и не
    проходят строгий SMILES-паттерн — исключаем их отдельно, иначе нормальная
    таблица ошибочно улетает в VLM-фоллбэк.
    """
    if "http" in word:
        return False
    if word.startswith("$") and word.endswith("$"):
        return False
    return not is_smiles(word)


def is_table_broken(html_markup: str) -> bool:
    """Определяет, сломана ли структура HTML-таблицы с помощью DOM-дерева."""
    if not html_markup or len(html_markup) < 30:
        return True

    soup = BeautifulSoup(html_markup, "html.parser")
    clean_text = soup.get_text(separator=" ")

    for w in clean_text.split():
        if len(w) > 35 and _is_garbage_word(w):
            return True

    cells = soup.find_all(["td", "th"])
    total_cells = len(cells)
    empty_cells = sum(1 for c in cells if not c.get_text(strip=True))

    if total_cells > 0 and (empty_cells / total_cells) > 0.4:
        return True

    return len(soup.find_all("tr")) == 0


_BOILERPLATE_TEXT_RE = re.compile(
    r"^(©|Copyright:?\s*©|COPYRIGHT$|WILEY-VCH$"
    r"|\(?https?://creativecommons\.org"
    r"|www\.\w[\w.-]*\.\w+/(locate|journal)"
    r"|A division of the American Chemical Society$"
    r"|Open Access[:.]? This article is licensed"
    r"|This article is protected by copyright"
    r"|This article was downloaded by:"
    r"|Received:?\s+\d{1,2}\s+\w+\s+\d{4}.{0,80}(Accepted|Published))"
    r"|[Ss]upporting [Ii]nformation.{0,40}is available (online|from)"
    r"|Publisher'?s Note Springer Nature"
    r"|Springer Nature remains neutral with regard to jurisdictional claims"
    r"|Springer Nature or its licensor"
    r"|[Tt]his (article|work) is (an )?open access.{0,60}distributed under"
    r"|under the terms of the Creative Commons"
    r"|Creative Commons Attribution",
    re.IGNORECASE,
)


def is_boilerplate_text(text: str) -> bool:
    """Ловит юридический/копирайтный текст, который MinerU иногда типизирует
    как обычный 'text', а не 'header'/'footer' — фильтрация по типу блока
    (см. run_parser.GARBAGE_TYPES) его не ловит."""
    return bool(_BOILERPLATE_TEXT_RE.search(text))


def clean_text_lite(text: str) -> str:
    """Очищает текст от базовых артефактов MinerU без потери химических формул."""
    text = text.replace("\u0001", "°").replace("\u0003", "-")

    if "<" in text and ">" in text:
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def to_docker_path(local_path: str) -> str:
    """Конвертирует локальный абсолютный путь винды в формат Docker (/data/...)."""
    if not local_path:
        return local_path

    # Заменяем обратные слеши на прямые
    normalized = local_path.replace("\\", "/")

    # Ищем папку data и отрезаем всё, что было до неё
    if "/data/" in normalized:
        return normalized[normalized.find("/data/") :]
    elif "data/" in normalized:
        return "/" + normalized[normalized.find("data/") :]

    return normalized


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
    mineru_data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_images_dir: str = ".",
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
    current_vlm_description = ""

    def flush_visual_buffer() -> None:
        nonlocal \
            vis_buffer_paths, \
            vis_buffer_bboxes, \
            vis_buffer_pages, \
            current_main_caption, \
            current_visual_id, \
            current_vlm_description
        if vis_buffer_paths:
            exact_id = current_visual_id or f"Vis_{len(visuals)}"
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", exact_id)
            out_path = os.path.join(output_images_dir, f"stitched_{safe_id}.png")

            final_path = stitch_visuals(
                vis_buffer_paths, vis_buffer_bboxes, vis_buffer_pages, out_path
            )
            if not final_path:
                final_path = os.path.abspath(vis_buffer_paths[0])

            visuals.append(
                VisualMeta(
                    id=exact_id,
                    path=to_docker_path(final_path),
                    caption=current_main_caption or None,
                    vlm_description=current_vlm_description or None,
                )
            )
            current_paragraphs.append(
                Paragraph(type="image", content=f"[{exact_id}: {current_main_caption}]")
            )

            vis_buffer_paths.clear()
            vis_buffer_bboxes.clear()
            vis_buffer_pages.clear()

        # Сбрасываем контекст подписи/ID всегда, а не только когда реально
        # была склеена картинка: иначе подпись, стоящая в тексте без своей
        # картинки рядом (например отдельный список "Figure captions" перед
        # блоком картинок в конце документа), "прилипает" как призрак ко всем
        # следующим безподписным картинкам вплоть до конца статьи — реальный
        # баг, склеивавший десятки не связанных друг с другом рисунков в один
        # гигантский файл под чужим ID.
        current_main_caption = ""
        current_visual_id = ""
        current_vlm_description = ""

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

            if current_paragraphs and not is_non_content_section(current_heading_str):
                sections.append(
                    Section(
                        original_heading=current_heading_str,
                        macro_category=current_heading_enum.value,  # type: ignore
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
                        image_fallback_path=to_docker_path(img_path),
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

            vlm_description = block.get("content", "").strip()

            new_id = extract_exact_visual_id(caption_text, "") if caption_text else ""

            if new_id and current_visual_id and new_id != current_visual_id:
                flush_visual_buffer()

            if caption_text and not current_main_caption:
                current_main_caption = caption_text

            if vlm_description and not current_vlm_description:
                current_vlm_description = vlm_description

            if new_id:
                current_visual_id = new_id
            elif caption_text and not current_visual_id:
                current_visual_id = f"Vis_{len(visuals)}"

            # Предохранитель: если у фигур/чартов подряд нет подписей (типично
            # для препринтов, где все Figure captions собраны отдельным списком
            # в конце документа, а сами картинки идут потом одна за одной без
            # единого текстового блока между ними — естественного повода
            # сбросить буфер просто не возникает), буфер иначе может расти
            # неограниченно и склеить вообще все рисунки статьи в одну
            # гигантскую картинку.
            #
            # Порог зависит от того, есть ли у буфера реальная подпись:
            # если current_main_caption пуст — у нас вообще нет сигнала, что
            # эти картинки относятся к одной фигуре, так что режем на любой
            # смене страницы (макс. 1 страница на буфер). Если подпись есть
            # (реально распознанный "Fig./Table/Scheme ...") — доверяем ей
            # больше: легитимные multi-page фигуры обычно не растягиваются
            # больше чем на пару-тройку соседних страниц.
            max_pages = 3 if current_main_caption else 1
            if (
                img_path
                and bbox
                and vis_buffer_pages
                and page_idx not in vis_buffer_pages
                and len(set(vis_buffer_pages)) >= max_pages
            ):
                flush_visual_buffer()

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
                        image_fallback_path=to_docker_path(img_path),
                    )
                )
            else:
                table_md = optimize_table_markup(table_html)
                table_content = f"Caption: {caption}\n\n{table_md}"
                current_paragraphs.append(Paragraph(type="table", content=table_content))
            continue

        if block_type == "text" and content and not is_boilerplate_text(content):
            current_paragraphs.append(Paragraph(type="text", content=content))

    flush_visual_buffer()

    if current_paragraphs and not is_non_content_section(current_heading_str):
        sections.append(
            Section(
                original_heading=current_heading_str,
                macro_category=current_heading_enum.value,  # type: ignore
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
