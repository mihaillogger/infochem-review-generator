"""Тесты для ядра парсера, извлечения таблиц и математических уравнений."""

import os
from typing import Any
from unittest.mock import patch

from parser.equations import validate_latex
from parser.extractor import (
    SectionType,
    build_parsed_document,
    extract_exact_visual_id,
    is_boilerplate_text,
    is_non_content_section,
    is_smiles,
    is_table_broken,
    normalize_section_name,
    optimize_table_markup,
)
from parser.schemas import ParsedDocument


def test_validate_latex_balanced() -> None:
    """Проверяет, что правильные формулы проходят AST-валидацию.

    Используются сырые строки (raw strings) для корректной передачи слешей в pylatexenc.
    """
    assert validate_latex(r"x^2 + y^2 = z^2") is True
    assert validate_latex(r"\int_{a}^{b} (x+1) dx") is True
    assert validate_latex(r"\begin{matrix} a & b \\ c & d \end{matrix}") is True


def test_validate_latex_broken() -> None:
    """Проверяет, что сломанные формулы отлавливаются AST-парсером."""
    assert validate_latex(r"{x^2 + y^2") is False
    assert validate_latex(r"\frac{1}{2") is False
    assert validate_latex(r"\begin{matrix} 1") is False
    assert validate_latex(r"\begin{equation} 1 \end{matrix}") is False


def test_is_smiles() -> None:
    """Проверяет регулярное выражение для детекта химической нотации SMILES."""
    assert is_smiles("CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)CCC(=O)OCC") is True
    assert is_smiles("O=C=O") is True
    assert is_smiles("Just a normal long english word that should fail") is False


def test_normalize_section_name() -> None:
    """Проверяет fuzzy-маппинг заголовков на основе словаря синонимов."""
    assert normalize_section_name("1. Introduction") == SectionType.INTRODUCTION
    assert (
        normalize_section_name("1.2.4 Reaction Mechanisms and Theory")
        == SectionType.CONCEPTS_AND_MECHANISMS
    )
    assert (
        normalize_section_name("II. Nanomaterial Synthesis") == SectionType.MATERIALS_AND_SYNTHESIS
    )
    assert normalize_section_name("Clinical Applications") == SectionType.APPLICATIONS
    assert normalize_section_name("Future Perspectives") == SectionType.PERSPECTIVES_AND_CONCLUSIONS

    # Проверка работы thefuzz для нестандартных заголовков
    assert normalize_section_name("Experimental Section") == SectionType.MATERIALS_AND_SYNTHESIS
    assert normalize_section_name("Custom Header Unknown") == SectionType.UNKNOWN

    # "Results and Discussion" — одна из самых частых секций в химических
    # статьях, раньше не входила ни в один список синонимов и всегда
    # улетала в Unknown (regression, ~14% всех секций корпуса).
    assert (
        normalize_section_name("3. Results and Discussion") == SectionType.CONCEPTS_AND_MECHANISMS
    )
    assert normalize_section_name("2.3. Characterization") == SectionType.MATERIALS_AND_SYNTHESIS
    assert normalize_section_name("2.1. Chemicals") == SectionType.MATERIALS_AND_SYNTHESIS
    assert normalize_section_name("Photoelectrochemical measurements") == (
        SectionType.MATERIALS_AND_SYNTHESIS
    )

    # Заголовок вразрядку ("A B S T R A C T") — OCR/MinerU иногда рендерит
    # так, fuzzy-мэтчинг на нём проигрывает из-за пробелов между буквами.
    assert normalize_section_name("A B S T R A C T") == SectionType.ABSTRACT

    # Regression: старый fuzz.partial_ratio по всей строке ловил случайные
    # (и по смыслу неверные) совпадения короткого ключевого слова внутри
    # совсем другого слова: "concept" vs "concentration" = 86,
    # "method" vs "methanol" = 80 — оба выше порога 80, оба реально
    # ловились на корпусе ("Effect of catalyst concentration" уезжало
    # в Concepts & Mechanisms). После перехода на word-level fuzz.ratio
    # такие заголовки должны честно оставаться Unknown, а не получать
    # неверный лейбл.
    assert normalize_section_name("Effect of catalyst concentration") == SectionType.UNKNOWN
    assert (
        normalize_section_name("3.5. Effects of initial pH, concentration of As(III) and Cr(VI)")
        == SectionType.UNKNOWN
    )

    # "Charge separation"/"Charge transfer" — устойчивая фраза про механизм
    # фотокатализа. Заодно проверяет, что "separation" не отдаёт это в
    # Materials & Synthesis через случайную коллизию с "preparation" (86).
    assert normalize_section_name("Charge separation") == SectionType.CONCEPTS_AND_MECHANISMS
    assert (
        normalize_section_name("3.3. Charge-carriers separation and transport")
        == SectionType.CONCEPTS_AND_MECHANISMS
    )
    # Но обычные "Preparation of X" (синтез) не должны пострадать.
    assert (
        normalize_section_name("2.1. Preparation of catalysts")
        == SectionType.MATERIALS_AND_SYNTHESIS
    )


def test_is_non_content_section() -> None:
    """Административные разделы (не содержательный текст статьи) должны
    отсекаться по заголовку целиком, а не попадать в вывод как Unknown."""
    for heading in [
        "Acknowledgements",
        "ACKNOWLEDGMENTS",
        "References",
        "References:",
        "CRediT authorship contribution statement",
        "Credit author statement",
        "Author Contributions",
        "Authors' contributions",
        "Funding",
        "Data Availability Statement",
        "Declaration of Competing Interest",
        "Compliance with ethical standards",
        "Declarations",
        "Additional information",
        "Publisher's note",
        "Correspondence",
        "ORCID",
        "Abbreviations",
        "ARTICLEINFO",
        "Accepted Manuscript",
        "Just Accepted",
        "Journal Pre-proof",
        "check for updates",
        "ARTICLE",
        "OPEN ACCESS",
        "PAPER",
        "Appendix A. Supplementary data",
        "Appendix A. Supporting information",
    ]:
        assert is_non_content_section(heading) is True, heading

    for heading in [
        "3. Results and Discussion",
        "1. Introduction",
        "Keywords",
        "Highlights",
        "Literature Review",
    ]:
        assert is_non_content_section(heading) is False, heading


def test_is_boilerplate_text() -> None:
    """MinerU иногда типизирует copyright/license-плашки как обычный 'text',
    а не 'header'/'footer' — их отлавливает уже контентный, а не типовой фильтр."""
    assert is_boilerplate_text("© 2021 The Authors. Published by Wiley-VCH.") is True
    assert (
        is_boilerplate_text("This article is protected by copyright. All rights reserved") is True
    )
    assert (
        is_boilerplate_text(
            "Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims."
        )
        is True
    )
    assert (
        is_boilerplate_text("Supporting Information is available from the Wiley Online Library.")
        is True
    )
    assert (
        is_boilerplate_text(
            "Springer Nature or its licensor holds exclusive rights to this article."
        )
        is True
    )
    assert (
        is_boilerplate_text(
            "Received: 13 February 2023 / Accepted: 20 April 2023 / Published online: 16 May 2023"
        )
        is True
    )
    assert (
        is_boilerplate_text(
            "Springer Nature remains neutral with regard to jurisdictional claims in maps."
        )
        is True
    )
    assert (
        is_boilerplate_text(
            "Copyright © 2014, Hydrogen Energy Publications, LLC. Published by Elsevier Ltd."
        )
        is True
    )
    assert is_boilerplate_text("www.elsevier.com/locate/nanoenergy") is True
    assert (
        is_boilerplate_text(
            "Copyright: © 2023 by the authors. Licensee MDPI, Basel, Switzerland. This "
            "article is an open access article distributed under the terms and "
            "conditions of the Creative Commons Attribution license."
        )
        is True
    )
    assert is_boilerplate_text("COPYRIGHT") is True
    assert is_boilerplate_text("(https://creativecommons.org/licenses/by/4.0/).") is True
    assert is_boilerplate_text("This article was downloaded by: [University of Arizona]") is True
    # Легитимная подпись к рисунку с атрибуцией перепечатки — не мусор, оставляем как есть.
    assert (
        is_boilerplate_text(
            "Band edge positioning of pristine g-C3N4. Reproduced with permission. "
            "Copyright 2019, Royal Society of Chemistry."
        )
        is False
    )
    assert (
        is_boilerplate_text(
            "Carbon nitride shows strong photocatalytic activity under visible light."
        )
        is False
    )


def test_extract_exact_visual_id() -> None:
    """Проверяет извлечение точного ID из подписей к изображениям и таблицам."""
    assert extract_exact_visual_id("Fig. 1 Shows the process", "Vis_1") == "Fig. 1"
    assert extract_exact_visual_id("Table 2: Results", "Vis_2") == "Table 2"
    assert extract_exact_visual_id("No prefix here", "Vis_3") == "Vis_3"


def test_table_optimization_and_validation() -> None:
    """Проверяет логику сжатия и валидации HTML-таблиц с использованием BeautifulSoup."""
    clean_flat_html = "<table><tr><td>A</td><td>B</td></tr></table>"
    clean_complex_html = (
        '<table><tr><th colspan="2">A</th></tr><tr><td>B</td><td>C</td></tr></table>'
    )
    broken_html = f"<table><tr><td>{'a' * 40}</td></tr></table>"
    smiles_html = f"<table><tr><td>{'C' * 40}</td></tr></table>"

    assert is_table_broken(broken_html) is True
    assert is_table_broken(smiles_html) is False
    assert is_table_broken(clean_flat_html) is False

    # Длинная LaTeX-формула составного материала — не мусор, таблица не должна
    # улетать в VLM-фоллбэк только из-за неё (regression: реальные статьи корпуса
    # содержат такие названия, например '$Ni_{1.5}Co_{1.5}S_{4}@g-C_{3}N_{4}$').
    latex_formula_html = (
        "<table><tr><td>$Ni_{1.5}Co_{1.5}S_{4}@g-C_{3}N_{4}$</td><td>ok</td></tr></table>"
    )
    assert is_table_broken(latex_formula_html) is False

    assert "|" in optimize_table_markup(clean_flat_html)
    assert "colspan" in optimize_table_markup(clean_complex_html)

    # markdownify по умолчанию экранирует '_' — ломает LaTeX-формулы в ячейках
    # (regression: 'C_3N_4' не должен превращаться в 'C\_3N\_4').
    latex_cell_html = "<table><tr><td>mpg- $C_3N_4$ [mg]</td></tr></table>"
    assert "C_3N_4" in optimize_table_markup(latex_cell_html)


def test_build_parsed_document_structure() -> None:
    """Проверяет сборку Pydantic-модели с новой логикой склейки, page_idx и ID-кэпшенами."""
    mock_mineru_data: list[dict[str, Any]] = [
        {"type": "text", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "This is a test paragraph."},
        {
            "type": "equation",
            "text": r"\frac{1}{2",
            "img_path": "img/broken_math.png",
            "page_idx": 1,
        },
        {
            "type": "image",
            "img_path": "img/mock_fig_1.png",
            "bbox": [10.0, 10.0, 100.0, 100.0],
            "page_idx": 1,
            "image_caption": "Fig. 1 Architecture of the model.",
            "content": "A schematic diagram of the model architecture.",
        },
    ]

    mock_metadata: dict[str, Any] = {
        "doi": "10.1000/xyz123",
        "title": "Test Review Paper",
        "authors": ["Ivanov I.I.", "Petrov P.P."],
        "year": 2026,
        "journal": "Nature Chemistry",
        "abstract": "Test abstract text",
    }

    output_dir = "test_images_output"

    with patch("os.path.exists", return_value=False):
        doc = build_parsed_document(
            mock_mineru_data,
            metadata=mock_metadata,
            output_images_dir=output_dir,
        )

    # Проверка базовой схемы и маппинга метаданных
    assert isinstance(doc, ParsedDocument)
    assert doc.doi == "10.1000/xyz123"
    assert doc.title == "Test Review Paper"
    assert len(doc.authors) == 2
    assert doc.year == 2026

    # Проверка парсинга секций
    assert len(doc.sections) == 1
    assert doc.sections[0].original_heading == "1. Introduction"
    assert doc.sections[0].macro_category == SectionType.INTRODUCTION.value
    assert doc.sections[0].level == 1

    # Должно быть 3 параграфа: текст, сломанная формула и картинка из буфера
    assert len(doc.sections[0].paragraphs) == 3

    # Проверка флагов поломки и конвертации путей
    broken_math_paragraph = doc.sections[0].paragraphs[1]
    assert broken_math_paragraph.is_broken is True
    assert broken_math_paragraph.image_fallback_path is not None

    # Проверяем, что путь конвертировался в докеровский формат или остался абсолютным
    fallback_path = broken_math_paragraph.image_fallback_path
    assert fallback_path.startswith("/") or os.path.isabs(fallback_path) is True

    expected_path = "img/broken_math.png"
    assert expected_path in fallback_path.replace("\\", "/")

    # Проверка парсинга изображений с кэпшенами и генерации путей склейки
    image_paragraph = doc.sections[0].paragraphs[2]
    assert image_paragraph.type == "image"
    assert image_paragraph.content is not None
    assert "Fig. 1" in image_paragraph.content

    assert len(doc.visuals) == 1
    visual = doc.visuals[0]
    assert visual.id == "Fig. 1"
    assert visual.caption == "Fig. 1 Architecture of the model."
    assert visual.path is not None
    assert visual.vlm_description == "A schematic diagram of the model architecture."

    assert "mock_fig_1.png" in visual.path.replace("\\", "/")


def test_stale_caption_does_not_leak_into_later_uncaptioned_images() -> None:
    """Regression: подпись без картинки рядом (напр. отдельный список
    'Figure captions' перед блоком картинок в конце документа — типичный
    формат accepted-manuscript препринтов) не должна "прилипать" ко всем
    последующим безподписным картинкам вплоть до конца статьи. Раньше
    flush_visual_buffer() при пустом буфере выходил раньше, чем успевал
    сбросить current_visual_id/caption, и десятки не связанных друг с другом
    рисунков склеивались в один файл под чужим ID."""
    mock_data: list[dict[str, Any]] = [
        {"type": "text", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "Scheme 2. A caption listed with no image right next to it."},
        {"type": "text", "text": "A normal body paragraph that should flush the empty buffer."},
        {
            "type": "image",
            "img_path": "img/uncaptioned.png",
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "page_idx": 5,
        },
    ]
    with patch("os.path.exists", return_value=False):
        doc = build_parsed_document(mock_data, metadata={"doi": "10.1/x"}, output_images_dir=".")

    assert len(doc.visuals) == 1
    assert doc.visuals[0].id == "Vis_0"  # не "Scheme 2" — та подпись стала мусором


def test_build_parsed_document_drops_non_content_sections() -> None:
    """Административные разделы (Acknowledgements, References, ...) не
    должны попадать в итоговый список sections — это чистый шум для
    review-контента и векторной БД, не содержательный текст статьи."""
    mock_data: list[dict[str, Any]] = [
        {"type": "text", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "Real body paragraph about carbon nitride photocatalysis."},
        {"type": "text", "text_level": 1, "text": "Acknowledgements"},
        {"type": "text", "text": "This work was supported by a grant from Foo Foundation."},
        {"type": "text", "text_level": 1, "text": "References"},
        {"type": "text", "text": "[1] A. Author et al., J. Catal. 2020, 1, 1."},
    ]
    doc = build_parsed_document(mock_data, metadata={"doi": "10.1/x"}, output_images_dir=".")

    headings = [s.original_heading for s in doc.sections]
    assert headings == ["1. Introduction"]


def test_visual_buffer_splits_on_page_change_without_caption() -> None:
    """Regression: без единой подписи между картинками (типичный
    accepted-manuscript препринт-формат, где все Figure captions собраны
    отдельным списком в конце документа) буфер раньше рос неограниченно и
    мог склеить вообще все рисунки статьи в одну гигантскую картинку (реально
    наблюдалось: 20000+ px высотой). Без подписи нет вообще никакого сигнала,
    что картинки с разных страниц относятся к одной фигуре — режем на каждой
    смене страницы."""
    mock_data: list[dict[str, Any]] = [
        {"type": "text", "text_level": 1, "text": "1. Introduction"},
    ] + [
        {
            "type": "image",
            "img_path": f"img/page{page}.png",
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "page_idx": page,
        }
        for page in range(4)
    ]
    with patch("os.path.exists", return_value=False):
        doc = build_parsed_document(mock_data, metadata={"doi": "10.1/x"}, output_images_dir=".")

    assert len(doc.visuals) == 4


def test_visual_buffer_allows_multi_page_span_with_real_caption() -> None:
    """С реально распознанной подписью ('Fig. 1') доверяем больше — легитимные
    multi-page фигуры не растягиваются больше чем на пару-тройку соседних
    страниц, поэтому порог мягче, чем для полностью безподписных картинок."""
    mock_data: list[dict[str, Any]] = [
        {"type": "text", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "Fig. 1 A figure spanning three pages."},
    ]
    mock_data += [
        {
            "type": "image",
            "img_path": f"img/page{page}.png",
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "page_idx": page,
        }
        for page in range(4)
    ]
    with patch("os.path.exists", return_value=False):
        doc = build_parsed_document(mock_data, metadata={"doi": "10.1/x"}, output_images_dir=".")

    # Страницы 0-2 (первые 3 разные) — один визуал под "Fig. 1", страница 3
    # (4-я разная) — уже следующая фигура.
    assert len(doc.visuals) == 2
    assert doc.visuals[0].id == "Fig. 1"
