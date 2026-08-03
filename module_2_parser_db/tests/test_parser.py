"""Тесты для ядра парсера, извлечения таблиц и математических уравнений."""

import os
from typing import Any
from unittest.mock import patch

from parser_db.equations import validate_latex
from parser_db.extractor import (
    SectionType,
    build_parsed_document,
    extract_exact_visual_id,
    is_smiles,
    is_table_broken,
    normalize_section_name,
    optimize_table_markup,
)
from parser_db.schemas import ParsedDocument


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

    assert "|" in optimize_table_markup(clean_flat_html)
    assert "colspan" in optimize_table_markup(clean_complex_html)


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

    assert "mock_fig_1.png" in visual.path.replace("\\", "/")
