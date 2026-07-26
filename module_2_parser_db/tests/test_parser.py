"""Тесты для ядра парсера, извлечения таблиц и математических уравнений."""

from typing import Any

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
    """Проверяет, что правильные формулы проходят валидацию, включая окружения."""
    assert validate_latex("{x^2 + y^2 = z^2}") is True
    assert validate_latex("\\int_{a}^{b} (x+1) dx") is True
    assert validate_latex("\\{A, B\\}") is True
    assert validate_latex("\\begin{matrix} a & b \\\\ c & d \\end{matrix}") is True


def test_validate_latex_broken() -> None:
    """Проверяет, что сломанные формулы и неверные окружения отлавливаются."""
    assert validate_latex("{x^2 + y^2") is False
    assert validate_latex("\\frac{1}{2") is False
    assert validate_latex("\\begin{matrix} 1") is False
    assert validate_latex("\\begin{equation} 1 \\end{matrix}") is False


def test_is_smiles() -> None:
    """Проверяет регулярное выражение для детекта SMILES нотаций."""
    assert is_smiles("CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)CCC(=O)OCC") is True
    assert is_smiles("O=C=O") is True
    assert is_smiles("Just a normal long english word that should fail") is False


def test_normalize_section_name() -> None:
    """Проверяет маппинг заголовков обзорных статей к строгому Enum SectionType."""
    assert normalize_section_name("1. Introduction") == SectionType.INTRODUCTION
    
    assert (
        normalize_section_name("Reaction Mechanisms and Theory") 
        == SectionType.CONCEPTS_AND_MECHANISMS
    )
    
    assert (
        normalize_section_name("Nanomaterial Synthesis") 
        == SectionType.MATERIALS_AND_SYNTHESIS
    )
    
    assert normalize_section_name("Clinical Applications") == SectionType.APPLICATIONS
    
    assert (
        normalize_section_name("Future Perspectives") 
        == SectionType.PERSPECTIVES_AND_CONCLUSIONS
    )
    
    assert normalize_section_name("Custom Header") == SectionType.UNKNOWN


def test_extract_exact_visual_id() -> None:
    """Проверяет извлечение точного ID из подписей."""
    assert extract_exact_visual_id("Fig. 1 Shows the process", "Vis_1") == "Fig. 1"
    assert extract_exact_visual_id("Table 2: Results", "Vis_2") == "Table 2"
    assert extract_exact_visual_id("No prefix here", "Vis_3") == "Vis_3"


def test_table_optimization_and_validation() -> None:
    """Проверяет логику сжатия и валидации HTML-таблиц с учетом химии."""
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
    """Проверяет сборку Pydantic-модели с учетом строгой отбраковки мусора."""
    mock_mineru_data: list[dict[str, Any]] = [
        {"type": "text", "layout_type": "heading", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "This is a test paragraph."},
        {
            "type": "equation",
            "text": "\\frac{1}{2",  
            "img_path": "/img/broken.png",
        },
        {
            "type": "equation",
            "text": "\\begin{matrix} 1 \\end{matrix}",
            "img_path": "/img/ok.png",
        },
        {
            "type": "image",
            "id": "Vis_99",
            "img_path": "/test/path.png",
            "image_caption": ["Figure 1. Test Image"],
        },
    ]

    doc = build_parsed_document(mock_mineru_data, doi="10.000", title="Test")

    assert isinstance(doc, ParsedDocument)
    assert len(doc.sections) == 1
    assert doc.sections[0].heading == SectionType.INTRODUCTION.value
    assert doc.sections[0].level == 1
    assert len(doc.sections[0].paragraphs) == 4

    assert doc.sections[0].paragraphs[1].is_broken is True
    assert doc.sections[0].paragraphs[1].image_fallback_path == "/img/broken.png"
    assert doc.sections[0].paragraphs[2].is_broken is False

    assert len(doc.visuals) == 1
    assert doc.visuals[0].id == "Figure 1"