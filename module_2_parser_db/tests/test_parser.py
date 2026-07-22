"""Тесты для ядра парсера, извлечения таблиц и математических уравнений."""

from typing import Any

from parser_db.equations import fix_latex_brackets, validate_latex
from parser_db.extractor import (
    build_parsed_document,
    extract_exact_visual_id,
    is_table_broken,
    normalize_section_name,
    optimize_table_markup,
)
from parser_db.schemas import ParsedDocument


def test_validate_latex_balanced() -> None:
    """Проверяет, что правильные формулы проходят валидацию, включая экранирование."""
    assert validate_latex("{x^2 + y^2 = z^2}") is True
    assert validate_latex("\\int_{a}^{b} (x+1) dx") is True
    assert validate_latex("\\{A, B\\}") is True  # Экранированные скобки


def test_validate_latex_broken() -> None:
    """Проверяет, что сломанные формулы отлавливаются."""
    assert validate_latex("{x^2 + y^2") is False
    assert validate_latex("\\frac{1}{2") is False
    assert validate_latex("\\begin{matrix} 1") is False


def test_fix_latex_brackets() -> None:
    """Проверяет, что функция лечит забытые скобки в конце."""
    broken = "\\frac{1}{2"
    fixed = fix_latex_brackets(broken)
    assert fixed == "\\frac{1}{2}"
    assert validate_latex(fixed) is True


def test_normalize_section_name() -> None:
    """Проверяет маппинг заголовков к API-стандартам."""
    assert normalize_section_name("1. Introduction") == "Introduction"
    assert normalize_section_name("Experimental Setup") == "Methodology"
    assert normalize_section_name("Results and Discussions") == "Results"
    assert normalize_section_name("Custom Header") == "Custom Header"


def test_extract_exact_visual_id() -> None:
    """Проверяет извлечение точного ID из подписей."""
    assert extract_exact_visual_id("Fig. 1 Shows the process", "Vis_1") == "Fig. 1"
    assert extract_exact_visual_id("Table 2: Results", "Vis_2") == "Table 2"
    assert extract_exact_visual_id("No prefix here", "Vis_3") == "Vis_3"


def test_table_optimization_and_validation() -> None:
    """Проверяет логику сжатия и валидации HTML-таблиц."""
    clean_flat_html = "<table><tr><td>A</td><td>B</td></tr></table>"
    clean_complex_html = (
        '<table><tr><th colspan="2">A</th></tr><tr><td>B</td><td>C</td></tr></table>'
    )
    broken_html = "<table><tr><td>" + "1" * 40 + "</td></tr></table>"

    assert is_table_broken(broken_html) is True
    assert is_table_broken(clean_flat_html) is False

    # Плоская таблица конвертируется в Markdown
    assert "|" in optimize_table_markup(clean_flat_html)
    # Сложная таблица остается в минифицированном HTML
    assert "colspan" in optimize_table_markup(clean_complex_html)


def test_build_parsed_document_structure() -> None:
    """Проверяет сборку Pydantic-модели с учетом новых флагов и иерархии."""
    mock_mineru_data: list[dict[str, Any]] = [
        {"type": "text", "layout_type": "heading", "text_level": 1, "text": "1. Introduction"},
        {"type": "text", "text": "This is a test paragraph."},
        {"type": "equation", "text": "\\frac{1}{2"},  # Будет вылечено
        {
            "type": "equation",
            "text": "\\begin{matrix} 1",
            "img_path": "/img/1.png",
        },  # Битое (is_broken)
        {
            "type": "image",
            "id": "Vis_99",
            "img_path": "/test/path.png",
            "image_caption": ["Figure 1. Test Image"],
        },
    ]

    doc = build_parsed_document(mock_mineru_data, doi="10.000", title="Test")

    # Проверка базовой структуры и нормализации
    assert isinstance(doc, ParsedDocument)
    assert len(doc.sections) == 1
    assert doc.sections[0].heading == "Introduction"
    assert doc.sections[0].level == 1

    # Должно быть 4 абзаца: текст, формула (вылеченная), формула (битая), текст-заглушка картинки
    assert len(doc.sections[0].paragraphs) == 4

    # Проверка флагов уравнений
    assert doc.sections[0].paragraphs[1].is_broken is False
    assert doc.sections[0].paragraphs[2].is_broken is True
    assert doc.sections[0].paragraphs[2].image_fallback_path == "/img/1.png"

    # Проверка точного извлечения Visual ID
    assert len(doc.visuals) == 1
    assert doc.visuals[0].id == "Figure 1"
