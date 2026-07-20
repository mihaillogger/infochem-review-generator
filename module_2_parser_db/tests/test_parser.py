from parser_db.equations import fix_latex_brackets, validate_latex
from parser_db.extractor import build_parsed_document
from parser_db.schemas import ParsedDocument


def test_validate_latex_balanced() -> None:
    """Проверяем, что правильные формулы проходят валидацию."""
    assert validate_latex("{x^2 + y^2 = z^2}") is True
    assert validate_latex("\\int_{a}^{b} (x+1) dx") is True


def test_validate_latex_broken() -> None:
    """Проверяем, что сломанные формулы отлавливаются."""
    assert validate_latex("{x^2 + y^2") is False  # Нет закрывающей
    assert validate_latex("\\frac{1}{2") is False  # Нет закрывающей


def test_fix_latex_brackets() -> None:
    """Проверяем, что функция лечит забытые скобки в конце."""
    broken = "\\frac{1}{2"
    fixed = fix_latex_brackets(broken)
    assert fixed == "\\frac{1}{2}"
    assert validate_latex(fixed) is True


def test_build_parsed_document_structure() -> None:
    """Проверяем, что сырой JSON корректно собирается в Pydantic-модель."""
    mock_mineru_data = [
        {"type": "text", "layout_type": "heading", "text": "1. Introduction"},
        {"type": "text", "text": "This is a test paragraph."},
        {"type": "equation", "text": "E=mc^2"},
        {"type": "image", "id": "Fig 1", "img_path": "/test/path.png", "caption": "Test Image"},
    ]

    doc = build_parsed_document(mock_mineru_data, doi="10.000", title="Test")

    # Проверяем, что Pydantic всё съел и ничего не упало
    assert isinstance(doc, ParsedDocument)
    assert len(doc.sections) == 1
    assert doc.sections[0].heading == "1. Introduction"
    # Должно быть 3 абзаца: текст, формула и текстовая заглушка для картинки
    assert len(doc.sections[0].paragraphs) == 3
    # Картинка должна улететь в метаданные visuals
    assert len(doc.visuals) == 1
    assert doc.visuals[0].caption == "Test Image"
