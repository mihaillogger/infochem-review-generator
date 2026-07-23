from parser_db.preprocessor import build_sandwiches
from parser_db.schemas import Paragraph


def test_build_sandwiches_simple_text() -> None:
    """Проверяем, что обычный текст не ломается и не получает левых флагов."""
    paragraphs = [
        Paragraph(type="text", content="Первый абзац.", is_broken=False),
        Paragraph(type="text", content="Второй абзац.", is_broken=False),
    ]

    blocks = build_sandwiches(paragraphs)

    assert len(blocks) == 2
    assert blocks[0]["text"] == "Первый абзац."
    assert blocks[0]["is_broken_table"] is False
    assert blocks[0]["is_broken_math"] is False
    assert not blocks[0]["is_sandwich"]


def test_build_sandwiches_with_broken_table() -> None:
    """Проверяем метод сэндвича и правильное распределение флагов битой таблицы."""
    paragraphs = [
        Paragraph(type="text", content="Текст до таблицы.", is_broken=False),
        Paragraph(
            type="table",
            content="<table>...</table>",
            is_broken=True,
            image_fallback_path="images/table1.jpg",
        ),
        Paragraph(type="text", content="Текст после таблицы.", is_broken=False),
    ]

    blocks = build_sandwiches(paragraphs)

    # Должен получиться ровно один склеенный блок (Сэндвич)
    assert len(blocks) == 1
    sandwich = blocks[0]

    assert sandwich["is_sandwich"] is True
    assert sandwich["contains_table"] is True
    assert sandwich["is_broken_table"] is True
    assert sandwich["is_broken_math"] is False
    assert sandwich["fallback_table_path"] == "images/table1.jpg"
    assert sandwich["fallback_math_path"] is None

    # Проверяем правильность склейки текста
    assert "Текст до таблицы." in sandwich["text"]
    assert "<table>...</table>" in sandwich["text"]
    assert "Текст после таблицы." in sandwich["text"]
