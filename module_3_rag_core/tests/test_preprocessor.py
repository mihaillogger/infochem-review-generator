"""Тесты функций предобработки (сэндвичи, токены, визуал)."""

from unittest.mock import AsyncMock, patch

from rag_core.preprocessor import (
    build_sandwiches,
    build_visuals_patterns,
    extract_visual_ids,
    split_recursively,
)
from rag_core.schemas import Paragraph, VisualMeta


def test_build_sandwiches_with_broken_table() -> None:
    """Проверяет склейку таблиц с соседними абзацами и проброс флагов."""
    paragraphs = [
        Paragraph(type="text", content="Текст до", is_broken=False),
        Paragraph(
            type="table",
            content="<table>",
            is_broken=True,
            image_fallback_path="img.jpg",
        ),
        Paragraph(type="text", content="Текст после", is_broken=False),
    ]

    blocks = build_sandwiches(paragraphs)

    assert len(blocks) == 1
    sandwich = blocks[0]
    assert sandwich["is_sandwich"] is True
    assert sandwich["contains_table"] is True
    assert sandwich["fallback_table_path"] == "img.jpg"
    assert "Текст до\n\n<table>\n\nТекст после" == sandwich["text"]


@patch("rag_core.preprocessor.count_tokens")
def test_split_recursively(mock_count_tokens: AsyncMock) -> None:
    """Проверяет рекурсивное разбиение текста без превышения лимитов токенов.

    Мокаем count_tokens, чтобы считать 1 символ = 1 токен для простоты теста.
    """
    mock_count_tokens.side_effect = len
    text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50

    # Лимит 60, значит куски по 50 должны разделиться
    chunks, is_broken = split_recursively(text, max_tokens=60)

    assert len(chunks) == 3
    assert chunks[0] == "A" * 50
    assert chunks[1] == "B" * 50
    assert is_broken is False


def test_extract_visual_ids() -> None:
    """Проверяет поиск идентификаторов картинок по регулярным выражениям."""
    visuals = [VisualMeta(id="Fig. 1", path="dummy.jpg")]
    patterns = build_visuals_patterns(visuals)
    image_map = {"Fig. 1": "/data/images/fig1.png"}

    text = "Как видно на fig. 1, результаты отличные. И еще Fig. 2."
    found = extract_visual_ids(text, patterns, image_map)

    # Должен найти Fig. 1 игнорируя регистр, Fig. 2 игнорируется
    assert "Fig. 1" in found
    assert len(found) == 1
