"""Тесты семантического чанкинга и иерархии разделов."""

from unittest.mock import MagicMock, patch

import numpy as np
from parser_db.chunker import chunk_document
from parser_db.schemas import Paragraph, ParsedDocument, Section, StandardSection


@patch("parser_db.chunker.NomicEmbedder")
def test_chunker_section_paths(mock_embedder_class: MagicMock) -> None:
    """Проверяет правильность формирования стека заголовков."""
    mock_embedder = mock_embedder_class.return_value
    mock_embedder.encode_batch.return_value = np.random.rand(2, 768)

    document = ParsedDocument(
        doi="10.123",
        title="Test",
        visuals=[],
        sections=[
            Section(
                level=1,
                original_heading="3. Methodology",
                macro_category=StandardSection.CONCEPTS_AND_MECHANISMS,
                paragraphs=[Paragraph(type="text", content="A", is_broken=False)],
            ),
            Section(
                level=2,
                original_heading="3.1 Dataset Setup",
                macro_category=StandardSection.UNKNOWN,
                paragraphs=[Paragraph(type="text", content="B", is_broken=False)],
            ),
        ],
    )

    chunks = chunk_document(document, mock_embedder)

    assert len(chunks) == 2
    assert chunks[0].metadata.macro_category_path == "Concepts & Mechanisms"
    assert chunks[0].metadata.original_heading_path == "3. Methodology"

    assert chunks[1].metadata.macro_category_path == "Concepts & Mechanisms > Unknown"
    assert chunks[1].metadata.original_heading_path == "3. Methodology > 3.1 Dataset Setup"


@patch("parser_db.chunker.NomicEmbedder")
def test_chunker_ema_split(mock_embedder_class: MagicMock) -> None:
    """Проверяет разрыв чанка при резком падении косинусного сходства (EMA)."""
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.9, 0.1])  # sim(1,2) ~ 0.9
    vec3 = np.array([-1.0, 0.0])  # sim(2,3) ~ -0.9 (резкое падение)

    mock_embedder = mock_embedder_class.return_value
    mock_embedder.encode_batch.return_value = np.array([vec1, vec2, vec3])

    document = ParsedDocument(
        doi="10.123",
        title="Test",
        visuals=[],
        sections=[
            Section(
                level=1,
                original_heading="Text",
                macro_category=StandardSection.UNKNOWN,
                paragraphs=[
                    Paragraph(type="text", content="B1", is_broken=False),
                    Paragraph(type="text", content="B2", is_broken=False),
                    Paragraph(type="text", content="B3", is_broken=False),
                ],
            )
        ],
    )

    with patch("parser_db.chunker.settings.CHUNK_LIMIT", 10000):
        chunks = chunk_document(document, mock_embedder)

    assert len(chunks) == 2
    assert "B1\n\nB2" in chunks[0].text
    assert "B3" in chunks[1].text


@patch("parser_db.chunker.NomicEmbedder")
def test_chunker_broken_table_flag(mock_embedder_class: MagicMock) -> None:
    """Проверяет проброс флага битой таблицы и путей к картинкам из абзаца в чанк."""
    mock_embedder = mock_embedder_class.return_value
    mock_embedder.encode_batch.return_value = np.random.rand(1, 768)

    document = ParsedDocument(
        doi="10.123",
        sections=[
            Section(
                level=1,
                original_heading="Test",
                macro_category=StandardSection.UNKNOWN,
                paragraphs=[
                    Paragraph(
                        type="table",
                        content="<table>",
                        is_broken=True,
                        image_fallback_path="/data/images/broken.png",
                    )
                ],
            )
        ],
    )

    chunks = chunk_document(document, mock_embedder)

    assert len(chunks) == 1
    assert chunks[0].metadata.has_broken_table is True
    assert "/data/images/broken.png" in chunks[0].metadata.fallback_table_paths
