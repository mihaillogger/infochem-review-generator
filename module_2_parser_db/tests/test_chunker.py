"""Тесты семантического чанкинга и иерархии разделов."""

from unittest.mock import MagicMock, patch

import numpy as np
from parser_db.chunker import chunk_document
from parser_db.schemas import Paragraph, ParsedDocument, Section


@patch("parser_db.chunker.NomicEmbedder")
def test_chunker_section_paths(mock_embedder_class: MagicMock) -> None:
    """Проверяет правильность формирования стека заголовков."""
    # Получаем инстанс, который вернет класс
    mock_instance = mock_embedder_class.return_value
    mock_instance.encode_batch.return_value = np.random.rand(2, 768)

    document = ParsedDocument(
        doi="10.123",
        title="Test",
        visuals=[],
        sections=[
            Section(
                level=1,
                heading="Methodology",
                paragraphs=[Paragraph(type="text", content="A", is_broken=False)],
            ),
            Section(
                level=2,
                heading="Dataset",
                paragraphs=[Paragraph(type="text", content="B", is_broken=False)],
            ),
        ],
    )

    chunks = chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].metadata.section_path == "Methodology"
    assert chunks[1].metadata.section_path == "Methodology > Dataset"


@patch("parser_db.chunker.NomicEmbedder")
def test_chunker_ema_split(mock_embedder_class: MagicMock) -> None:
    """Проверяет разрыв чанка при резком падении косинусного сходства (EMA)."""
    # Симулируем 3 блока. 1 и 2 очень похожи, 3 - совершенно другой.
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.9, 0.1])  # sim(1,2) ~ 0.9
    vec3 = np.array([-1.0, 0.0])  # sim(2,3) ~ -0.9 (резкое падение)

    mock_instance = mock_embedder_class.return_value
    mock_instance.encode_batch.return_value = np.array([vec1, vec2, vec3])

    document = ParsedDocument(
        doi="10.123",
        title="Test",
        visuals=[],
        sections=[
            Section(
                level=1,
                heading="Text",
                paragraphs=[
                    Paragraph(type="text", content="B1", is_broken=False),
                    Paragraph(type="text", content="B2", is_broken=False),
                    Paragraph(type="text", content="B3", is_broken=False),
                ],
            )
        ],
    )

    # Принудительно ставим лимит токенов огромным, чтобы разрыв был ТОЛЬКО по EMA
    with patch("parser_db.chunker.settings.CHUNK_LIMIT", 10000):
        chunks = chunk_document(document)

    # Должно получиться 2 чанка: [B1, B2] и [B3]
    assert len(chunks) == 2
    assert "B1\n\nB2" in chunks[0].text
    assert "B3" in chunks[1].text
