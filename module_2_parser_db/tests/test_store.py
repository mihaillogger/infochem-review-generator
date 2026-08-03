"""Тесты слоя данных Qdrant (AsyncQdrantStore)."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from parser_db.schemas import DBChunk, DBChunkMetadata
from parser_db.store import AsyncQdrantStore


@pytest.mark.asyncio
@patch("parser_db.store.AsyncQdrantClient")
@patch("parser_db.store.NomicEmbedder")
@patch("parser_db.store.SparseTextEmbedding")
async def test_insert_chunks_deletes_old(
    mock_sparse: MagicMock, mock_nomic: MagicMock, mock_qdrant: MagicMock
) -> None:
    """Проверяет, что перед вставкой вызывается удаление старых чанков по DOI."""
    mock_client = AsyncMock()
    mock_qdrant.return_value = mock_client

    mock_nomic.return_value.encode_batch.return_value = np.array([[0.1, 0.2]])
    mock_sparse.return_value.embed.return_value = [
        MagicMock(indices=np.array([1]), values=np.array([0.5]))
    ]

    store = AsyncQdrantStore()

    chunk = DBChunk(
        chunk_id="test-id",
        text="text_content",
        metadata=DBChunkMetadata(
            doi="10.999",
            original_heading_path="Heading",
            macro_category_path="Category",
        ),
    )

    await store.insert_chunks([chunk])

    # Проверка удаления старых данных
    mock_client.delete.assert_called_once()
    call_kwargs = mock_client.delete.call_args[1]
    assert call_kwargs["collection_name"] == store.collection_name
    assert call_kwargs["points_selector"].must[0].key == "doi"
    assert call_kwargs["points_selector"].must[0].match.value == "10.999"

    # Проверка вставки новых
    mock_client.upsert.assert_called_once()
