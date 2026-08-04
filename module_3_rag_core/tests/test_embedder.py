"""Тесты обертки над SentenceTransformer (префиксы Nomic)."""

from unittest.mock import MagicMock, patch

import torch
from rag_core.config import settings
from rag_core.embedder import NomicEmbedder


@patch("rag_core.embedder.SentenceTransformer")
def test_nomic_embedder_prefixes(mock_transformer: MagicMock) -> None:
    """Проверяет, что эмбеддер подставляет правильные префиксы перед кодированием."""
    NomicEmbedder._instance = None

    mock_instance = mock_transformer.return_value
    mock_instance.encode.return_value = torch.ones((2, 768))

    embedder = NomicEmbedder()

    # 1. Тест документа
    texts_doc = ["Первый", "Второй"]
    embedder.encode_batch(texts_doc, is_document=True)

    call_args = mock_instance.encode.call_args[0][0]
    assert call_args == [
        f"{settings.EMBEDDING_PREFIX_DOC}Первый",
        f"{settings.EMBEDDING_PREFIX_DOC}Второй",
    ]

    # 2. Тест запроса
    texts_query = ["Запрос"]
    embedder.encode_batch(texts_query, is_document=False)

    call_args_query = mock_instance.encode.call_args[0][0]
    assert call_args_query == [f"{settings.EMBEDDING_PREFIX_QUERY}Запрос"]


@patch("rag_core.embedder.SentenceTransformer")
def test_nomic_embedder_batch_zero(mock_transformer: MagicMock) -> None:
    """Проверяет, что при batch_size=0 размер батча равен длине массива (отключение батчей)."""
    NomicEmbedder._instance = None

    mock_instance = mock_transformer.return_value
    mock_instance.encode.return_value = torch.ones((3, 768))

    embedder = NomicEmbedder()
    texts = ["Один", "Два", "Три"]

    embedder.encode_batch(texts, is_document=True, batch_size=0)

    # При batch_size=0 внутренний размер батча должен динамически стать равен len(texts)
    call_kwargs = mock_instance.encode.call_args[1]
    assert call_kwargs["batch_size"] == 3


@patch("rag_core.embedder.SentenceTransformer")
def test_encode_batch_empty_list(mock_transformer: MagicMock) -> None:
    """Проверяет корректную отработку пустого списка для векторизации."""
    NomicEmbedder._instance = None
    embedder = NomicEmbedder()

    result = embedder.encode_batch([])

    assert result.size == 0
    mock_transformer.return_value.encode.assert_not_called()
