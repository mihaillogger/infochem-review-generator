"""Тесты обертки над SentenceTransformer (префиксы Nomic)."""

from unittest.mock import MagicMock, patch

import torch
from parser_db.embedder import NomicEmbedder


@patch("parser_db.embedder.SentenceTransformer")
def test_nomic_embedder_prefixes(mock_transformer: MagicMock) -> None:
    """Проверяет, что эмбеддер подставляет правильные префиксы перед кодированием."""
    NomicEmbedder._instance = None

    # Мокаем возврат тензоров, чтобы прошел этап L2-нормализации
    mock_instance = mock_transformer.return_value
    mock_instance.encode.return_value = torch.ones((2, 768))

    embedder = NomicEmbedder()

    # 1. Тест документа
    texts_doc = ["Первый", "Второй"]
    embedder.encode_batch(texts_doc, is_document=True)

    # Проверяем, с какими аргументами реально вызвалась модель
    call_args = mock_instance.encode.call_args[0][0]
    assert call_args == ["search_document: Первый", "search_document: Второй"]

    # 2. Тест запроса
    texts_query = ["Запрос"]
    embedder.encode_batch(texts_query, is_document=False)

    call_args_query = mock_instance.encode.call_args[0][0]
    assert call_args_query == ["search_query: Запрос"]
