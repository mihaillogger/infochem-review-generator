"""Тесты асинхронного воркера (TaskIQ)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from parser_db.schemas import ParsedDocument, ParseTaskResult
from parser_db.worker import vectorize_document_task, warmup_models
from pydantic import ValidationError


@pytest.mark.asyncio
@patch("parser_db.worker.Path")
@patch("parser_db.worker.get_store")
@patch("parser_db.worker.chunk_document")
@patch("parser_db.worker.enrich_document")
async def test_vectorize_document_task_success(
    mock_enrich: AsyncMock,
    mock_chunk: MagicMock,
    mock_get_store: AsyncMock,
    mock_path_class: MagicMock,
) -> None:
    """Проверяет успешный проход графа обработки документа в воркере."""

    # Мокаем файловую систему
    mock_pdf_path = MagicMock()
    mock_pdf_path.stem = "test_article"

    mock_json_path = MagicMock()
    mock_json_path.exists.return_value = True

    # Фейковый минимальный JSON документа
    fake_json = '{"doi": "10.000", "sections": [], "visuals": []}'
    mock_json_path.read_text.return_value = fake_json

    # Магия Path: чтобы Path(file_path) и Path(PROCESSED_DATA_ROOT) / file возвращали наши моки
    mock_path_class.return_value = mock_pdf_path
    mock_path_class.return_value.__truediv__.return_value = mock_json_path

    # Мокаем пайплайн обработки
    mock_enrich.return_value = ParsedDocument.model_validate_json(fake_json)
    mock_chunk.return_value = [MagicMock()]  # Возвращаем список из 1 чанка

    mock_store_instance = AsyncMock()
    mock_get_store.return_value = mock_store_instance

    # Мокаем контекст TaskIQ
    mock_context = MagicMock()
    mock_context.message.task_id = "task-123"

    # Выполняем задачу
    result = await vectorize_document_task("/dummy/test_article.pdf", context=mock_context)

    # Проверки
    assert isinstance(result, ParseTaskResult)
    assert result.status == "success"

    mock_enrich.assert_called_once()
    mock_chunk.assert_called_once()
    mock_store_instance.insert_chunks.assert_called_once()


@pytest.mark.asyncio
@patch("parser_db.worker.warmup_tokenizer")
@patch("parser_db.worker.get_store")
async def test_warmup_models(mock_get_store: AsyncMock, mock_tokenizer: MagicMock) -> None:
    """Проверяет хук прогрева ML моделей при старте воркера."""
    mock_store = AsyncMock()
    mock_get_store.return_value = mock_store

    mock_state = MagicMock()
    await warmup_models(mock_state)  # type: ignore[misc]

    mock_tokenizer.assert_called_once()
    mock_store.dense_embedder.encode_batch.assert_called_once()


@pytest.mark.asyncio
async def test_vectorize_document_task_empty_path() -> None:
    """Проверяет выброс ValueError при передаче пустого пути."""
    with pytest.raises(ValueError, match="пустой путь"):
        await vectorize_document_task("", context=MagicMock())


@pytest.mark.asyncio
@patch("parser_db.worker.Path")
async def test_vectorize_document_task_file_not_found(mock_path_class: MagicMock) -> None:
    """Проверяет FileNotFoundError, если парсер не сгенерировал JSON-файл."""
    mock_pdf_path = MagicMock()
    mock_pdf_path.stem = "test"
    mock_json_path = MagicMock()
    mock_json_path.exists.return_value = False

    mock_path_class.return_value = mock_pdf_path
    mock_path_class.return_value.__truediv__.return_value = mock_json_path

    with pytest.raises(FileNotFoundError, match="JSON-файл парсера не найден"):
        await vectorize_document_task("/dummy/test.pdf", context=MagicMock())


@pytest.mark.asyncio
@patch("parser_db.worker.Path")
async def test_vectorize_document_task_invalid_json(mock_path_class: MagicMock) -> None:
    """Проверяет ValidationError при несовпадении схемы Pydantic в JSON."""
    mock_pdf_path = MagicMock()
    mock_pdf_path.stem = "test"
    mock_json_path = MagicMock()
    mock_json_path.exists.return_value = True
    mock_json_path.read_text.return_value = '{"bad": "schema_data"}'

    mock_path_class.return_value = mock_pdf_path
    mock_path_class.return_value.__truediv__.return_value = mock_json_path

    with pytest.raises(ValidationError):
        await vectorize_document_task("/dummy/test.pdf", context=MagicMock())
