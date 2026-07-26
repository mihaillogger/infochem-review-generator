"""Тесты API-шлюза (FastAPI)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from parser_db.main import app
from parser_db.store import get_store

client = TestClient(app)


@patch("parser_db.main.get_store")
def test_search_validation_error_rfc9457(mock_get_store: AsyncMock) -> None:
    """
    Проверяет структуру ошибки RFC 9457 при невалидном запросе.

    Убеждается, что LLM-агент получит правильные инструкции для self-healing.
    """
    mock_get_store.return_value = AsyncMock()
    app.dependency_overrides[get_store] = lambda: mock_get_store.return_value

    response = client.post("/api/v1/search", json={})

    assert response.status_code == 422
    data = response.json()

    assert "type" in data
    assert "title" in data
    assert "detail" in data
    assert "errors" in data
    assert "Твой JSON-запрос не прошел валидацию" in data["detail"]
    assert "query" in data["detail"]

    app.dependency_overrides.clear()


@patch("parser_db.main.get_store")
def test_search_success(mock_get_store: AsyncMock) -> None:
    """Проверяет успешный гибридный поиск и корректность SearchResponse."""
    mock_store = AsyncMock()
    mock_store.hybrid_search.return_value = [
        {
            "chunk_id": "uuid-123",
            "text": "Найденный текст",
            "metadata": {"doi": "10.123", "section_path": "H1"},
        }
    ]
    mock_get_store.return_value = mock_store
    app.dependency_overrides[get_store] = lambda: mock_store

    response = client.post("/api/v1/search", json={"query": "тест", "limit": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert data["data"][0]["chunk_id"] == "uuid-123"

    app.dependency_overrides.clear()


@patch("parser_db.main.get_store")
def test_search_internal_error_rfc9457(mock_get_store: AsyncMock) -> None:
    """Проверяет упаковку 500-й ошибки БД в стандарт RFC 9457."""
    mock_store = AsyncMock()
    mock_store.hybrid_search.side_effect = Exception("Qdrant is down")
    mock_get_store.return_value = mock_store
    app.dependency_overrides[get_store] = lambda: mock_store

    response = client.post("/api/v1/search", json={"query": "тест", "limit": 1})

    assert response.status_code == 500
    data = response.json()
    assert data["status"] == 500
    assert "Qdrant is down" in data["detail"]
    assert "Внутренняя ошибка сервера" in data["detail"]

    app.dependency_overrides.clear()


@patch("parser_db.main.parse_pdf_task.kiq", new_callable=AsyncMock)
def test_ingest_documents_success(mock_kiq: AsyncMock) -> None:
    """Проверяет корректность постановки задачи в очередь TaskIQ."""

    class MockTask:
        task_id = "test-task-123"

    mock_kiq.return_value = MockTask()
    payload = {"file_paths": ["/data/pdfs/article1.pdf"]}
    response = client.post("/api/v1/documents", json=payload)

    assert response.status_code == 202
    assert response.json()["task_id"] == "test-task-123"
    mock_kiq.assert_called_once_with(["/data/pdfs/article1.pdf"])
