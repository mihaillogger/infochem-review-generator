"""Тесты API-шлюза (FastAPI)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from rag_core.config import settings
from rag_core.main import app

client = TestClient(app)

# Валидный заголовок для прохождения авторизации в тестах
AUTH_HEADERS = {"X-API-Key": settings.API_KEY}


def test_auth_unauthorized() -> None:
    """Проверяет отклонение запросов без валидного X-API-Key."""
    response = client.post("/api/v1/search", json={"query": "test", "limit": 1})
    assert response.status_code == 401

    response = client.post("/api/v1/documents", json={"file_paths": ["/data/pdfs/1.pdf"]})
    assert response.status_code == 401


def test_search_validation_error_rfc9457(mock_store: AsyncMock) -> None:
    """
    Проверяет структуру ошибки RFC 9457 при невалидном запросе.
    Убеждается, что LLM-агент получит правильные инструкции для self-healing.
    """
    response = client.post("/api/v1/search", json={}, headers=AUTH_HEADERS)

    assert response.status_code == 422
    data = response.json()

    assert "type" in data
    assert "title" in data
    assert "detail" in data
    assert "errors" in data
    assert "Твой JSON-запрос не прошел валидацию" in data["detail"]
    assert "query" in data["detail"]


def test_search_success(mock_store: AsyncMock) -> None:
    """Проверяет успешный гибридный поиск и корректность SearchResponse."""
    mock_store.hybrid_search.return_value = [
        {
            "chunk_id": "uuid-123",
            "text": "Найденный текст",
            "metadata": {
                "doi": "10.123",
                "macro_category_path": "Introduction",
                "original_heading_path": "1. Introduction",
            },
        }
    ]

    response = client.post(
        "/api/v1/search", json={"query": "тест", "limit": 1}, headers=AUTH_HEADERS
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert data["data"][0]["chunk_id"] == "uuid-123"


def test_search_internal_error_rfc9457(mock_store: AsyncMock) -> None:
    """Проверяет упаковку 500-й ошибки БД в стандарт RFC 9457."""
    mock_store.hybrid_search.side_effect = Exception("Qdrant is down")

    response = client.post(
        "/api/v1/search", json={"query": "тест", "limit": 1}, headers=AUTH_HEADERS
    )

    assert response.status_code == 500
    data = response.json()
    assert data["status"] == 500
    assert "Внутренняя ошибка сервера" in data["detail"]
    assert "Qdrant is down" not in data["detail"]


@patch("rag_core.main.vectorize_document_task.kiq", new_callable=AsyncMock)
def test_ingest_documents_success(mock_kiq: AsyncMock) -> None:
    """Проверяет корректность постановки задачи в очередь TaskIQ."""

    class MockTask:
        task_id = "test-task-123"

    mock_kiq.return_value = MockTask()

    safe_path = f"{settings.SHARED_DATA_ROOT}/article1.pdf"
    payload = {"file_paths": [safe_path]}
    response = client.post("/api/v1/documents", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 202
    assert response.json()["task_ids"] == ["test-task-123"]

    # Path Traversal Validator резолвит путь в абсолютный, поэтому сравниваем корректно
    import pathlib

    resolved_path = str(pathlib.Path(safe_path).resolve())
    mock_kiq.assert_called_once_with(resolved_path)


def test_ingest_documents_path_traversal() -> None:
    """Проверяет работу валидатора путей (защита от Path Traversal)."""
    malicious_path = f"{settings.SHARED_DATA_ROOT}/../../../etc/shadow"
    payload = {"file_paths": [malicious_path]}

    response = client.post("/api/v1/documents", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert "выходит за пределы защищенной директории" in response.text
