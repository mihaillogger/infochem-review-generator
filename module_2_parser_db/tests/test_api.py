from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from parser_db.main import app

client = TestClient(app)


def test_search_validation_error_rfc9457() -> None:
    """
    Проверяем, что при отправке кривого JSON агент получает
    правильно отформатированную ошибку с инструкцией.
    """

    # Отправляем пустой запрос (без обязательного поля query)
    response = client.post("/api/v1/search", json={})

    assert response.status_code == 422
    data = response.json()

    # Проверяем структуру RFC 9457
    assert "type" in data
    assert "title" in data
    assert "detail" in data
    assert "errors" in data

    # Проверяем, что в detail есть наша кастомная инструкция для LLM
    assert "Твой JSON-запрос не прошел валидацию" in data["detail"]
    assert "query" in data["detail"]  # Указание на пропущенное поле
    assert "исправь тип данных и повтори вызов" in data["detail"]


@patch("parser_db.main.parse_pdf_task.kiq", new_callable=AsyncMock)
def test_ingest_documents_success(mock_kiq: AsyncMock) -> None:
    """Проверяем, что эндпоинт загрузки корректно принимает список файлов."""

    class MockTask:
        task_id = "test-task-123"

    mock_kiq.return_value = MockTask()

    payload = {"file_paths": ["/data/pdfs/article1.pdf", "/data/pdfs/article2.pdf"]}

    response = client.post("/api/v1/documents", json=payload)

    assert response.status_code == 202
    data = response.json()

    assert data["status"] == "accepted"
    assert "task_id" in data
    assert data["task_id"] == "test-task-123"

    mock_kiq.assert_called_once_with(["/data/pdfs/article1.pdf", "/data/pdfs/article2.pdf"])
