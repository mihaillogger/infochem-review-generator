"""
Глобальные настройки и фикстуры для pytest.
"""

import os
import tempfile
from collections.abc import Generator

import pytest

test_log_dir = tempfile.mkdtemp(prefix="infochem_test_logs_")

os.environ["LOG_DIR"] = test_log_dir
os.environ["API_KEY"] = "test_fake_api_key_for_pytest"
os.environ["REDIS_PASSWORD"] = "test_fake_redis_password_for_pytest"


from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from parser_db.main import app  # noqa: E402
from parser_db.store import get_store  # noqa: E402


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """
    Глобальный тестовый клиент FastAPI, доступный всем тестам.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_store() -> Generator[AsyncMock, None, None]:
    """
    Глобальная фикстура для подмены слоя БД (Qdrant).
    """
    store = AsyncMock()
    app.dependency_overrides[get_store] = lambda: store
    yield store
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def cleanup_temp_dir() -> Generator[None, None, None]:
    """
    Автоматическая фикстура, которая очищает временную папку логов
    после того, как все тесты завершатся.
    """
    yield
    try:
        import shutil

        shutil.rmtree(test_log_dir, ignore_errors=True)
    except Exception:
        pass
