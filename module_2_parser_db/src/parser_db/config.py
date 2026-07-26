"""
Глобальные настройки Модуля 2.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Метаданные API
    API_TITLE: str = "Infochem RAG Core API"
    API_DESCRIPTION: str = "Ядро семантического поиска и парсинга научных статей."
    API_VERSION: str = "1.0.0"

    # Инфраструктурные переменные
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    REDIS_URL: str = "redis://localhost:6379/0"

    # Настройки моделей
    EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIM: int = 768
    MAX_TOKENS: int = 8192
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    # Настройки коллекции Qdrant
    COLLECTION_NAME: str = "infochem_docs"
    DENSE_DISTANCE: str = "Cosine"
    DENSE_DATATYPE: str = "float32"
    SPARSE_ON_DISK: bool = False

    # Настройки текстового индекса
    TEXT_INDEX_MIN_LEN: int = 2
    TEXT_INDEX_MAX_LEN: int = 15
    TEXT_INDEX_LOWERCASE: bool = True

    # Настройки чанкинга
    CHUNK_LIMIT: int = 1024
    EMA_ALPHA: float = 0.5  # Коэффициент сглаживания для экспоненциального скользящего среднего
    DROP_THRESHOLD: float = 0.15  # Порог падения сходства для разрыва чанка

    # Настройки LLM обогащения
    USE_LLM_ENRICHMENT: bool = False
    GEMINI_API_KEY: str | None = None
    GEMINI_API_URL: str | None = None

    LLM_CONCURRENCY_LIMIT: int = 5  # Защита от Rate Limit (HTTP 429)
    LLM_RETRY_ATTEMPTS: int = 3  # Количество попыток при падении сети
    LLM_RETRY_MIN_WAIT: int = 2  # Минимальная пауза между ретраями (сек)
    LLM_RETRY_MAX_WAIT: int = 10  # Максимальная пауза (сек)
    LLM_TIMEOUT: float = 30.0  # Таймаут HTTP-запроса

    # Настройки агентов и поиска
    SEARCH_DEFAULT_LIMIT: int = 5
    SEARCH_MAX_LIMIT: int = 20

    # Настройки RFC 9457 (Ошибки и инструкции для агентов)
    RFC_TYPE_VALIDATION: str = "https://datatracker.ietf.org/doc/html/rfc9457#section-3"
    RFC_TYPE_INTERNAL: str = "about:blank"

    # Системные промпты для агентов
    LLM_INSTRUCTION_VALIDATION: str = (
        "Изучи OpenAPI спецификацию этого метода, исправь тип данных и повтори вызов."
    )
    LLM_INSTRUCTION_INTERNAL: str = (
        "Внутренняя ошибка сервера "
        "(возможно, векторная БД недоступна). "
        "Сделай паузу и повтори запрос позже."
    )

    # Настройки логирования и профилирования
    DEBUG: bool = False
    LOG_DIR: str = "logs"
    LOG_FILE: str = "app.log"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_log_path(self) -> Path:
        """Возвращает абсолютный путь к файлу логов, создавая папку при необходимости."""
        log_path = Path(self.LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path / self.LOG_FILE


settings = Settings()
