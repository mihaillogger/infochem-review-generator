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
    API_KEY: str
    API_KEY_HEADER_NAME: str = "X-API-Key"

    # Инфраструктурные переменные
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    REDIS_PASSWORD: str
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SOCKET_TIMEOUT: int = 10

    PROCESSED_DATA_ROOT: str = "/data/processed_db"
    SHARED_DATA_ROOT: str = "/data/test_pdfs"
    PARSED_FILE_SUFFIX: str = "_final.json"

    # Настройки вычислений
    COMPUTE_DEVICE: str = "cpu"

    # Настройки моделей
    EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = (
        4  # 0 для снятия ограничения (векторизация всего массива за один проход)
    )
    EMBEDDING_PREFIX_DOC: str = "search_document: "
    EMBEDDING_PREFIX_QUERY: str = "search_query: "
    MAX_TOKENS: int = 8192

    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    # Настройки коллекции Qdrant
    COLLECTION_NAME: str = "infochem_docs"
    DENSE_DISTANCE: str = "Cosine"
    DENSE_DATATYPE: str = "float32"
    SPARSE_ON_DISK: bool = False

    QDRANT_BASE_THRESHOLD: float = 0.70

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
    SEARCH_MAX_LIMIT: int = 10000
    SEARCH_PREFETCH_MULTIPLIER: int = 2

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

    LLM_PROMPT_TABLE_SUMMARY: str = (
        "You are an expert AI in chemistry and material science. "
        "Analyze the following table from a scientific paper and provide a concise, "
        "dense text summary of its contents, trends, and key variables. "
        "Constraints:\n"
        "1. Respond strictly in English.\n"
        "2. Do not use markdown tables or bullet points.\n"
        "3. Output ONLY the raw text summary without any introductory phrases "
        "(e.g., do not write 'Here is...').\n"
        "4. If the table is empty, unreadable, or contains only meaningless formatting artifacts, "
        "output exactly: UNREADABLE_TABLE.\n\n"
    )

    # Настройки логирования и профилирования
    DEBUG: bool = False
    LOG_DIR: str = "logs"
    LOG_FILE: str = "app.log"
    PROFILING_FILE: str = "profiler.jsonl"

    # Файлы логов
    LOG_MAX_BYTES: int = 10485760  # 10 МБ
    LOG_BACKUP_COUNT: int = 5  # 5 старых архивов

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_log_path(self) -> Path:
        """
        Возвращает абсолютный путь к файлу логов, создавая папку при необходимости.

        Returns:
            Path: Абсолютный путь к файлу логов.
        """
        log_path = Path(self.LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path / self.LOG_FILE

    def get_profiling_path(self) -> Path:
        """
        Возвращает абсолютный путь к файлу метрик профилирования.

        Returns:
            Path: Абсолютный путь к файлу метрик.
        """
        log_path = Path(self.LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path / self.PROFILING_FILE


settings = Settings()  # type: ignore[call-arg]
