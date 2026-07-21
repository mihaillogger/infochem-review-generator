"""
Глобальные настройки Модуля 2.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Инфраструктурные переменные
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    REDIS_URL: str = "redis://localhost:6379/0"

    # Настройки Nomic
    EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIM: int = 768
    MAX_TOKENS: int = 8192
    CHUNK_LIMIT: int = 1024  # Оптимальный лимит для агентов, чтобы избежать Lost in the Middle

    # Настройки чанкинга
    EMA_ALPHA: float = 0.5  # Коэффициент сглаживания для экспоненциального скользящего среднего
    DROP_THRESHOLD: float = 0.15  # Порог падения сходства для разрыва чанка

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
