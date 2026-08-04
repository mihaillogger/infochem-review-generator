"""Брокер задач TaskIQ для Модуля 2."""

from taskiq_redis import RedisStreamBroker

from rag_core.config import settings

broker = RedisStreamBroker(
    url=settings.REDIS_URL,
    password=settings.REDIS_PASSWORD,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
    health_check_interval=settings.REDIS_SOCKET_TIMEOUT,
    retry_on_timeout=True,
)
