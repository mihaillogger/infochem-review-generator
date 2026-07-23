"""Брокер задач TaskIQ для Модуля 2."""

from taskiq_redis import RedisStreamBroker

from parser_db.config import settings

broker = RedisStreamBroker(url=settings.REDIS_URL)
