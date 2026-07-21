"""Брокер задач TaskIQ для Модуля 2."""

from taskiq_redis import ListQueueBroker

from parser_db.config import settings

broker = ListQueueBroker(settings.REDIS_URL)
