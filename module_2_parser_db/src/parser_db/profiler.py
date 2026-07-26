"""Модуль профилирования времени выполнения функций."""

import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import structlog

from parser_db.config import settings

logger = structlog.get_logger(__name__)


def profile_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для замера времени выполнения функции.

    Работает только если settings.DEBUG == True.
    Поддерживает как синхронные, так и асинхронные функции.

    Args:
        func: Оборачиваемая функция.

    Returns:
        Обёртка, которая логирует время выполнения, или сама функция (если не дебаг).
    """
    if not settings.DEBUG:
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            duration = round(time.perf_counter() - start_time, 4)

            logger.debug("profiling_result", function=func.__name__, duration_s=duration)
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = round(time.perf_counter() - start_time, 4)

        logger.debug("profiling_result", function=func.__name__, duration_s=duration)
        return result

    return sync_wrapper
