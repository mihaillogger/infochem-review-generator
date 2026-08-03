"""Модуль профилирования времени выполнения функций."""

import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import structlog

from parser_db.config import settings

logger = structlog.get_logger("parser_db.profiler")


def _extract_loggable_kwargs(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Извлекает и фильтрует аргументы функции для безопасного логирования.

    Args:
        func: Оборачиваемая функция.
        args: Позиционные аргументы.
        kwargs: Именованные аргументы.

    Returns:
        Словарь со скалярными параметрами и длинами коллекций.
    """
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        log_data = {}
        for key, value in bound_args.arguments.items():
            # Защита от переполнения логов: пишем только метрики, влияющие на скорость
            if isinstance(value, (int, float, bool, str)) and len(str(value)) < 100:
                log_data[key] = value
            elif isinstance(value, (list, tuple, dict, set)):
                log_data[f"{key}_len"] = len(value)
        return log_data
    except Exception:
        # Fallback предотвращает падение приложения при ошибках рефлексии
        return {}


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

            extra_logs = _extract_loggable_kwargs(func, args, kwargs)
            logger.debug(
                "profiling_result", function=func.__name__, duration_s=duration, **extra_logs
            )
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = round(time.perf_counter() - start_time, 4)

        extra_logs = _extract_loggable_kwargs(func, args, kwargs)
        logger.debug("profiling_result", function=func.__name__, duration_s=duration, **extra_logs)
        return result

    return sync_wrapper
