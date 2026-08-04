"""Модуль профилирования времени выполнения функций."""

import inspect
import time
import psutil
from collections.abc import Callable
from functools import wraps
from typing import Any

import structlog

from rag_core.config import settings

logger = structlog.get_logger("parser_db.profiler")
BYTES_IN_MB = 1048576


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
            elif isinstance(value, (list, tuple, set)):
                log_data[f"{key}_len"] = len(value)
                # Считаем символы для массивов строк (например, батчи текстов)
                try:
                    if value and isinstance(value[0], str):
                        log_data[f"{key}_chars"] = sum(len(s) for s in value if isinstance(s, str))
                except Exception:
                    pass
            elif isinstance(value, dict):
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
            process = psutil.Process()
            ram_mb_before = process.memory_info().rss / BYTES_IN_MB
            psutil.cpu_percent(interval=None)  # Инициализация счетчика CPU

            start_time = time.perf_counter()
            status = "success"
            error_type = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                raise
            finally:
                duration = round(time.perf_counter() - start_time, 4)
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_mb_after = process.memory_info().rss / BYTES_IN_MB

                extra_logs = _extract_loggable_kwargs(func, args, kwargs)
                logger.debug(
                    "profiling_result",
                    function=func.__name__,
                    duration_s=duration,
                    status=status,
                    error_type=error_type,
                    cpu_percent=cpu_percent,
                    ram_mb=round(ram_mb_after, 2),
                    ram_diff_mb=round(ram_mb_after - ram_mb_before, 2),
                    **extra_logs
                )

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        process = psutil.Process()
        ram_mb_before = process.memory_info().rss / BYTES_IN_MB
        psutil.cpu_percent(interval=None)  # Инициализация счетчика CPU

        start_time = time.perf_counter()
        status = "success"
        error_type = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            raise
        finally:
            duration = round(time.perf_counter() - start_time, 4)
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_mb_after = process.memory_info().rss / BYTES_IN_MB

            extra_logs = _extract_loggable_kwargs(func, args, kwargs)
            logger.debug(
                "profiling_result",
                function=func.__name__,
                duration_s=duration,
                status=status,
                error_type=error_type,
                cpu_percent=cpu_percent,
                ram_mb=round(ram_mb_after, 2),
                ram_diff_mb=round(ram_mb_after - ram_mb_before, 2),
                **extra_logs
            )

    return sync_wrapper
