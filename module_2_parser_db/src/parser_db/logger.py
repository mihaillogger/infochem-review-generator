"""Модуль настройки структурированного логирования."""

import logging
import logging.config
import logging.handlers

import structlog

from parser_db.config import settings

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("taskiq").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


def setup_logging() -> None:
    """
    Конфигурирует подсистему логирования для всего приложения.

    В консоль логи выводятся в человекочитаемом цветном формате.
    В файл LOG_FILE пишутся общие системные логи в формате JSON.
    В файл PROFILING_FILE пишутся исключительно метрики времени выполнения.
    В режиме DEBUG добавляется детальная информация о вызываемом коде.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.DEBUG:
        shared_processors.insert(
            1,
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
        )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.processors.JSONRenderer(),
                },
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(colors=True),
                },
            },
            "handlers": {
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": log_level,
                    "stream": "ext://sys.stdout",
                },
                "app_file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(settings.get_log_path()),
                    "maxBytes": settings.LOG_MAX_BYTES,
                    "backupCount": settings.LOG_BACKUP_COUNT,
                    "formatter": "json",
                    "level": log_level,
                    "encoding": "utf-8",
                },
                "profiler_file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(settings.get_profiling_path()),
                    "maxBytes": settings.LOG_MAX_BYTES,
                    "backupCount": settings.LOG_BACKUP_COUNT,
                    "formatter": "json",
                    "level": "DEBUG",
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console_handler", "app_file_handler"],
                    "level": log_level,
                },
                "parser_db.profiler": {
                    "handlers": ["profiler_file_handler"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            },
        }
    )

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
