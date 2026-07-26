"""Модуль настройки структурированного логирования."""

import logging
import sys

import structlog

from parser_db.config import settings


def setup_logging() -> None:
    """
    Конфигурирует подсистему логирования для всего приложения.

    В консоль логи выводятся в человекочитаемом цветном формате.
    В файл логи пишутся в формате JSON для удобного парсинга машинами.
    В режиме DEBUG добавляется детальная информация о вызываемом коде.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    shared_processors: list[structlog.types.Processor] = [
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

    # Форматирование для консоли
    console_renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Форматирование для файла (строгий JSON)
    json_renderer = structlog.processors.JSONRenderer()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    file_handler = logging.FileHandler(settings.get_log_path(), encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            # Маршрутизатор: в консоль красиво, в лог-файл (если он есть) - JSON
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter_console = structlog.stdlib.ProcessorFormatter(processor=console_renderer)
    handlers[0].setFormatter(formatter_console)

    formatter_file = structlog.stdlib.ProcessorFormatter(processor=json_renderer)
    file_handler.setFormatter(formatter_file)
