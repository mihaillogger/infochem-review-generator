"""Асинхронный клиент для работы с Gemini API."""

import re
from typing import Any, cast

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_core.config import settings
from rag_core.profiler import profile_time

logger = structlog.get_logger(__name__)


def _on_retry_error(retry_state: Any) -> None:
    """
    Коллбэк tenacity при исчерпании всех попыток.

    Args:
        retry_state (Any): Текущее состояние попыток и исключений tenacity.

    Returns:
        None
    """
    logger.error("llm_retries_exhausted", error=str(retry_state.outcome.exception()))
    return None


class AsyncGeminiClient:
    def __init__(self) -> None:
        """Инициализирует клиент для работы с Gemini API."""
        self.api_key = settings.GEMINI_API_KEY
        self.url = f"{settings.GEMINI_API_URL}?key={self.api_key}"

    @retry(
        stop=stop_after_attempt(settings.LLM_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1, min=settings.LLM_RETRY_MIN_WAIT, max=settings.LLM_RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type(httpx.HTTPError),
        retry_error_callback=_on_retry_error,
    )
    @profile_time
    async def summarize_table(self, table_markup: str, caption: str = "") -> str | None:
        """Отправляет сырую таблицу в LLM и возвращает плотное текстовое саммари.

        Args:
            table_markup (str): Markdown-разметка таблицы для анализа.
            caption (str, optional): Подпись к таблице. По умолчанию "".

        Returns:
            str | None: Текстовое саммари таблицы или None, если API-ключ не задан
                или ответ имеет неожиданный формат.
        """
        if not self.api_key:
            return None

        user_content = ""
        if caption:
            user_content += f"Table Caption: {caption}\n"
        user_content += f"Table Markdown:\n{table_markup}"

        payload = {
            "systemInstruction": {"parts": [{"text": settings.LLM_PROMPT_TABLE_SUMMARY}]},
            "contents": [{"parts": [{"text": user_content}]}],
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()

            data = response.json()
            try:
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                summary = cast(str, raw_text).strip()

                # Удаляем markdown-обертки
                summary = re.sub(r"^```[a-zA-Z]*\n|```$", "", summary).strip()

                if "UNREADABLE_TABLE" in summary:
                    logger.warning("llm_unreadable_table_detected", table_length=len(table_markup))
                    return None

                logger.debug("llm_summarize_success", table_length=len(table_markup))
                return summary
            except (KeyError, IndexError):
                logger.error("llm_unexpected_response_format", response_data=data)
                return None
