"""Асинхронный клиент для работы с Gemini API."""

from typing import cast

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from parser_db.config import settings
from parser_db.profiler import profile_time


logger = structlog.get_logger(__name__)


class AsyncGeminiClient:
    def __init__(self) -> None:
        self.api_key = settings.GEMINI_API_KEY
        self.url = f"{settings.GEMINI_API_URL}?key={self.api_key}"

    @retry(
        stop=stop_after_attempt(settings.LLM_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1, min=settings.LLM_RETRY_MIN_WAIT, max=settings.LLM_RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    @profile_time
    async def summarize_table(self, table_markup: str, caption: str = "") -> str | None:
        """Отправляет сырую таблицу в LLM и возвращает плотное текстовое саммари."""
        if not self.api_key:
            return None

        prompt = (
            "You are an expert AI in chemistry and material science. "
            "Analyze the following table from a scientific paper and provide a concise, "
            "dense text summary of its contents, trends, and key variables. "
            "Do not use markdown tables in your response. Write only the summary.\n\n"
        )
        if caption:
            prompt += f"Table Caption: {caption}\n"
        prompt += f"Table Markdown:\n{table_markup}"

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()

            data = response.json()
            try:
                summary = data["candidates"][0]["content"]["parts"][0]["text"]
                logger.debug("llm_summarize_success", table_length=len(table_markup))

                return cast(str, summary).strip()
            except (KeyError, IndexError):
                logger.error("llm_unexpected_response_format", response_data=data)
                return None
