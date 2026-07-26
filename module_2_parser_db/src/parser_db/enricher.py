"""Модуль LLM-обогащения документа перед чанкингом."""

import asyncio

import structlog

from parser_db.config import settings
from parser_db.llm import AsyncGeminiClient
from parser_db.profiler import profile_time
from parser_db.schemas import Paragraph, ParsedDocument

logger = structlog.get_logger(__name__)


async def _enrich_paragraph(
    para: Paragraph, client: AsyncGeminiClient, semaphore: asyncio.Semaphore
) -> None:
    """Обогащает один абзац (таблицу), соблюдая лимиты параллелизма."""
    if para.type != "table" or para.is_broken:
        return

    async with semaphore:
        try:
            summary = await client.summarize_table(para.content)
            if summary:
                # Вшиваем флаг, что это саммари, чтобы эмбеддер понял контекст
                para.enriched_summary = f"[Table Summary]: {summary}"
        except Exception as e:
            logger.warning("llm_enrichment_failed", error=str(e), table_content=para.content[:100])


@profile_time
async def enrich_document(doc: ParsedDocument) -> ParsedDocument:
    """
    Прогоняет таблицы документа через LLM для получения саммари.
    Обеспечивает полный Offline Fallback, если LLM выключена или недоступна.
    """
    if (
        not settings.USE_LLM_ENRICHMENT
        or not settings.GEMINI_API_KEY
        or not settings.GEMINI_API_URL
    ):
        logger.info("llm_enrichment_skipped")
        return doc

    client = AsyncGeminiClient()
    semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY_LIMIT)

    tasks = []
    for section in doc.sections:
        for para in section.paragraphs:
            tasks.append(_enrich_paragraph(para, client, semaphore))

    if tasks:
        logger.info("llm_enrichment_started", tasks_count=len(tasks), doi=doc.doi)
        await asyncio.gather(*tasks)
        logger.info("llm_enrichment_completed", doi=doc.doi)

    return doc
