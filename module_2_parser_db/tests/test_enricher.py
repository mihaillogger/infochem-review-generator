"""Тесты модуля LLM-обогащения."""

from unittest.mock import AsyncMock, patch

import pytest
from parser_db.enricher import enrich_document
from parser_db.schemas import Paragraph, ParsedDocument, Section, StandardSection


@pytest.mark.asyncio
@patch("parser_db.enricher.settings.USE_LLM_ENRICHMENT", True)
@patch("parser_db.enricher.settings.GEMINI_API_KEY", "test_key")
@patch("parser_db.enricher.settings.GEMINI_API_URL", "test_url")
async def test_enrich_document_success() -> None:
    """Проверяет успешное обогащение таблиц."""
    doc = ParsedDocument(
        doi="10.123",
        sections=[
            Section(
                level=1,
                original_heading="H1",
                macro_category=StandardSection.UNKNOWN,
                paragraphs=[
                    Paragraph(type="table", content="<table>1</table>", is_broken=False),
                    Paragraph(type="text", content="Just text", is_broken=False),
                ],
            )
        ],
    )

    mock_client = AsyncMock()
    mock_client.summarize_table.return_value = "Awesome Summary"

    enriched_doc = await enrich_document(doc, mock_client)

    mock_client.summarize_table.assert_called_once_with("<table>1</table>")
    assert (
        enriched_doc.sections[0].paragraphs[0].enriched_summary
        == "[Table Summary]: Awesome Summary"
    )
    assert enriched_doc.sections[0].paragraphs[1].enriched_summary is None


@pytest.mark.asyncio
@patch("parser_db.enricher.settings.USE_LLM_ENRICHMENT", False)
async def test_enrichment_skipped_if_disabled() -> None:
    """Проверяет Graceful Degradation: пропуск при выключенной интеграции с LLM."""
    doc = ParsedDocument(doi="10.123", sections=[])
    mock_client = AsyncMock()

    result = await enrich_document(doc, mock_client)

    assert result == doc
    mock_client.summarize_table.assert_not_called()
