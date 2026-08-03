"""Модуль семантического чанкинга на основе подготовленных блоков текста."""

import uuid
from typing import Any

import numpy as np
import structlog

from parser_db.config import settings
from parser_db.embedder import NomicEmbedder
from parser_db.preprocessor import (
    build_sandwiches,
    build_visuals_patterns,
    count_tokens,
    extract_visual_ids,
    split_recursively,
)
from parser_db.profiler import profile_time
from parser_db.schemas import DBChunk, DBChunkMetadata, ParsedDocument

logger = structlog.get_logger(__name__)


class ChunkBuilder:
    """Сборщик семантических чанков с инкапсуляцией состояния (Builder)."""

    text_blocks: list[str]
    tokens_count: int
    metadata: DBChunkMetadata

    def __init__(
        self, document: ParsedDocument, original_heading_path: str, macro_category_path: str
    ):
        """
        Инициализирует сборщик чанков.

        Args:
            document (ParsedDocument): Полный распарсенный документ для извлечения метаданных.
            original_heading_path (str): Иерархический путь оригинальных заголовков.
            macro_category_path (str): Иерархический путь макро-категорий.
        """
        self.document = document
        self.original_heading_path = original_heading_path
        self.macro_category_path = macro_category_path
        self.reset()

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние для сборки нового чанка."""
        self.text_blocks: list[str] = []
        self.tokens_count: int = 0
        self.metadata = DBChunkMetadata(
            doi=self.document.doi,
            title=self.document.title,
            authors=self.document.authors,
            year=self.document.year,
            journal=self.document.journal,
            abstract=self.document.abstract,
            original_heading_path=self.original_heading_path,
            macro_category_path=self.macro_category_path,
            linked_images={},
            contains_table=False,
            contains_math=False,
            raw_table_markup=None,
            raw_math_markup=[],
            has_broken_table=False,
            has_broken_math=False,
            has_broken_text=False,
            fallback_table_paths=[],
            fallback_math_paths=[],
        )

    def add_block(self, block: dict[str, Any], block_tokens: int, images: dict[str, str]) -> None:
        """
        Добавляет блок текста и обновляет метаданные.

        Args:
            block (dict[str, Any]): Словарь с данными извлекаемого абзаца.
            block_tokens (int): Количество токенов в блоке.
            images (dict[str, str]): Словарь найденных картинок (ID -> Path).
        """
        self.text_blocks.append(block["text"])
        self.tokens_count += block_tokens
        self.metadata.linked_images.update(images)

        if block.get("contains_table"):
            self.metadata.contains_table = True
            self.metadata.raw_table_markup = block.get("raw_table_markup")

        math_markup = block.get("raw_math_markup")
        if block.get("contains_math") and isinstance(math_markup, list):
            self.metadata.contains_math = True
            if self.metadata.raw_math_markup is None:
                self.metadata.raw_math_markup = []
            self.metadata.raw_math_markup.extend(math_markup)

        if block.get("is_broken_table"):
            self.metadata.has_broken_table = True
            fallback_table = block.get("fallback_table_path")
            if fallback_table:
                self.metadata.fallback_table_paths.append(fallback_table)

        if block.get("is_broken_math"):
            self.metadata.has_broken_math = True
            fallback_math = block.get("fallback_math_path")
            if fallback_math:
                self.metadata.fallback_math_paths.append(fallback_math)

        if block.get("is_broken_text"):
            self.metadata.has_broken_text = True

    def build(self) -> DBChunk | None:
        """
        Собирает и возвращает готовый чанк, затем сбрасывает состояние.

        Returns:
            DBChunk | None: Готовый объект чанка или None, если блоков не было.
        """
        if not self.text_blocks:
            return None

        chunk = DBChunk(
            chunk_id=str(uuid.uuid4()),
            text="\n\n".join(self.text_blocks),
            metadata=self.metadata.model_copy(),
        )
        self.reset()
        return chunk


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Вычисляет косинусное сходство для L2-нормализованных векторов.

    Поскольку входящие векторы уже нормализованы эмбеддером,
    косинусное сходство математически равно их скалярному произведению.

    Args:
        vec1 (np.ndarray): Первый L2-нормализованный вектор.
        vec2 (np.ndarray): Второй L2-нормализованный вектор.

    Returns:
        float: Значение косинусного сходства от -1.0 до 1.0.
    """
    return float(np.dot(vec1, vec2))


@profile_time
def chunk_document(document: ParsedDocument, embedder: NomicEmbedder) -> list[DBChunk]:
    """
    Пайплайн нарезки документа на семантические чанки со скользящим средним.

    Args:
        document (ParsedDocument): Распарсенный документ от парсера.
        embedder (NomicEmbedder): Инициализированный клиент для векторизации.

    Returns:
        list[DBChunk]: Готовые чанки для загрузки в БД.
    """
    logger.info("chunking_started", doi=document.doi, sections_count=len(document.sections))

    chunks: list[DBChunk] = []

    image_map = {
        visual.id: visual.path for visual in document.visuals if getattr(visual, "path", None)
    }

    visual_patterns = build_visuals_patterns(document.visuals)

    heading_stack: list[tuple[int, str, str]] = []

    for section in document.sections:
        while heading_stack and heading_stack[-1][0] >= section.level:
            heading_stack.pop()

        # Кладем текущий заголовок в стек
        heading_stack.append((section.level, section.original_heading, section.macro_category))

        current_original_path = " > ".join(orig for _, orig, _ in heading_stack)
        current_macro_path = " > ".join(macro for _, _, macro in heading_stack)

        raw_blocks = build_sandwiches(section.paragraphs)
        if not raw_blocks:
            continue

        blocks: list[Any] = []
        for block in raw_blocks:
            if count_tokens(block["text"]) > settings.CHUNK_LIMIT and not block.get("is_sandwich"):
                split_texts, is_broken_text = split_recursively(block["text"], settings.CHUNK_LIMIT)
                for st in split_texts:
                    blocks.append({**block, "text": st, "is_broken_text": is_broken_text})
            else:
                blocks.append({**block, "is_broken_text": False})

        if not blocks:
            continue

        embeddings = embedder.encode_batch([b["text"] for b in blocks], is_document=True)

        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(blocks) - 1)
        ]

        cut_indices = set()
        if similarities:
            ema = similarities[0]
            for i, sim in enumerate(similarities):
                # Обновляем экспоненциальное скользящее среднее
                ema = settings.EMA_ALPHA * sim + (1 - settings.EMA_ALPHA) * ema

                # Разрыв, если текущее сходство сильно ниже локального EMA
                if sim < ema - settings.DROP_THRESHOLD:
                    cut_indices.add(i)
                    # Сбрасываем EMA для нового чанка
                    if i + 1 < len(similarities):
                        ema = similarities[i + 1]

        builder = ChunkBuilder(document, current_original_path, current_macro_path)

        for i, block in enumerate(blocks):
            block_tokens = count_tokens(block["text"])

            # Триггеры для принудительного разрыва ДО добавления текущего блока
            token_overflow = (builder.tokens_count + block_tokens > settings.CHUNK_LIMIT) and bool(
                builder.text_blocks
            )
            table_collision = builder.metadata.contains_table and block.get("contains_table")

            if token_overflow or table_collision:
                chunk = builder.build()
                if chunk:
                    chunks.append(chunk)

            # Добавляем данные блока в собираемый чанк
            images = extract_visual_ids(block["text"], visual_patterns, image_map)
            builder.add_block(block, block_tokens, images)

            # Семантический разрыв ПОСЛЕ добавления блока (по скользящему среднему)
            if i in cut_indices and builder.text_blocks and i < len(blocks) - 1:
                chunk = builder.build()
                if chunk:
                    chunks.append(chunk)

        # Сохраняем хвост
        chunk = builder.build()
        if chunk:
            chunks.append(chunk)

    logger.info("chunking_finished", doi=document.doi, total_chunks=len(chunks))
    return chunks
