"""Модуль семантического чанкинга на основе подготовленных блоков текста."""

import uuid
from typing import Any, TypedDict

import numpy as np

from parser_db.config import settings
from parser_db.embedder import NomicEmbedder
from parser_db.preprocessor import (
    build_sandwiches,
    count_tokens,
    extract_validated_visuals,
    split_recursively,
)
from parser_db.schemas import DBChunk, DBChunkMetadata, ParsedDocument

# Инициализируем модель
embedder = NomicEmbedder()


class ChunkMeta(TypedDict):
    contains_table: bool
    contains_math: bool
    raw_table_markup: str | None
    raw_math_markup: list[str]


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


def _save_chunk(
    doi: str,
    section_heading: str,
    chunk_text: list[str],
    meta: ChunkMeta,
    images: set[str],
    chunks_list: list[DBChunk],
) -> None:
    """
    Формирует объект DBChunk и добавляет его в итоговый массив.

    Args:
        doi (str): Идентификатор статьи.
        section_heading (str): Название раздела.
        chunk_text (list[str]): Накопленные строки чанка.
        meta (dict): Накопленные флаги и сырые данные (таблицы/формулы).
        images (set): Множество найденных идентификаторов картинок.
        chunks_list (list[DBChunk]): Массив для добавления результата.
    """
    metadata = DBChunkMetadata(
        doi=doi,
        section_path=section_heading,
        linked_images=list(images),
        contains_table=meta.get("contains_table", False),
        contains_math=meta.get("contains_math", False),
        raw_table_markup=meta.get("raw_table_markup"),
        raw_math_markup=meta.get("raw_math_markup") if meta.get("raw_math_markup") else None,
    )

    chunks_list.append(
        DBChunk(chunk_id=str(uuid.uuid4()), text="\n\n".join(chunk_text), metadata=metadata)
    )


def chunk_document(document: ParsedDocument) -> list[DBChunk]:
    """
    Пайплайн нарезки документа на семантические чанки со скользящим средним.

    Args:
        document (ParsedDocument): Распарсенный документ от парсера.

    Returns:
        list[DBChunk]: Готовые чанки для загрузки в БД.
    """
    chunks: list[DBChunk] = []

    for section in document.sections:
        raw_blocks = build_sandwiches(section.paragraphs)
        if not raw_blocks:
            continue

        # 1. Разбиваем огромные блоки рекурсивным сплиттером
        blocks: list[dict[str, Any]] = []
        for block in raw_blocks:
            if count_tokens(block["text"]) > settings.CHUNK_LIMIT and not block.get("is_sandwich"):
                split_texts = split_recursively(block["text"], settings.CHUNK_LIMIT)
                for st in split_texts:
                    blocks.append({**block, "text": st})
            else:
                blocks.append(block)

        if not blocks:
            continue

        # 2. Вычисляем эмбеддинги и косинусное сходство
        embeddings = np.array([embedder.encode(b["text"], is_document=True) for b in blocks])

        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(blocks) - 1)
        ]

        # 3. EMA для семантических разрывов
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

        # 4. Собираем чанки с жестким контролем токенов
        current_chunk_text: list[str] = []
        current_tokens = 0
        current_meta: ChunkMeta = {
            "contains_table": False,
            "contains_math": False,
            "raw_table_markup": None,
            "raw_math_markup": [],
        }
        linked_images: set[str] = set()

        for i, block in enumerate(blocks):
            block_tokens = count_tokens(block["text"])

            # Триггеры для принудительного разрыва ДО добавления текущего блока
            token_overflow = (
                current_tokens + block_tokens > settings.CHUNK_LIMIT
            ) and current_chunk_text
            table_collision = current_meta["contains_table"] and block.get("contains_table")

            if token_overflow or table_collision:
                _save_chunk(
                    document.doi,
                    section.heading,
                    current_chunk_text,
                    current_meta,
                    linked_images,
                    chunks,
                )
                current_chunk_text = []
                current_tokens = 0
                current_meta = {
                    "contains_table": False,
                    "contains_math": False,
                    "raw_table_markup": None,
                    "raw_math_markup": [],
                }
                linked_images = set()

            # Добавляем данные блока в собираемый чанк
            current_chunk_text.append(block["text"])
            current_tokens += block_tokens

            # Метаданные
            if block.get("contains_table"):
                current_meta["contains_table"] = True
                current_meta["raw_table_markup"] = block.get("raw_table_markup")

            math_markup = block.get("raw_math_markup")
            if block.get("contains_math") and isinstance(math_markup, list):
                current_meta["contains_math"] = True
                current_meta["raw_math_markup"].extend(math_markup)

            linked_images.update(extract_validated_visuals(block["text"], document.visuals))

            # Семантический разрыв ПОСЛЕ добавления блока (по скользящему среднему)
            if i in cut_indices and current_chunk_text and i < len(blocks) - 1:
                _save_chunk(
                    document.doi,
                    section.heading,
                    current_chunk_text,
                    current_meta,
                    linked_images,
                    chunks,
                )
                current_chunk_text = []
                current_tokens = 0
                current_meta = {
                    "contains_table": False,
                    "contains_math": False,
                    "raw_table_markup": None,
                    "raw_math_markup": [],
                }
                linked_images = set()

        # Сохраняем хвост
        if current_chunk_text:
            _save_chunk(
                document.doi,
                section.heading,
                current_chunk_text,
                current_meta,
                linked_images,
                chunks,
            )

    return chunks
