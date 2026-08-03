"""
Модуль для работы с векторной базой данных Qdrant.
"""

import asyncio
from typing import Any

import structlog
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from parser_db.config import settings
from parser_db.embedder import NomicEmbedder
from parser_db.profiler import profile_time
from parser_db.schemas import DBChunk

logger = structlog.get_logger(__name__)


class AsyncQdrantStore:
    """
    Асинхронный класс-интерфейс для взаимодействия с Qdrant.

    Инкапсулирует логику гибридного поиска и хранения семантических чанков.
    """

    def __init__(self) -> None:
        """Инициализирует подключение к БД и загружает модели."""
        self.client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.COLLECTION_NAME

        self.dense_embedder = NomicEmbedder()
        self.sparse_embedder = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)

    async def ensure_collection_exists(self) -> None:
        """Асинхронно проверяет наличие коллекции и создает ее при необходимости с индексами."""
        if not await self.client.collection_exists(self.collection_name):
            logger.info("creating_qdrant_collection", collection_name=self.collection_name)

            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclid": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
                "Manhattan": models.Distance.MANHATTAN,
            }
            datatype_map = {
                "float32": models.Datatype.FLOAT32,
                "float16": models.Datatype.FLOAT16,
                "uint8": models.Datatype.UINT8,
            }

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=distance_map.get(settings.DENSE_DISTANCE, models.Distance.COSINE),
                        datatype=datatype_map.get(settings.DENSE_DATATYPE, models.Datatype.FLOAT32),
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=settings.SPARSE_ON_DISK)
                    )
                },
            )

            # Индексы для точного совпадения и булевых флагов
            await self.client.create_payload_index(
                self.collection_name, "doi", models.PayloadSchemaType.KEYWORD
            )
            await self.client.create_payload_index(
                self.collection_name, "contains_table", models.PayloadSchemaType.BOOL
            )
            await self.client.create_payload_index(
                self.collection_name, "contains_math", models.PayloadSchemaType.BOOL
            )
            await self.client.create_payload_index(
                self.collection_name, "has_broken_table", models.PayloadSchemaType.BOOL
            )
            await self.client.create_payload_index(
                self.collection_name, "has_broken_math", models.PayloadSchemaType.BOOL
            )
            await self.client.create_payload_index(
                self.collection_name, "has_broken_text", models.PayloadSchemaType.BOOL
            )

            # Текстовый индекс для разделов
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="macro_category_path",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=settings.TEXT_INDEX_MIN_LEN,
                    max_token_len=settings.TEXT_INDEX_MAX_LEN,
                    lowercase=settings.TEXT_INDEX_LOWERCASE,
                ),
            )

    async def close(self) -> None:
        """Асинхронно закрывает пулы сетевых соединений клиента Qdrant."""
        logger.info("closing_qdrant_connection", collection=self.collection_name)
        await self.client.close()

    @profile_time
    async def insert_chunks(self, chunks: list[DBChunk]) -> None:
        """
        Асинхронно векторизует и загружает список чанков в базу данных.
        Перед загрузкой удаляет старые чанки с таким же DOI.

        Args:
            chunks: Список сформированных объектов DBChunk.
        """
        if not chunks:
            return

        # Удаляем старые чанки для этих статей
        unique_dois = {chunk.metadata.doi for chunk in chunks if chunk.metadata.doi}
        if unique_dois:
            logger.info("deleting_old_chunks", dois=list(unique_dois))

        for doi in unique_dois:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="doi", match=models.MatchValue(value=doi))]
                ),
            )

        points = []
        texts = [chunk.text for chunk in chunks]

        dense_embeddings = await asyncio.to_thread(
            self.dense_embedder.encode_batch, texts, is_document=True
        )
        sparse_embeddings = await asyncio.to_thread(list, self.sparse_embedder.embed(texts))

        for idx, chunk in enumerate(chunks):
            dense_vector = dense_embeddings[idx]
            sparse_vector = sparse_embeddings[idx]

            point = models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense_vector": dense_vector.tolist(),
                    "sparse_vector": models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                },
                payload={"text": chunk.text, **chunk.metadata.model_dump()},
            )
            points.append(point)

        await self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("chunks_inserted", count=len(points), collection=self.collection_name)

    @profile_time
    async def hybrid_search(
        self,
        query: str,
        limit: int = settings.SEARCH_DEFAULT_LIMIT,
        doi_filter: str | None = None,
        section_filter: str | None = None,
        require_table: bool = False,
        require_math: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Асинхронно выполняет гибридный поиск по базе с использованием RRF.

        Args:
            query: Текст запроса от LLM-агента.
            limit: Максимальное количество возвращаемых чанков.
            doi_filter: Ограничить поиск конкретной статьей.
            section_filter: Искать только в определенном разделе (например, "Methods").
            require_table: Искать только в чанках с таблицами.
            require_math: Искать только в чанках с формулами.

        Returns:
            Список словарей с результатами поиска.
        """
        dense_ndarray = await asyncio.to_thread(
            self.dense_embedder.encode_batch, [query], is_document=False
        )
        dense_query = dense_ndarray[0].tolist()

        sparse_query = await asyncio.to_thread(
            lambda q: list(self.sparse_embedder.query_embed(q))[0], query
        )

        # Формируем фильтры (Payload Filters)
        must_conditions: list[models.Condition] = []
        if doi_filter:
            must_conditions.append(
                models.FieldCondition(key="doi", match=models.MatchValue(value=doi_filter))
            )
        if section_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="macro_category_path", match=models.MatchText(text=section_filter)
                )
            )
        if require_table:
            must_conditions.append(
                models.FieldCondition(key="contains_table", match=models.MatchValue(value=True))
            )
        if require_math:
            must_conditions.append(
                models.FieldCondition(key="contains_math", match=models.MatchValue(value=True))
            )

        q_filter = models.Filter(must=must_conditions) if must_conditions else None

        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense_vector",
                    limit=limit * settings.SEARCH_PREFETCH_MULTIPLIER,
                    filter=q_filter,
                    score_threshold=settings.QDRANT_BASE_THRESHOLD,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse_vector",
                    limit=limit * settings.SEARCH_PREFETCH_MULTIPLIER,
                    filter=q_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for point in results.points:
            payload = point.payload or {}
            formatted_results.append(
                {
                    "chunk_id": str(point.id),
                    "text": payload.get("text", ""),
                    "metadata": {
                        "doi": payload.get("doi", ""),
                        "title": payload.get("title"),
                        "authors": payload.get("authors", []),
                        "year": payload.get("year"),
                        "journal": payload.get("journal"),
                        "abstract": payload.get("abstract"),
                        "original_heading_path": payload.get("original_heading_path", ""),
                        "macro_category_path": payload.get("macro_category_path", ""),
                        "linked_images": payload.get("linked_images", {}),
                        "contains_table": payload.get("contains_table", False),
                        "contains_math": payload.get("contains_math", False),
                        "raw_table_markup": payload.get("raw_table_markup"),
                        "raw_math_markup": payload.get("raw_math_markup"),
                        "has_broken_table": payload.get("has_broken_table", False),
                        "has_broken_math": payload.get("has_broken_math", False),
                        "has_broken_text": payload.get("has_broken_text", False),
                        "fallback_table_paths": payload.get("fallback_table_paths", []),
                        "fallback_math_paths": payload.get("fallback_math_paths", []),
                    },
                }
            )

        logger.info(
            "hybrid_search_executed",
            query=query,
            limit=limit,
            doi_filter=doi_filter,
            section_filter=section_filter,
            require_table=require_table,
            require_math=require_math,
            results=len(formatted_results),
        )
        return formatted_results


_store_instance = None
_store_lock = asyncio.Lock()


async def get_store() -> AsyncQdrantStore:
    """
    Возвращает единственный экземпляр базы данных (Singleton).

    Используется для прогрева при старте сервера и как провайдер
    зависимостей (Dependency) для эндпоинтов FastAPI.

    Returns:
        AsyncQdrantStore: Инициализированный клиент векторной базы.
    """
    global _store_instance

    if _store_instance is None:
        async with _store_lock:
            # Double-checked locking для безопасности при конкурентном доступе
            if _store_instance is None:
                store = AsyncQdrantStore()
                await store.ensure_collection_exists()
                _store_instance = store

    return _store_instance
