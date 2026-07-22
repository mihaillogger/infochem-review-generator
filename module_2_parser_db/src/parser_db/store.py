"""
Модуль для работы с векторной базой данных Qdrant.
"""

from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from parser_db.config import settings
from parser_db.embedder import NomicEmbedder
from parser_db.schemas import DBChunk


class QdrantStore:
    """
    Класс-интерфейс для взаимодействия с Qdrant.

    Инкапсулирует логику гибридного поиска и хранения семантических чанков.
    """

    def __init__(
        self,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        collection_name: str = "infochem_docs",
    ):
        """
        Инициализирует подключение к БД и загружает модели.

        Args:
            host: Хост Qdrant.
            port: REST API порт Qdrant.
            collection_name: Название коллекции для текущего пайплайна.
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        self.dense_embedder = NomicEmbedder()
        self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Проверяет наличие коллекции и создает ее при необходимости с индексами."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT32,
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

            # Индексы для точного совпадения и булевых флагов
            self.client.create_payload_index(
                self.collection_name, "doi", models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                self.collection_name, "contains_table", models.PayloadSchemaType.BOOL
            )
            self.client.create_payload_index(
                self.collection_name, "contains_math", models.PayloadSchemaType.BOOL
            )
            self.client.create_payload_index(
                self.collection_name, "has_broken_table", models.PayloadSchemaType.BOOL
            )
            self.client.create_payload_index(
                self.collection_name, "has_broken_math", models.PayloadSchemaType.BOOL
            )

            # Текстовый индекс для разделов
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_path",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )

    def insert_chunks(self, chunks: list[DBChunk]) -> None:
        """
        Векторизует и загружает список чанков в базу данных.

        Args:
            chunks: Список сформированных объектов DBChunk.
        """
        if not chunks:
            return

        points = []
        texts = [chunk.text for chunk in chunks]

        sparse_embeddings = list(self.sparse_embedder.embed(texts))

        for idx, chunk in enumerate(chunks):
            dense_vector = self.dense_embedder.encode(chunk.text, is_document=True).tolist()
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

        self.client.upsert(collection_name=self.collection_name, points=points)

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        doi_filter: str | None = None,
        section_filter: str | None = None,
        require_table: bool = False,
        require_math: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Выполняет гибридный поиск по базе с использованием RRF.

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
        dense_query = self.dense_embedder.encode(query, is_document=False).tolist()
        sparse_query = list(self.sparse_embedder.query_embed(query))[0]

        # Формируем фильтры (Payload Filters)
        must_conditions: list[models.Condition] = []
        if doi_filter:
            must_conditions.append(
                models.FieldCondition(key="doi", match=models.MatchValue(value=doi_filter))
            )
        if section_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="section_path", match=models.MatchText(text=section_filter)
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

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense_vector",
                    limit=limit * 2,
                    filter=q_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse_vector",
                    limit=limit * 2,
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
                    "chunk_id": point.id,
                    "text": payload.get("text", ""),
                    "doi": payload.get("doi", ""),
                    "section": payload.get("section_path", ""),
                    "raw_table_markup": payload.get("raw_table_markup"),
                    "raw_math_markup": payload.get("raw_math_markup"),
                    "linked_images": payload.get("linked_images"),
                    "has_broken_table": payload.get("has_broken_table", False),
                    "has_broken_math": payload.get("has_broken_math", False),
                    "fallback_table_paths": payload.get("fallback_table_paths", []),
                    "fallback_math_paths": payload.get("fallback_math_paths", []),
                }
            )

        return formatted_results
