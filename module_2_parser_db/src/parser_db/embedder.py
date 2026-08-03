"""Модуль генерации эмбеддингов с использованием моделей SentenceTransformers."""

import threading

import numpy as np
import structlog
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from parser_db.config import settings
from parser_db.profiler import profile_time

logger = structlog.get_logger(__name__)


class NomicEmbedder:
    _instance = None
    _lock = threading.Lock()
    model: SentenceTransformer

    def __new__(cls) -> "NomicEmbedder":
        """
        Создает и возвращает единственный экземпляр эмбеддера (Singleton).

        Returns:
            NomicEmbedder: Экземпляр класса NomicEmbedder.
        """
        with cls._lock:
            if cls._instance is None:
                logger.info("embedder_init_start", model_name=settings.EMBEDDING_MODEL_NAME)

                cls._instance = super().__new__(cls)
                # trust_remote_code=True обязательно для Nomic
                cls._instance.model = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    trust_remote_code=True,
                    device=settings.COMPUTE_DEVICE,
                )

                logger.info("embedder_init_success", device="cpu")

        return cls._instance

    @profile_time
    def encode_batch(
        self, texts: list[str], is_document: bool = True, batch_size: int | None = None
    ) -> np.ndarray:
        """
        Векторизует батч.
        Nomic требует строгие префиксы для разделения документов и запросов.

        Args:
            texts (list[str]): Список строк для векторизации.
            is_document (bool): Если True, используется префикс для документов,
                иначе для запросов. По умолчанию True.
            batch_size (int | None): Размер батча. Если None, используется
                значение из конфигурации. По умолчанию None.

        Returns:
            np.ndarray: L2-нормализованные эмбеддинги.
        """
        if not texts:
            return np.array([])

        actual_batch_size = batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE

        # Если установлен флаг 0, отключаем нарезку на батчи (передаем весь массив)
        if actual_batch_size == 0:
            actual_batch_size = len(texts)

        logger.debug(
            "encoding_batch",
            total_texts=len(texts),
            actual_batch_size=batch_size,
            is_document=is_document,
        )

        prefix = settings.EMBEDDING_PREFIX_DOC if is_document else settings.EMBEDDING_PREFIX_QUERY
        full_texts = [prefix + t for t in texts]

        embeddings = self.model.encode(
            full_texts, convert_to_tensor=True, batch_size=actual_batch_size
        )

        # Обязательная L2-нормализация (требование архитектуры Nomic)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
