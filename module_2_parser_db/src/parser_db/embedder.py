"""Модуль генерации эмбеддингов с использованием моделей SentenceTransformers."""

import numpy as np
import structlog
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from parser_db.config import settings
from parser_db.profiler import profile_time


logger = structlog.get_logger(__name__)


class NomicEmbedder:
    _instance = None
    model: SentenceTransformer

    def __new__(cls) -> "NomicEmbedder":
        # Паттерн Синглтон, чтобы модель грузилась в память только один раз
        # при старте воркера TaskIQ, а не при каждом запросе
        if cls._instance is None:
            logger.info("embedder_init_start", model_name=settings.EMBEDDING_MODEL_NAME)

            cls._instance = super().__new__(cls)
            # trust_remote_code=True обязательно для Nomic
            cls._instance.model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
                device="cpu",  # На Фазе 2 поменяем на "cuda"
            )

            logger.info("embedder_init_success", device="cpu")

        return cls._instance

    @profile_time
    def encode_batch(self, texts: list[str], is_document: bool = True) -> np.ndarray:
        """
        Векторизует батч.
        Nomic требует строгие префиксы для разделения документов и запросов.
        """
        if not texts:
            return np.array([])

        logger.debug("encoding_batch", batch_size=len(texts), is_document=is_document)

        prefix = "search_document: " if is_document else "search_query: "
        full_texts = [prefix + t for t in texts]

        # Получаем эмбеддинги
        embeddings = self.model.encode(full_texts, convert_to_tensor=True)

        # Обязательная L2-нормализация (требование архитектуры Nomic)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
