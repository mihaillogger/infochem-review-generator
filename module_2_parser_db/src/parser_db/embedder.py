import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from parser_db.config import settings


class NomicEmbedder:
    _instance = None
    model: SentenceTransformer

    def __new__(cls) -> "NomicEmbedder":
        # Паттерн Синглтон, чтобы модель грузилась в память только один раз
        # при старте воркера TaskIQ, а не при каждом запросе
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # trust_remote_code=True обязательно для Nomic
            cls._instance.model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
                device="cpu",  # На Фазе 2 поменяем на "cuda"
            )
        return cls._instance

    def encode(self, text: str, is_document: bool = True) -> np.ndarray:
        """
        Векторизует текст.
        Nomic требует строгие префиксы для разделения документов и запросов.
        """
        prefix = "search_document: " if is_document else "search_query: "
        full_text = prefix + text

        # Получаем эмбеддинг
        embeddings = self.model.encode([full_text], convert_to_tensor=True)

        # Обязательная L2-нормализация (требование архитектуры Nomic)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Возвращаем плоский список float
        return embeddings[0].cpu().numpy()
