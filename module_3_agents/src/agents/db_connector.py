import requests
import config
from typing import List, Dict


def fetch_chunks(query: str, target_paths: List[str] = None, limit: int = 70) -> List[Dict]:
    payload = {"query": query, "limit": limit}

    headers = {"X-API-Key": config.CONNECT_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(config.DB_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        chunks = response.json().get("data", [])

        results = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            chunk_id = chunk.get("chunk_id")
            metadata = chunk.get("metadata", {})

            results.append({"id": str(chunk_id), "text": text, "metadata": metadata})

        return results
    except Exception as e:
        print(f"[ERROR] Ошибка БД: {e}")
        return []
