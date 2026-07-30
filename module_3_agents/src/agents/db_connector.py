import requests
import config

def search_chunks(query: str, limit: int = 3) -> list[str]:
    payload = {
        "query": query,
        "limit": limit
    }
    
    try:
        response = requests.post(config.FASTAPI_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        
        # По контракту API возвращает список словарей: [{"text": "...", "doi": "...", ...}]
        results = response.json()
        
        chunks = [item.get("text", "") for item in results if item.get("text")]
        
        if not chunks:
            print(f"[WARNING] База вернула пустой список для запроса: '{query}'")
            
        return chunks

    except requests.exceptions.Timeout:
        print(f"[ERROR] Сервер БД не ответил за 10 секунд (Timeout). Запрос: '{query}'")
        return []
        
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Не удалось подключиться к базе по адресу: {config.FASTAPI_ENDPOINT}. Сервер запущен?")
        return []
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Сервер БД вернул ошибку ({response.status_code}): {e}")
        return []
        
    except Exception as e:
        print(f"[ERROR] Неизвестная ошибка при запросе к БД: {e}")
        return []