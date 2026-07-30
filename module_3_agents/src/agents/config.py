import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

<<<<<<< HEAD

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: GOOGLE_API_KEY не найден. Проверь файл .env")

FASTAPI_ENDPOINT = "http://localhost:8000/search"

LLM_MODEL = "gemini-3.5-flash-lite"
=======
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY не найден. Проверьте файл .env")
>>>>>>> f68b5f5529685c707b09ec080e7ebecb83d9d5e2
