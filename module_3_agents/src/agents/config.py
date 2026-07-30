import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: GOOGLE_API_KEY не найден. Проверьте файл .env")

FASTAPI_ENDPOINT = "http://localhost:8000/search"

LLM_MODEL = "gemini-3.5-flash-lite"
