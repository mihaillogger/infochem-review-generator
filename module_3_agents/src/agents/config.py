import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: GOOGLE_API_KEY не найден. Проверьте файл .env")

CONNECT_API_KEY = os.getenv("CONNECT_API_KEY")

if not CONNECT_API_KEY:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: INFOCHEM_API_KEY не найден. Проверьте файл .env")

DB_API_URL = os.getenv("DB_API_URL")

if not DB_API_URL:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: DB_API_URL не найден. Проверьте файл .env")

LLM_MODEL = "gemini-3.5-flash-lite"

MAX_RETRIES = 9
