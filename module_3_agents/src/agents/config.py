import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() будет сканировать папки вверх, пока не найдет .env
load_dotenv(find_dotenv())

# Достаем ключ
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: GOOGLE_API_KEY не найден. Проверь файл .env")