"""Шаг 1 T1: один запрос к CrossRef по ключевому слову.

Задача этого шага — НЕ написать модуль, а понять:
  1) как выглядит URL поиска по словам (не по DOI!);
  2) во что завёрнут ответ (envelope) и где лежат сами статьи.
"""

import httpx

# Базовый эндпоинт REST API CrossRef. Тот же, что и для лукапа по DOI,
# но БЕЗ /{doi} на конце — здесь мы ищем, а не достаём конкретную работу.
BASE_URL = "https://api.crossref.org/works"

# query.bibliographic — «умный» поиск по библиографическим полям
# (title + авторы + журнал). Для поиска статей по теме это то, что нужно.
# rows=3 — просим всего 3 результата, чтобы глазами разобрать ответ.
params = {
    "query.bibliographic": "oxidation induction time",
    "rows": 3,
}

response = httpx.get(BASE_URL, params=params, timeout=20.0)
response.raise_for_status()
data = response.json()

# --- 1. Что на верхнем уровне (envelope) ---
print("Ключи верхнего уровня:", list(data.keys()))
print("status:", data.get("status"))
print("message-type:", data.get("message-type"))

# Всё полезное — внутри "message".
message = data["message"]
print("\nКлючи внутри message:", list(message.keys()))
print("total-results (сколько всего нашлось):", message.get("total-results"))
print("items (сколько вернулось сейчас):", len(message.get("items", [])))

# --- 2. Смотрим на одну статью ---
first = message["items"][0]
print("\nПоля одной статьи (items[0]):")
for key in sorted(first.keys()):
    print("   ", key)
