"""Шаг 5: как меняется выдача при разном rows (5..30, шаг 5).

Для каждого rows делаем отдельный запрос и смотрим:
  - сколько статей реально вернулось;
  - score последней (самой слабой) статьи в выборке;
  - грубую оценку «сколько по теме» — эвристика по словам в заголовке.
Так видно, что точность (доля релевантных) падает с ростом N.
"""

import httpx

BASE_URL = "https://api.crossref.org/works"

# Слова-маркеры «это про нашу химию», а не про математику/медицину,
# которые лезут за общие слова induction/time.
TOPIC_WORDS = (
    "oxidation", "oxid", "antioxid", "polymer", "polyolefin", "polyethylene",
    "thermal", "dsc", "stabil", "oil", "lubric", "degrad", "filler",
)


def is_on_topic(title: str) -> bool:
    low = title.lower()
    return any(w in low for w in TOPIC_WORDS)


base_params = {
    "query.bibliographic": "oxidation induction time",
    "filter": "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31",
    "select": "DOI,title,score",
}

print(f"{'rows':>5}  {'вернулось':>9}  {'score послед.':>13}  {'по теме':>8}  {'точность':>8}")
print("-" * 60)

for rows in range(5, 31, 5):
    params = {**base_params, "rows": rows}
    items = httpx.get(BASE_URL, params=params, timeout=20.0).json()["message"]["items"]

    on_topic = sum(is_on_topic((w.get("title") or [""])[0]) for w in items)
    last_score = items[-1]["score"] if items else 0
    precision = on_topic / len(items) * 100 if items else 0

    print(
        f"{rows:>5}  {len(items):>9}  {last_score:>13.1f}  "
        f"{on_topic:>8}  {precision:>7.0f}%"
    )
