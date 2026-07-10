"""Шаг 4: как понять, сколько статей брать — смотрим на падение score.

CrossRef сортирует по релевантности (score). У топовых он высокий, дальше
плавно падает. Печатаем score топ-30, чтобы глазами увидеть «обрыв» —
после него статьи уже слабо относятся к теме.
"""

import httpx

BASE_URL = "https://api.crossref.org/works"
params = {
    "query.bibliographic": "oxidation induction time",
    "filter": "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31",
    "select": "DOI,title,score",
    "rows": 30,
}

items = httpx.get(BASE_URL, params=params, timeout=20.0).json()["message"]["items"]

max_score = items[0]["score"] if items else 0
print(f"{'#':>2}  {'score':>7}  {'% от макс':>9}  title")
print("-" * 80)
for i, w in enumerate(items, 1):
    score = w["score"]
    pct = (score / max_score * 100) if max_score else 0
    title = (w.get("title") or ["—"])[0]
    # Простая визуализация «обрыва»: длина бара пропорциональна % от макс.
    bar = "█" * int(pct / 5)
    print(f"{i:>2}  {score:>7.1f}  {pct:>7.0f}%  {bar} {title[:45]}")
