"""Шаг 7: ОТКУДА берётся порог — из «шумового пола» самого запроса.

Проблема прошлого шага: α=0.8 было взято с потолка. Здесь порог считается
из данных: у каждого запроса есть фоновый уровень мусора (плато низких score,
куда сваливаются статьи, совпавшие за общие слова). Релевантные торчат над ним.

Критерий:
    floor = медиана score хвоста пула   (уровень шума)
    max   = score лучшей статьи
    порог = floor + BETA * (max - floor) (насколько нужно подняться над полом)
    N     = сколько статей выше порога
BETA заменяет α, но теперь он привязан к РАЗМАХУ конкретного запроса,
а не к абсолютному максимуму. Его тоже можно калибровать, но он устойчивее.
"""

import statistics

import httpx

BASE_URL = "https://api.crossref.org/works"
POOL = 60
BETA = 0.5       # брать тех, кто поднялся выше середины диапазона [floor..max]
TAIL = 20        # сколько статей с конца считаем «шумовым полом»


def fetch_scores(keyword: str) -> list[float]:
    params = {
        "query.bibliographic": keyword,
        "filter": "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31",
        "select": "DOI,score",
        "rows": POOL,
    }
    items = httpx.get(BASE_URL, params=params, timeout=20.0).json()["message"]["items"]
    return [w["score"] for w in items]


def adaptive_n(scores: list[float]) -> tuple[int, float, float, float]:
    max_score = scores[0]
    floor = statistics.median(scores[-TAIL:])   # уровень шума из хвоста
    threshold = floor + BETA * (max_score - floor)
    n = sum(1 for s in scores if s >= threshold)
    return n, max_score, floor, threshold


KEYWORDS = [
    "oxidation induction time",
    "antioxidants",
    "polymer thermal stabilization",
]

print(f"{'ключевик':<32} {'max':>5} {'floor':>6} {'порог':>6} {'размах':>7} {'N':>4}")
print("-" * 68)
for kw in KEYWORDS:
    scores = fetch_scores(kw)
    n, mx, fl, thr = adaptive_n(scores)
    print(f"{kw:<32} {mx:>5.1f} {fl:>6.1f} {thr:>6.1f} {mx - fl:>7.1f} {n:>4}")
