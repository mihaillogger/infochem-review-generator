"""Шаг 6: ПЛАВАЮЩИЙ N — сколько статей брать, подбирается автоматически.

Идея: score у CrossRef не сравним между запросами (у одной темы топ=17,
у другой=40), но ОТНОСИТЕЛЬНО своего максимума — сравним. Поэтому режем не
по числу, а по доле от лучшего score. N выходит разный для разных тем.

Два сигнала:
  1) относительный порог: берём всё, у кого score >= alpha * max_score;
  2) детектор обрыва: самый большой разрыв между соседними score в топе.
Финальный N — порог, зажатый в [MIN_N, MAX_N].
"""

import httpx

BASE_URL = "https://api.crossref.org/works"

ALPHA = 0.80     # порог: не хуже 80% от лучшего score
MIN_N = 5        # меньше брать бессмысленно
MAX_N = 40       # больше — почти наверняка шум
POOL = 50        # сколько кандидатов тянем, чтобы было из чего резать


def fetch_scores(keyword: str) -> list[float]:
    params = {
        "query.bibliographic": keyword,
        "filter": "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31",
        "select": "DOI,title,score",
        "rows": POOL,
    }
    items = httpx.get(BASE_URL, params=params, timeout=20.0).json()["message"]["items"]
    return [w["score"] for w in items]


def adaptive_n(scores: list[float]) -> int:
    """Сколько статей взять по относительному порогу, зажатое в [MIN_N, MAX_N]."""
    if not scores:
        return 0
    max_score = scores[0]
    cutoff = ALPHA * max_score
    n = sum(1 for s in scores if s >= cutoff)
    return max(MIN_N, min(n, MAX_N))


def biggest_gap(scores: list[float]) -> int:
    """Позиция самого большого относительного обрыва в топе (для сравнения)."""
    if len(scores) < 2:
        return len(scores)
    best_pos, best_drop = len(scores), 0.0
    # Ищем обрыв только в разумной зоне (не в самом хвосте).
    for i in range(1, min(len(scores), MAX_N)):
        drop = (scores[i - 1] - scores[i]) / scores[0]
        if drop > best_drop:
            best_drop, best_pos = drop, i
    return best_pos


KEYWORDS = [
    "oxidation induction time",
    "antioxidants",
    "polymer thermal stabilization",
]

print(f"{'ключевое слово':<34}  {'max':>5}  {'N (порог)':>9}  {'N (обрыв)':>9}")
print("-" * 66)
for kw in KEYWORDS:
    scores = fetch_scores(kw)
    n_thr = adaptive_n(scores)
    n_gap = biggest_gap(scores)
    top = scores[0] if scores else 0
    print(f"{kw:<34}  {top:>5.1f}  {n_thr:>9}  {n_gap:>9}")
