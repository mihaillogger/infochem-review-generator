"""Поиск статей в CrossRef по ключевым словам (Модуль 1, задача T3).

Вход:  список ключевых слов.
Выход: метаданные найденных статей в формате контракта metadata.json
       (DOI -> pdf_path + title + authors + year + journal + статус).

Сколько статей брать на тему — не хардкод, а ПЛАВАЮЩИЙ N: score у CrossRef
(релевантность BM25) не имеет абсолютной шкалы и сравним только внутри одного
запроса, поэтому режем относительно «шумового пола» самого запроса —
статьи, которые поднялись над фоном мусора хотя бы на BETA от размаха.
"""

from __future__ import annotations

import statistics
from typing import Any, cast

import httpx

CROSSREF_URL = "https://api.crossref.org/works"

# --- Параметры отбора (кандидаты в конфиг Модуля, задача T6) ---
POOL = 60  # сколько кандидатов тянем, чтобы оценить шумовой пол
BETA = 0.5  # порог = floor + BETA*(max-floor): «выше середины над шумом»
MIN_N = 5  # не меньше — иначе тема почти пустая
MAX_N = 40  # не больше — дальше почти наверняка шум
TAIL = 20  # сколько статей с конца считаем шумовым полом
TIMEOUT = 20.0


def _build_filters(from_year: int, until_year: int) -> str:
    """Строка filter: только журнальные статьи в заданном диапазоне лет."""
    return ",".join(
        [
            "type:journal-article",
            f"from-pub-date:{from_year}-01-01",
            f"until-pub-date:{until_year}-12-31",
        ]
    )


def _search_pool(
    client: httpx.Client, keyword: str, *, from_year: int, until_year: int
) -> list[dict[str, Any]]:
    """Возвращает пул кандидатов (works), отсортированный по score убыванию.

    Явный sort=score/order=desc — не полагаемся на дефолтный порядок API.
    Плюс подстраховываемся локальной сортировкой: весь отбор ниже опирается
    на то, что pool[0] — самый релевантный, а pool[:n] — топ.
    """
    params: dict[str, str | int] = {
        "query.bibliographic": keyword,
        "filter": _build_filters(from_year, until_year),
        "select": "DOI,title,author,issued,container-title,score",
        "sort": "score",
        "order": "desc",
        "rows": POOL,
    }
    resp = client.get(CROSSREF_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    items = cast(list[dict[str, Any]], payload["message"]["items"])
    items.sort(key=lambda w: w.get("score", 0.0), reverse=True)
    return items


def _adaptive_cutoff(scores: list[float]) -> int:
    """Сколько статей взять: относительный порог от шумового пола, зажат в [MIN_N, MAX_N].

    floor  = медиана хвоста (устойчива к выбросам, работает на дробях);
    порог  = floor + BETA*(max-floor);
    N      = сколько score >= порога.
    """
    if not scores:
        return 0
    max_score = scores[0]
    floor = statistics.median(scores[-TAIL:]) if len(scores) >= TAIL else scores[-1]
    threshold = floor + BETA * (max_score - floor)
    n = sum(1 for s in scores if s >= threshold)
    return max(MIN_N, min(n, MAX_N))


def _extract_authors(work: dict[str, Any]) -> list[str]:
    """author[] CrossRef ({given, family}) -> список строк 'Given Family'."""
    names: list[str] = []
    for a in work.get("author", []):
        full = f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
        if full:
            names.append(full)
    return names


def _extract_year(work: dict[str, Any]) -> int | None:
    """issued.date-parts -> [[YYYY, MM, DD]]; берём год."""
    parts = work.get("issued", {}).get("date-parts", [[None]])
    return parts[0][0] if parts and parts[0] else None


def _to_metadata(work: dict[str, Any], keyword: str) -> dict[str, Any]:
    """Одна работа CrossRef -> запись контракта metadata.json."""
    return {
        "pdf_path": None,  # заполнит качалка (T4) после скачивания
        "title": (work.get("title") or [None])[0],
        "authors": _extract_authors(work),
        "year": _extract_year(work),
        "journal": (work.get("container-title") or [None])[0],
        "matched_keywords": [keyword],  # по каким темам статья найдена
        "download_status": "pending",  # pending -> ok / not_found / error
        "source": None,  # чем скачано (заполнит качалка)
    }


def search(
    keywords: list[str],
    *,
    from_year: int = 2015,
    until_year: int = 2024,
    mailto: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Ищет статьи по списку ключевых слов, возвращает метаданные с дедупом по DOI.

    Args:
        keywords: ключевые слова/темы.
        from_year, until_year: границы по году публикации.
        mailto: e-mail для polite pool CrossRef (стабильнее лимиты).

    Returns:
        Маппинг DOI -> запись метаданных (контракт metadata.json).
    """
    user_agent = "infochem-review-generator/0.1 (module_1_downloader)"
    if mailto:
        user_agent += f" mailto:{mailto}"

    manifest: dict[str, dict[str, Any]] = {}
    with httpx.Client(headers={"User-Agent": user_agent}) as client:
        for keyword in keywords:
            pool = _search_pool(client, keyword, from_year=from_year, until_year=until_year)
            n = _adaptive_cutoff([w.get("score", 0.0) for w in pool])
            for work in pool[:n]:
                doi = work.get("DOI")
                if not doi:
                    continue
                if doi in manifest:
                    # Статья уже найдена по другой теме — просто помечаем.
                    manifest[doi]["matched_keywords"].append(keyword)
                    continue
                manifest[doi] = _to_metadata(work, keyword)
    return manifest


if __name__ == "__main__":
    import json

    result = search(["oxidation induction time", "antioxidants"])
    print(f"Всего уникальных статей: {len(result)}\n")
    print(json.dumps(dict(list(result.items())[:3]), ensure_ascii=False, indent=2))
