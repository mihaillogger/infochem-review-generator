"""Шаг 3: из результатов поиска собираем metadata.json в формате,
который потребовал Модуль 2 (парсер + ChromaDB).

Контракт (минимум, который жёстко просят напарники):
    DOI -> { pdf_path, title, authors, year }
Плюс бесплатно кладём journal (тем же запросом приходит) — пригодится в Chroma.

pdf_path на этапе ПОИСКА ещё неизвестен (файл появится после скачивания
через Sci-Hub), поэтому ставим placeholder null — качалка (T4) впишет путь.
"""

import json

import httpx

BASE_URL = "https://api.crossref.org/works"

# В select добавили author — без него CrossRef не вернёт авторов.
params = {
    "query.bibliographic": "oxidation induction time",
    "filter": "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31",
    "select": "DOI,title,author,issued,container-title",
    "rows": 3,
}

data = httpx.get(BASE_URL, params=params, timeout=20.0).json()
items = data["message"]["items"]


def extract_authors(work: dict) -> list[str]:
    """author[] в CrossRef — список объектов {given, family}. Собираем строки."""
    names: list[str] = []
    for a in work.get("author", []):
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        full = f"{given} {family}".strip()
        if full:
            names.append(full)
    return names


def extract_year(work: dict) -> int | None:
    """issued.date-parts -> [[YYYY, MM, DD]]; год — первый элемент."""
    parts = work.get("issued", {}).get("date-parts", [[None]])
    return parts[0][0] if parts and parts[0] else None


# Ключ — DOI (как и просят: маппинг DOI -> данные).
manifest: dict[str, dict] = {}
for work in items:
    doi = work.get("DOI")
    if not doi:
        continue
    manifest[doi] = {
        "pdf_path": None,  # заполнит качалка после скачивания
        "title": (work.get("title") or [None])[0],
        "authors": extract_authors(work),
        "year": extract_year(work),
        "journal": (work.get("container-title") or [None])[0],
        "download_status": "pending",  # pending -> ok / not_found / error
    }

print(json.dumps(manifest, ensure_ascii=False, indent=2))
