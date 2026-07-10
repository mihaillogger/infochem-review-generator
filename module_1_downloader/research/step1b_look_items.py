"""Шаг 1b: смотрим СОДЕРЖИМОЕ найденных работ — type / title / score.

Цель — увидеть своими глазами, что без фильтра в выдачу лезет мусор
(стандарты, книги и т.п.), а не только journal-article.
"""

import httpx

BASE_URL = "https://api.crossref.org/works"
params = {"query.bibliographic": "oxidation induction time", "rows": 5}

data = httpx.get(BASE_URL, params=params, timeout=20.0).json()
items = data["message"]["items"]

for i, work in enumerate(items):
    # title в CrossRef — это ВСЕГДА список строк (обычно из одного элемента),
    # поэтому берём первый элемент, иначе получим ['...'] в скобках.
    title_list = work.get("title") or ["<без названия>"]
    title = title_list[0]
    print(f"[{i}] type={work.get('type'):<20} score={work.get('score'):.1f}")
    print(f"    DOI:   {work.get('DOI')}")
    print(f"    title: {title[:80]}")
    print()
