"""Шаг 2: фильтры (type/даты) + select (только нужные поля).

Что разбираем:
  - filter=  — несколько условий через ЗАПЯТУЮ, каждое как ключ:значение;
  - select=  — какие поля вернуть (через запятую), чтобы ответ был лёгким.
"""

import httpx

BASE_URL = "https://api.crossref.org/works"

# filter — это ОДНА строка "k1:v1,k2:v2,...". httpx сам её не соберёт из dict,
# поэтому склеиваем руками. Так же и select — список полей через запятую.
filters = ",".join(
    [
        "type:journal-article",       # только журнальные статьи
        "from-pub-date:2015-01-01",   # нижняя граница по дате публикации
        "until-pub-date:2024-12-31",  # верхняя граница (Sci-Hub ~до 2024)
    ]
)

params = {
    "query.bibliographic": "oxidation induction time",
    "filter": filters,
    "select": "DOI,title,type,container-title,issued,author",
    "rows": 5,
}

data = httpx.get(BASE_URL, params=params, timeout=20.0).json()
message = data["message"]

print("total-results теперь:", message["total-results"])
print()

for i, work in enumerate(message["items"]):
    title = (work.get("title") or ["<без названия>"])[0]
    journal = (work.get("container-title") or ["<без журнала>"])[0]
    # issued.date-parts — вложенный список [[YYYY, MM, DD]]; год — первый элемент.
    date_parts = work.get("issued", {}).get("date-parts", [[None]])
    year = date_parts[0][0]
    print(f"[{i}] type={work.get('type')}  year={year}")
    print(f"    DOI:     {work.get('DOI')}")
    print(f"    journal: {journal}")
    print(f"    title:   {title[:75]}")
    print()
