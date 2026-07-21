"""Тесты поиска по CrossRef. Без сети — _search_pool мокается."""

from __future__ import annotations

from typing import Any

import pytest
from downloader import crossref_search as cs

# ── _build_filters ────────────────────────────────────────────────


def test_build_filters_shape() -> None:
    got = cs._build_filters(2015, 2024)
    assert got == "type:journal-article,from-pub-date:2015-01-01,until-pub-date:2024-12-31"


# ── _adaptive_cutoff ──────────────────────────────────────────────
# Сердце отбора: сколько статей взять на тему. Логику проверяем на
# посчитанных вручную примерах, а не на живых score.


def test_cutoff_empty_is_zero() -> None:
    assert cs._adaptive_cutoff([]) == 0


def test_cutoff_floors_to_min_n() -> None:
    # Один явный лидер над плоским шумом: порог отсечёт всё, кроме пары
    # значений, но нижняя граница MIN_N не даёт вернуть меньше пяти.
    scores = [20.0, 20.0, 20.0] + [10.0] * 17 + [1.0] * 20
    assert cs._adaptive_cutoff(scores) == cs.MIN_N


def test_cutoff_caps_to_max_n() -> None:
    # Много одинаково высоких: без верхней границы вернулось бы 45.
    scores = [100.0] * 45
    assert cs._adaptive_cutoff(scores) == cs.MAX_N


def test_cutoff_returns_value_between_bounds() -> None:
    # 10 статей над нулевым шумовым полом: floor=0, порог=50, ровно 10 >= 50.
    scores = [100.0] * 10 + [0.0] * 20
    assert cs._adaptive_cutoff(scores) == 10


def test_cutoff_short_list_uses_last_as_floor() -> None:
    # Короче TAIL: полом считается последний элемент, а не медиана хвоста.
    scores = [10.0, 9.0, 8.0, 7.0, 6.0]  # floor=6, порог=8, три >= 8 -> MIN_N
    assert cs._adaptive_cutoff(scores) == cs.MIN_N


# ── _extract_authors ──────────────────────────────────────────────


def test_authors_given_and_family() -> None:
    work = {"author": [{"given": "Ada", "family": "Lovelace"}]}
    assert cs._extract_authors(work) == ["Ada Lovelace"]


def test_authors_family_only_and_given_only() -> None:
    work = {"author": [{"family": "Curie"}, {"given": "Plato"}]}
    assert cs._extract_authors(work) == ["Curie", "Plato"]


def test_authors_skips_fully_empty() -> None:
    work = {"author": [{"given": "", "family": ""}, {"given": None, "family": None}]}
    assert cs._extract_authors(work) == []


def test_authors_missing_key() -> None:
    assert cs._extract_authors({}) == []


# ── _extract_year ─────────────────────────────────────────────────


def test_year_full_date() -> None:
    assert cs._extract_year({"issued": {"date-parts": [[2020, 5, 1]]}}) == 2020


def test_year_year_only() -> None:
    assert cs._extract_year({"issued": {"date-parts": [[2019]]}}) == 2019


def test_year_missing_issued() -> None:
    assert cs._extract_year({}) is None


def test_year_empty_parts() -> None:
    assert cs._extract_year({"issued": {"date-parts": [[]]}}) is None


# ── _to_metadata ──────────────────────────────────────────────────


def test_to_metadata_full_record() -> None:
    work = {
        "DOI": "10.1/x",
        "title": ["A Title"],
        "author": [{"given": "A", "family": "B"}],
        "issued": {"date-parts": [[2021]]},
        "container-title": ["Journal X"],
    }
    rec = cs._to_metadata(work, "kw1")
    assert rec == {
        "pdf_path": None,
        "title": "A Title",
        "authors": ["A B"],
        "year": 2021,
        "journal": "Journal X",
        "matched_keywords": ["kw1"],
        "download_status": "pending",
        "source": None,
    }


def test_to_metadata_missing_title_and_journal() -> None:
    rec = cs._to_metadata({"DOI": "10.1/x"}, "kw")
    assert rec["title"] is None
    assert rec["journal"] is None


# ── search (мок _search_pool) ─────────────────────────────────────


def _work(doi: str, score: float = 10.0) -> dict[str, Any]:
    return {"DOI": doi, "title": [f"title {doi}"], "score": score}


def test_search_dedups_and_accumulates_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    pools = {
        "kw1": [_work("10.1/a"), _work("10.1/b")],
        "kw2": [_work("10.1/b"), _work("10.1/c")],
    }

    def fake_pool(_client: Any, keyword: str, **_kw: Any) -> list[dict[str, Any]]:
        return pools[keyword]

    monkeypatch.setattr(cs, "_search_pool", fake_pool)

    result = cs.search(["kw1", "kw2"])

    assert set(result) == {"10.1/a", "10.1/b", "10.1/c"}
    # Статья из обоих пулов помечена обоими ключевиками.
    assert result["10.1/b"]["matched_keywords"] == ["kw1", "kw2"]
    assert result["10.1/a"]["matched_keywords"] == ["kw1"]


def test_search_skips_works_without_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pool(_client: Any, _keyword: str, **_kw: Any) -> list[dict[str, Any]]:
        return [{"title": ["no doi"], "score": 10.0}, _work("10.1/ok")]

    monkeypatch.setattr(cs, "_search_pool", fake_pool)

    result = cs.search(["kw"])
    assert list(result) == ["10.1/ok"]


def test_search_empty_keywords() -> None:
    assert cs.search([]) == {}
