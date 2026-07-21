"""Тесты качалки и связки. Сеть и браузер мокаются — прогон офлайн."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import requests
from downloader import scihub_downloader as sd

PDF_BYTES = b"%PDF-1.5\n...binary..."
HTML_BYTES = b"<!DOCTYPE html><html><body>paywall</body></html>"


class FakeResponse:
    """Замена requests.Response для тестов — только то, что читает код."""

    def __init__(
        self,
        status_code: int = 200,
        content: bytes = PDF_BYTES,
        url: str = "https://example.org/a.pdf",
        content_type: str = "application/pdf",
        payload: Any = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.url = url
        self.headers = {"Content-Type": content_type}
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _patch_requests_get(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    # Патчим общий модуль requests — тот же объект, что использует качалка.
    monkeypatch.setattr(requests, "get", fn)


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)


# ── get_unpaywall_urls ────────────────────────────────────────────


def test_unpaywall_collects_all_locations(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "is_oa": True,
        "best_oa_location": {"url_for_pdf": "A", "url": "landingA"},
        "oa_locations": [{"url_for_pdf": "B"}, {"url": "landingC"}],
    }
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload=payload))
    assert sd.get_unpaywall_urls("10.1/x") == ["A", "landingA", "B", "landingC"]


def test_unpaywall_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "is_oa": True,
        "best_oa_location": {"url_for_pdf": "A"},
        "oa_locations": [{"url_for_pdf": "A"}],
    }
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload=payload))
    assert sd.get_unpaywall_urls("10.1/x") == ["A"]


def test_unpaywall_not_oa(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload={"is_oa": False}))
    assert sd.get_unpaywall_urls("10.1/x") == []


def test_unpaywall_bad_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(status_code=404))
    assert sd.get_unpaywall_urls("10.1/x") == []


def test_unpaywall_swallows_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> FakeResponse:
        raise RuntimeError("network down")

    _patch_requests_get(monkeypatch, boom)
    assert sd.get_unpaywall_urls("10.1/x") == []


# ── get_semanticscholar_urls / get_europepmc_urls ─────────────────


def test_semanticscholar_returns_url(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"openAccessPdf": {"url": "https://s2/pdf"}}
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload=payload))
    assert sd.get_semanticscholar_urls("10.1/x") == ["https://s2/pdf"]


def test_semanticscholar_null_oa(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload={"openAccessPdf": None}))
    assert sd.get_semanticscholar_urls("10.1/x") == []


def test_europepmc_keeps_only_pdf_style(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "resultList": {
            "result": [
                {
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {"documentStyle": "pdf", "url": "PDF"},
                            {"documentStyle": "html", "url": "HTML"},
                        ]
                    }
                }
            ]
        }
    }
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload=payload))
    assert sd.get_europepmc_urls("10.1/x") == ["PDF"]


def test_europepmc_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(payload={"resultList": {}}))
    assert sd.get_europepmc_urls("10.1/x") == []


# ── get_oa_urls (мок трёх геттеров) ───────────────────────────────


def test_oa_urls_merges_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd, "get_unpaywall_urls", lambda _d: ["A"])
    monkeypatch.setattr(sd, "get_semanticscholar_urls", lambda _d: ["A", "B"])
    monkeypatch.setattr(sd, "get_europepmc_urls", lambda _d: ["C"])
    assert sd.get_oa_urls("10.1/x") == ["A", "B", "C"]


# ── _get_with_retry ───────────────────────────────────────────────


def _sequence(codes: list[int]) -> tuple[Any, dict[str, int]]:
    """Заглушка requests.get, отдающая коды по очереди, и счётчик вызовов."""
    state = {"i": 0}

    def fn(*_a: Any, **_k: Any) -> FakeResponse:
        code = codes[state["i"]]
        state["i"] += 1
        return FakeResponse(status_code=code)

    return fn, state


def test_retry_success_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    fn, state = _sequence([200])
    _patch_requests_get(monkeypatch, fn)
    assert sd._get_with_retry("u") is not None
    assert state["i"] == 1


def test_retry_recovers_after_403(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    fn, state = _sequence([403, 200])
    _patch_requests_get(monkeypatch, fn)
    assert sd._get_with_retry("u") is not None
    assert state["i"] == 2


def test_retry_gives_up_on_fatal_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # 404 не входит в RETRIABLE_CODES — сдаёмся сразу, без повторов.
    _no_sleep(monkeypatch)
    fn, state = _sequence([404])
    _patch_requests_get(monkeypatch, fn)
    assert sd._get_with_retry("u") is None
    assert state["i"] == 1


def test_retry_exhausts_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    fn, state = _sequence([500, 500, 500])
    _patch_requests_get(monkeypatch, fn)
    assert sd._get_with_retry("u") is None
    assert state["i"] == sd.RETRY_ATTEMPTS


# ── _find_pdf_link ────────────────────────────────────────────────


def test_find_pdf_link_resolves_relative() -> None:
    html = b'<meta name="citation_pdf_url" content="/files/a.pdf">'
    assert sd._find_pdf_link(html, "https://pub.org/article/1") == "https://pub.org/files/a.pdf"


def test_find_pdf_link_absent() -> None:
    assert sd._find_pdf_link(b"<html>nothing</html>", "https://pub.org") is None


# ── download_file ─────────────────────────────────────────────────


def test_download_file_saves_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "_get_with_retry", lambda _u: FakeResponse(content=PDF_BYTES))
    dest = tmp_path / "out.pdf"
    assert sd.download_file("u", dest) is True
    assert dest.read_bytes() == PDF_BYTES


def test_download_file_rejects_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    html = FakeResponse(content=HTML_BYTES, content_type="text/html")
    monkeypatch.setattr(sd, "_get_with_retry", lambda _u: html)
    dest = tmp_path / "out.pdf"
    assert sd.download_file("u", dest) is False
    assert not dest.exists()


def test_download_file_follows_landing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    landing_html = b'<meta name="citation_pdf_url" content="https://pub.org/real.pdf">'
    landing = FakeResponse(
        content=landing_html, url="https://pub.org/art", content_type="text/html"
    )
    monkeypatch.setattr(sd, "_get_with_retry", lambda _u: landing)
    # Переход на найденную ссылку идёт напрямую через requests.get.
    _patch_requests_get(monkeypatch, lambda *a, **k: FakeResponse(content=PDF_BYTES))
    dest = tmp_path / "out.pdf"
    assert sd.download_file("u", dest) is True
    assert dest.read_bytes() == PDF_BYTES


def test_download_file_none_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "_get_with_retry", lambda _u: None)
    assert sd.download_file("u", tmp_path / "out.pdf") is False


# ── safe_filename ─────────────────────────────────────────────────


def test_safe_filename_replaces_slashes() -> None:
    assert sd.safe_filename("10.1016/j.corsci.2015.11.011") == "10.1016_j.corsci.2015.11.011"


def test_safe_filename_strips_forbidden_chars() -> None:
    assert sd.safe_filename('a/b:c*d?"e<f>g|h') == "a_b_c_d__e_f_g_h"


# ── _write_metadata / _load_metadata ──────────────────────────────


def test_metadata_roundtrip(tmp_path: Path) -> None:
    data = {"10.1/x": {"title": "статья", "download_status": "ok"}}
    path = tmp_path / "sub" / "metadata.json"
    sd._write_metadata(data, path)  # создаёт и родительскую папку
    assert sd._load_metadata(path) == data
    assert not path.with_suffix(".json.tmp").exists()  # временный файл убран


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert sd._load_metadata(tmp_path / "nope.json") == {}


def test_load_corrupt_json_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert sd._load_metadata(path) == {}


# ── fetch_article (ветки источников) ──────────────────────────────


def test_fetch_via_oa_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "get_oa_urls", lambda _d: ["http://oa/pdf"])
    monkeypatch.setattr(sd, "download_file", lambda _u, _p: True)
    result = sd.fetch_article("10.1/x", save_dir=tmp_path)
    assert result.ok is True
    assert result.source == "oa"


def test_fetch_oa_falls_back_to_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "get_oa_urls", lambda _d: ["http://oa/pdf"])
    monkeypatch.setattr(sd, "download_file", lambda _u, _p: False)
    monkeypatch.setattr(sd, "download_via_browser", lambda _u, _p, headless=True: "http://oa/pdf")
    result = sd.fetch_article("10.1/x", save_dir=tmp_path)
    assert result.ok is True
    assert result.source == "oa"


def test_fetch_scihub_off_when_no_oa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "get_oa_urls", lambda _d: [])
    result = sd.fetch_article("10.1/x", save_dir=tmp_path, use_scihub=False)
    assert result.ok is False


def test_fetch_via_scihub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sd, "get_oa_urls", lambda _d: [])
    monkeypatch.setattr(
        sd, "get_scihub_url_playwright", lambda _d, headless=False: "http://sci/pdf"
    )
    monkeypatch.setattr(sd, "download_file", lambda _u, _p: True)
    result = sd.fetch_article("10.1/x", save_dir=tmp_path)
    assert result.ok is True
    assert result.source == "scihub"


# ── run (мок search и fetch_article) ──────────────────────────────


def _stub_search(records: dict[str, dict[str, Any]]) -> Any:
    def fn(*_a: Any, **_k: Any) -> dict[str, dict[str, Any]]:
        return {doi: dict(rec) for doi, rec in records.items()}

    return fn


def test_run_fills_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = {
        "10.1/ok": {"download_status": "pending"},
        "10.1/no": {"download_status": "pending"},
    }
    monkeypatch.setattr(sd, "search", _stub_search(records))

    def fake_fetch(doi: str, **_k: Any) -> sd.FetchResult:
        if doi == "10.1/ok":
            return sd.FetchResult(ok=True, pdf_path="p.pdf", source="oa")
        return sd.FetchResult(ok=False)

    monkeypatch.setattr(sd, "fetch_article", fake_fetch)

    out = sd.run(["kw"], pdf_dir=tmp_path / "pdfs", metadata_path=tmp_path / "m.json")
    assert out["10.1/ok"]["download_status"] == "ok"
    assert out["10.1/ok"]["source"] == "oa"
    assert out["10.1/no"]["download_status"] == "not_found"


def test_run_respects_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = {f"10.1/{i}": {"download_status": "pending"} for i in range(5)}
    monkeypatch.setattr(sd, "search", _stub_search(records))

    seen: list[str] = []

    def fake_fetch(doi: str, **_k: Any) -> sd.FetchResult:
        seen.append(doi)
        return sd.FetchResult(ok=True, pdf_path="p", source="oa")

    monkeypatch.setattr(sd, "fetch_article", fake_fetch)

    sd.run(["kw"], pdf_dir=tmp_path / "pdfs", metadata_path=tmp_path / "m.json", limit=2)
    assert len(seen) == 2


def test_run_skips_already_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meta = tmp_path / "m.json"
    # Прошлый прогон: одна статья уже скачана.
    sd._write_metadata({"10.1/done": {"download_status": "ok", "pdf_path": "old.pdf"}}, meta)

    records = {
        "10.1/done": {"download_status": "pending"},
        "10.1/new": {"download_status": "pending"},
    }
    monkeypatch.setattr(sd, "search", _stub_search(records))

    seen: list[str] = []

    def fake_fetch(doi: str, **_k: Any) -> sd.FetchResult:
        seen.append(doi)
        return sd.FetchResult(ok=True, pdf_path="p", source="oa")

    monkeypatch.setattr(sd, "fetch_article", fake_fetch)

    sd.run(["kw"], pdf_dir=tmp_path / "pdfs", metadata_path=meta)
    # Уже скачанную не трогаем, качаем только новую.
    assert seen == ["10.1/new"]


def test_run_survives_fetch_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = {"10.1/boom": {"download_status": "pending"}}
    monkeypatch.setattr(sd, "search", _stub_search(records))

    def fake_fetch(_doi: str, **_k: Any) -> sd.FetchResult:
        raise RuntimeError("browser crashed")

    monkeypatch.setattr(sd, "fetch_article", fake_fetch)

    out = sd.run(["kw"], pdf_dir=tmp_path / "pdfs", metadata_path=tmp_path / "m.json")
    # Падение одной статьи не роняет прогон — она помечается error.
    assert out["10.1/boom"]["download_status"] == "error"
