"""Скачивание PDF по DOI: Unpaywall (легально) → Sci-Hub через Playwright.

Двухэтапная стратегия: сначала быстрый и легальный Unpaywall, и только если
статья не в открытом доступе — браузерный обход Sci-Hub с перехватом трафика.

Здесь же — связка с поиском: `run()` берёт DOI из crossref_search, качает
PDF и пишет metadata.json, контракт для Модуля 2.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urljoin

import requests
from playwright.sync_api import BrowserContext, Download, Response, sync_playwright
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .crossref_search import search

UNPAYWALL_EMAIL = "berestovsisa@gmail.com"

# Пути общего volume — на них завязан Модуль 2 (см. его README).
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
METADATA_PATH = DATA_DIR / "metadata.json"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)
# 403 тут — обычно не «нельзя», а «слишком часто»: проверено на ijcmr.com,
# где ссылка отдавала 403 в прогоне и 200 через несколько секунд подряд.
RETRIABLE_CODES = frozenset({403, 429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 3

# Без Accept и Accept-Language часть издателей отвечает 403 сразу.
DOWNLOAD_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Порядок = приоритет. .ru проверен рабочим на живом прогоне; .cat взят из
# редиректа самого Sci-Hub, но как вход по DOI пока НЕ подтверждён.
# Ранее тут были .se и .st — оба мертвы (NAME_NOT_RESOLVED и TIMED_OUT).
SCIHUB_MIRRORS = (
    "https://sci-hub.ru",
    "https://sci-hub.cat",
)
MIRROR_TIMEOUT_MS = 15000  # мёртвый домен не должен вешать прогон надолго
ALTCHA_WAIT_MS = 8000  # ждём, пока JS дорисует виджет проверки на робота
PDF_WAIT_SECONDS = 20  # сколько ждём, пока PDF пролетит по сети
TIMEOUT = 20.0


class IPBlocked(Exception):
    """DDoS-Guard завернул нас по IP — на других зеркалах будет то же самое."""


class FetchResult(NamedTuple):
    """Итог попытки скачать одну статью."""

    ok: bool
    pdf_path: str | None = None
    source: str | None = None  # "oa" (открытые базы) | "scihub"


def get_unpaywall_urls(doi: str) -> list[str]:
    """Первый этап: все известные Unpaywall места, где лежит открытая копия.

    Берём не только best_oa_location, как раньше: у одной статьи бывает
    несколько зеркал (репозиторий вуза, PMC, DOAJ), и если основное отдаёт
    403 — копия рядом может скачаться без проблем.
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    candidates: list[str] = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            if data.get("is_oa"):
                locations = [data.get("best_oa_location"), *(data.get("oa_locations") or [])]
                for loc in locations:
                    if not loc:
                        continue
                    # url_for_pdf — прямая ссылка, url — лендинг статьи;
                    # с лендинга PDF достаём через citation_pdf_url.
                    for key in ("url_for_pdf", "url"):
                        candidate = loc.get(key)
                        if candidate and candidate not in candidates:
                            candidates.append(str(candidate))
    except Exception as e:
        print(f"[~] Ошибка Unpaywall: {e}")
    return candidates


def get_semanticscholar_urls(doi: str) -> list[str]:
    """Открытая копия по данным Semantic Scholar.

    Полезен тем, что знает копии под ДРУГИМ DOI: препринт или версию в ином
    журнале, о которых Unpaywall по исходному идентификатору не в курсе.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    try:
        response = requests.get(
            url, params={"fields": "openAccessPdf"}, headers=DOWNLOAD_HEADERS, timeout=15
        )
        if response.status_code != 200:
            return []
        oa = response.json().get("openAccessPdf") or {}
        return [str(oa["url"])] if oa.get("url") else []
    except Exception as e:
        print(f"[~] Ошибка Semantic Scholar: {e}")
        return []


def get_europepmc_urls(doi: str) -> list[str]:
    """Открытые полные тексты Europe PMC (биомед и смежная химия)."""
    try:
        response = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"},
            headers=DOWNLOAD_HEADERS,
            timeout=15,
        )
        if response.status_code != 200:
            return []
        results = response.json().get("resultList", {}).get("result", [])
        if not results:
            return []
        links = (results[0].get("fullTextUrlList") or {}).get("fullTextUrl", [])
        return [str(link["url"]) for link in links if link.get("documentStyle") == "pdf"]
    except Exception as e:
        print(f"[~] Ошибка Europe PMC: {e}")
        return []


def get_oa_urls(doi: str) -> list[str]:
    """Все бесплатные источники разом: Unpaywall + Semantic Scholar + Europe PMC.

    Порядок источников = порядок надёжности по нашим прогонам. Дубли убираем:
    одна и та же ссылка нередко приходит из двух баз сразу.
    """
    candidates: list[str] = []
    for name, getter in (
        ("Unpaywall", get_unpaywall_urls),
        ("Semantic Scholar", get_semanticscholar_urls),
        ("Europe PMC", get_europepmc_urls),
    ):
        found = getter(doi)
        if found:
            print(f"[+] {name}: {len(found)} ссылк(и)")
        for url in found:
            if url not in candidates:
                candidates.append(url)
    return candidates


def _try_mirror(context: BrowserContext, mirror: str, doi: str) -> str | None:
    """Одна попытка достать PDF с конкретного зеркала."""
    page = context.new_page()

    # Ловим PDF по content-type, а не по тегам в разметке: Sci-Hub часто
    # меняет вёрстку, но сам файл всё равно проходит через сеть.
    pdf_urls_found: list[str] = []

    def handle_response(response: Response) -> None:
        try:
            content_type = response.header_value("content-type")
            if content_type and "application/pdf" in content_type.lower():
                print(f"[+] Перехвачен PDF-трафик: {response.url}")
                pdf_urls_found.append(response.url)
        except Exception:
            pass

    page.on("response", handle_response)

    try:
        try:
            page.goto(
                f"{mirror}/{doi}",
                wait_until="domcontentloaded",
                timeout=MIRROR_TIMEOUT_MS,
            )
        except PlaywrightError as e:
            # Домен мёртв или не резолвится — просто пробуем следующее зеркало.
            print(f"[-] {mirror} недоступно: {str(e).splitlines()[0]}")
            return None

        # Перед самим Sci-Hub стоит DDoS-Guard. Если он решил проверить
        # браузер — мы до статьи не доехали, и ждать PDF бессмысленно.
        # Без этой проверки код молча ждал 20 секунд и врал «не вылезла».
        # Частая причина — включённый VPN: защита режет по IP.
        if "DDOS-GUARD" in page.title().upper():
            raise IPBlocked(mirror)

        # Altcha показывает вопрос с кнопкой «Нет». Виджет дорисовывается
        # джаваскриптом уже после domcontentloaded, поэтому именно ЖДЁМ его
        # появления: is_visible() проверяет мгновенно и на не отрисованной
        # ещё странице всегда скажет «нет».
        # Если PDF уже прилетел — проверки не было вовсе, ждать её нечего.
        if not pdf_urls_found:
            answer_btn = page.locator("div.answer").first
            try:
                answer_btn.wait_for(state="visible", timeout=ALTCHA_WAIT_MS)
            except PlaywrightTimeoutError:
                print("[*] Проверка на робота не вылезла, ждём загрузки документа...")
            else:
                print("[*] Обнаружена защита Altcha. Нажимаем «Нет»...")
                answer_btn.click()
                print("[*] Ждём завершения проверки и начала загрузки файла...")

        for _ in range(PDF_WAIT_SECONDS):
            if pdf_urls_found:
                break
            page.wait_for_timeout(1000)

        if pdf_urls_found:
            final_url = pdf_urls_found[0]
            return f"https:{final_url}" if final_url.startswith("//") else final_url

        # Резерв: иногда Sci-Hub просто редиректит прямо на PDF.
        if ".pdf" in page.url.lower():
            return page.url

        print(f"[-] {mirror}: перехватчик не поймал PDF-файл.")
        return None

    finally:
        page.close()


def get_scihub_url_playwright(doi: str, *, headless: bool = False) -> str | None:
    """Второй этап: обход защиты с перехватом трафика, с перебором зеркал.

    Зеркала перебираем не ради обхода блокировок по IP (от них это не спасает —
    DDoS-Guard режет одинаково на всех доменах), а потому что домены Sci-Hub
    периодически умирают по решениям судов: живое зеркало сегодня может не
    отвечать завтра.

    headless=False по умолчанию и менять не стоит: с headless=True проверка
    не проходится (проверено на рабочем DOI — с окном качает, без окна нет),
    видимо, режим распознаётся защитой. Отсюда неприятное следствие: пакетный
    прогон идёт с открывающимся окном браузера и без человека не оставить.
    """
    if headless:
        print("[~] headless=True: на Sci-Hub так не проходит, ожидай провала.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=BROWSER_UA,
            viewport={"width": 1280, "height": 720},
        )
        try:
            for mirror in SCIHUB_MIRRORS:
                print(f"[*] Пробуем зеркало {mirror}...")
                try:
                    pdf_url = _try_mirror(context, mirror, doi)
                except IPBlocked:
                    # Блокировка по IP, а не по домену — остальные зеркала
                    # ответят тем же. Не долбим их впустую.
                    print("[-] DDoS-Guard проверяет браузер. Отключи VPN и повтори.")
                    return None
                if pdf_url:
                    return pdf_url
            return None
        finally:
            browser.close()


def _get_with_retry(url: str) -> requests.Response | None:
    """GET с повтором на временных отказах.

    Издатели режут всплески запросов: одна и та же ссылка отдаёт 403 в середине
    прогона и спокойно 200 через несколько секунд. Поэтому 403/429/5xx считаем
    временными и ждём с нарастающей паузой, а не хороним статью сразу.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=TIMEOUT)
        except Exception as e:
            print(f"[~] Попытка {attempt + 1}: {type(e).__name__}: {str(e)[:60]}")
        else:
            if response.status_code == 200:
                return response
            if response.status_code not in RETRIABLE_CODES:
                print(f"[-] {response.status_code} от сервера, повтор не поможет.")
                return None
            print(f"[~] {response.status_code} от сервера (лимит запросов?), попытка {attempt + 1}")

        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))

    print("[-] Сервер так и не отдал файл.")
    return None


def _find_pdf_link(html: bytes, base_url: str) -> str | None:
    """Ищет на странице статьи мета-тег citation_pdf_url со ссылкой на PDF.

    Стандартный тег, который проставляет большинство издателей и агрегаторов,
    — надёжнее, чем угадывать ссылку по вёрстке конкретного сайта.
    """
    match = re.search(rb'citation_pdf_url"?\s+content="([^"]+)', html)
    return urljoin(base_url, match.group(1).decode(errors="replace")) if match else None


def download_file(pdf_url: str, save_path: Path) -> bool:
    """Скачивание файла по ссылке; при лендинге — переход на сам PDF.

    Проверяем не Content-Type, а сигнатуру `%PDF-` в начале файла: заголовок
    врёт в обе стороны — часть серверов отдаёт PDF как octet-stream, а часть
    возвращает HTML-заглушку с типом application/pdf.
    """
    try:
        response = _get_with_retry(pdf_url)
        if response is None:
            return False

        # Попали на страницу статьи вместо файла — ищем на ней ссылку на PDF.
        if not response.content.startswith(b"%PDF-"):
            link = _find_pdf_link(response.content, response.url)
            if not link or link == pdf_url:
                ctype = response.headers.get("Content-Type", "?")
                print(f"[-] По ссылке не PDF (Content-Type: {ctype}).")
                return False
            print(f"[*] Это лендинг, идём за PDF: {link}")
            response = requests.get(link, headers=DOWNLOAD_HEADERS, timeout=TIMEOUT)
            if not response.content.startswith(b"%PDF-"):
                print("[-] По ссылке с лендинга тоже не PDF.")
                return False

        save_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"[-] Ошибка при скачивании файла: {e}")
        return False


def _browser_download(context: BrowserContext, url: str, save_path: Path) -> bool:
    """Забирает PDF через браузер, когда обычный HTTP-клиент словил 403.

    Издатели (MDPI, PMC, DOAJ) отшивают requests, но пускают настоящий браузер.
    Файл берём двумя путями: либо Chromium сам инициирует скачивание, либо PDF
    просто пролетает по сети — тогда читаем тело ответа. Через requests повторно
    не ходим: ссылка на файл нередко одноразовая и завязана на сессию.
    """
    page = context.new_page()
    pdf_bodies: list[bytes] = []
    downloads: list[Download] = []

    def handle_response(response: Response) -> None:
        try:
            content_type = response.header_value("content-type")
            if content_type and "application/pdf" in content_type.lower():
                pdf_bodies.append(response.body())
        except Exception:
            pass

    page.on("response", handle_response)
    page.on("download", lambda d: downloads.append(d))

    try:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=MIRROR_TIMEOUT_MS)
        except PlaywrightError:
            # Прямая ссылка на файл рвёт навигацию (ERR_ABORTED) — это норма,
            # скачивание при этом идёт своим чередом.
            pass

        for _ in range(PDF_WAIT_SECONDS):
            if pdf_bodies or downloads:
                break
            page.wait_for_timeout(1000)

        if downloads:
            downloads[0].save_as(save_path)
            return save_path.exists() and save_path.stat().st_size > 0

        if pdf_bodies and pdf_bodies[0].startswith(b"%PDF-"):
            save_path.write_bytes(pdf_bodies[0])
            return True

        # Открылся лендинг — ищем на нём прямую ссылку и пробуем ещё раз.
        link = _find_pdf_link(page.content().encode(), page.url)
        if link and link != url:
            print(f"[*] Лендинг в браузере, идём за PDF: {link}")
            return _browser_download(context, link, save_path)

        return False
    except Exception as e:
        print(f"[-] Браузер не смог забрать файл: {e}")
        return False
    finally:
        page.close()


def download_via_browser(urls: list[str], save_path: Path, *, headless: bool = False) -> str | None:
    """Пробует забрать PDF браузером по каждой ссылке; вернёт сработавшую."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=BROWSER_UA,
            viewport={"width": 1280, "height": 720},
            accept_downloads=True,
        )
        try:
            for i, url in enumerate(urls, start=1):
                print(f"[*] Браузером [{i}/{len(urls)}]: {url[:70]}")
                if _browser_download(context, url, save_path):
                    return url
            return None
        finally:
            browser.close()


def safe_filename(doi: str) -> str:
    """DOI -> имя файла без спецсимволов, недопустимых в путях."""
    return re.sub(r'[\\/*?:"<>|]', "_", doi)


def fetch_article(
    doi: str,
    save_dir: Path | str = ".",
    *,
    headless: bool = False,
    use_scihub: bool = True,
) -> FetchResult:
    """Основной оркестратор: достать ссылку и скачать PDF по одному DOI.

    use_scihub=False нужен для прогона с включённым VPN: часть издателей
    отдаёт статьи только из определённых регионов, а Sci-Hub при VPN наглухо
    закрыт DDoS-Guard. Гонять его в таком заходе — только терять время.
    """
    print(f"\n{'=' * 40}\n[*] Обработка DOI: {doi}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{safe_filename(doi)}.pdf"

    # Сначала Unpaywall, при неудаче — Sci-Hub. Именно ПРИ НЕУДАЧЕ СКАЧИВАНИЯ,
    # а не только когда ссылки нет: Unpaywall часто отдаёт ссылку на лендинг
    # издателя вместо PDF, и раньше такие статьи терялись, хотя на Sci-Hub
    # они вполне могли быть.
    candidates = get_oa_urls(doi)
    if candidates:
        print(f"[+] Открытых источников всего: {len(candidates)}")
        for i, candidate in enumerate(candidates, start=1):
            print(f"[*] [{i}/{len(candidates)}] {candidate}")
            if download_file(candidate, save_path):
                print(f"[+] Сохранено: {save_path}")
                return FetchResult(ok=True, pdf_path=str(save_path), source="oa")
        # HTTP-клиента издатели часто отшивают по 403, а браузер пускают —
        # ссылки-то рабочие, проверено руками. Поэтому прежде чем идти на
        # Sci-Hub, пробуем те же самые OA-ссылки, но через Playwright.
        # Тут headless=True жёстко, независимо от аргумента: OA-источники
        # его пропускают (проверено на MDPI/CERN), а окно ради них открывать
        # незачем — видимый браузер нужен только Sci-Hub.
        print("[!] HTTP-клиент не справился. Пробуем те же ссылки браузером...")
        used = download_via_browser(candidates, save_path, headless=True)
        if used:
            print(f"[+] Сохранено: {save_path}")
            return FetchResult(ok=True, pdf_path=str(save_path), source="unpaywall")
        print("[!] Браузер тоже не забрал OA-копию.")
    else:
        print("[!] Открытых копий не знает ни одна база.")

    if not use_scihub:
        print("[-] Sci-Hub отключён (use_scihub=False) — пропускаем.")
        return FetchResult(ok=False)

    print("[*] Запускаем Playwright для Sci-Hub...")
    source = "scihub"
    pdf_url = get_scihub_url_playwright(doi, headless=headless)

    if not pdf_url:
        print("[-] Провал: ссылку достать не удалось.")
        return FetchResult(ok=False)

    print(f"[+] Ссылка получена: {pdf_url}")
    if download_file(pdf_url, save_path):
        print(f"[+] Сохранено: {save_path}")
        return FetchResult(ok=True, pdf_path=str(save_path), source=source)
    return FetchResult(ok=False)


def _write_metadata(manifest: dict[str, dict[str, Any]], path: Path) -> None:
    """Атомарная запись контракта: сначала во временный файл, потом подмена.

    Иначе обрыв посреди записи оставит Модулю 2 битый JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Читает прошлый прогон, чтобы не качать заново уже скачанное."""
    if not path.exists():
        return {}
    try:
        loaded: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except (json.JSONDecodeError, OSError) as e:
        print(f"[~] Не смог прочитать {path}, начинаю с нуля: {e}")
        return {}


def run(
    keywords: list[str],
    *,
    from_year: int = 2015,
    until_year: int = 2024,
    mailto: str | None = None,
    pdf_dir: Path = PDF_DIR,
    metadata_path: Path = METADATA_PATH,
    headless: bool = False,
    limit: int | None = None,
    use_scihub: bool = True,
) -> dict[str, dict[str, Any]]:
    """Полный проход: найти статьи по темам и скачать их PDF.

    Поиск отдаёт заготовки записей со статусом "pending", качалка их
    дозаполняет: `pdf_path`, `source`, финальный статус. metadata.json пишем
    после каждой статьи — прогон идёт минутами и может оборваться, терять
    из-за этого уже скачанное не хочется.

    Args:
        keywords: ключевые слова/темы для поиска.
        from_year, until_year: границы по году публикации.
        mailto: e-mail для polite pool CrossRef.
        pdf_dir: куда складывать PDF (общий volume Модуля 2).
        metadata_path: куда писать контракт metadata.json.
        headless: гонять браузер Sci-Hub без окна. Не работает — защита
            распознаёт headless; оставлено параметром на случай, если
            когда-нибудь заработает.
        limit: ограничить число скачиваний (удобно для проверки связки).
        use_scihub: пробовать ли Sci-Hub. Выключить для захода с VPN —
            под VPN Sci-Hub закрыт, зато открываются гео-ограниченные
            издатели, так что выборка добирается двумя прогонами.

    Returns:
        Итоговый manifest: DOI -> запись метаданных.
    """
    print(f"[*] Поиск по темам: {', '.join(keywords)}")
    found = search(keywords, from_year=from_year, until_year=until_year, mailto=mailto)
    print(f"[+] CrossRef вернул {len(found)} уникальных статей")

    # Прошлый прогон в приоритете: уже скачанные записи переносим как есть.
    manifest = _load_metadata(metadata_path)
    for doi, record in found.items():
        if doi not in manifest:
            manifest[doi] = record

    pending = [doi for doi, rec in manifest.items() if rec.get("download_status") != "ok"]
    already = len(manifest) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    print(f"[*] К скачиванию: {len(pending)} (уже скачано ранее: {already})")

    for i, doi in enumerate(pending, start=1):
        print(f"\n--- [{i}/{len(pending)}] ---")
        try:
            result = fetch_article(doi, save_dir=pdf_dir, headless=headless, use_scihub=use_scihub)
        except Exception as e:
            # Одна упавшая статья не должна ронять весь прогон.
            print(f"[-] Ошибка на {doi}: {e}")
            manifest[doi]["download_status"] = "error"
        else:
            if result.ok:
                manifest[doi]["pdf_path"] = result.pdf_path
                manifest[doi]["source"] = result.source
                manifest[doi]["download_status"] = "ok"
            else:
                manifest[doi]["download_status"] = "not_found"

        _write_metadata(manifest, metadata_path)

    ok = sum(1 for rec in manifest.values() if rec.get("download_status") == "ok")
    print(f"\n{'=' * 40}")
    print(f"[+] Готово: {ok}/{len(manifest)} статей с PDF")
    print(f"[+] Контракт записан: {metadata_path}")
    return manifest


if __name__ == "__main__":
    run(["oxidation induction time", "antioxidants"], limit=3)
