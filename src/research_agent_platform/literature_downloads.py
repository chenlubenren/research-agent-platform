from __future__ import annotations

import asyncio
import re
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import httpx

from .connectors.scholar import LiteratureBundle, PaperRecord
from .literature_sources import DownloadSource, discover_download_sources, extract_download_queries, download_targets_markdown


PDF_PATH_HINTS = ("/pdf/", ".pdf")
DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/")
UNPAYWALL_API = "https://api.unpaywall.org/v2"


async def _resolve_unpaywall_pdf(client: httpx.AsyncClient, doi: str, email: str) -> str:
    """Resolve an open-access PDF URL for a DOI via Unpaywall; return '' on any failure."""
    doi = _normalize_doi(doi)
    email = (email or "").strip()
    if not doi or not email:
        return ""
    try:
        response = await client.get(
            f"{UNPAYWALL_API}/{doi}",
            params={"email": email},
            headers={"User-Agent": "research-agent-platform"},
        )
        if response.status_code in (404, 422):
            return ""
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    best = data.get("best_oa_location") or {}
    if isinstance(best, dict):
        candidate = best.get("url_for_pdf") or best.get("url")
        if candidate:
            return str(candidate)
    for location in data.get("oa_locations", []) or []:
        if isinstance(location, dict):
            candidate = location.get("url_for_pdf") or location.get("url")
            if candidate:
                return str(candidate)
    return ""


def _safe_stem(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", value).strip(" ._")
    return (cleaned or fallback)[:100]


def _is_pdf_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    path = parsed.path.casefold()
    query = parsed.query.casefold()
    return path.endswith(".pdf") or any(hint in path for hint in PDF_PATH_HINTS) or "format=pdf" in query or "type=pdf" in query


def _normalize_doi(value: str) -> str:
    cleaned = value.strip()
    for prefix in DOI_PREFIXES:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.strip().rstrip(".,;)")


def _doi_to_resolver_url(doi: str) -> str:
    doi = _normalize_doi(doi)
    return f"https://doi.org/{doi}" if doi else ""


def _candidate_urls(paper: PaperRecord, source: DownloadSource | None = None) -> list[str]:
    urls: list[str] = []
    candidates = [
        paper.pdf_url,
        paper.url,
    ]
    if source:
        candidates.extend([source.url, _doi_to_resolver_url(source.doi)])
    if paper.identifiers.get("doi"):
        candidates.append(_doi_to_resolver_url(paper.identifiers["doi"]))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip()
            if normalized not in urls:
                urls.append(normalized)
    return urls


async def download_public_pdfs(
    bundle: LiteratureBundle,
    workspace_root: Path,
    *,
    enabled: bool = True,
    limit: int = 30,
    max_mb: int = 50,
    timeout_seconds: float = 60.0,
    unpaywall_email: str = "",
    write_file: Callable[[str, bytes], object] | None = None,
    concurrency: int = 6,
) -> dict[str, object]:
    """Download public PDFs concurrently, preserving the ``limit``/``skipped_limit`` contract.

    The caller is expected to pre-filter/-order ``bundle.papers`` (e.g. by screen tier), so
    the first ``limit`` papers that expose a public URL/DOI are fetched concurrently and any
    further URL-bearing papers are marked ``skipped_limit``.
    """

    papers_root = workspace_root / "bib" / "papers"
    papers_root.mkdir(parents=True, exist_ok=True)
    max_bytes = max(1, max_mb) * 1024 * 1024
    candidate_sources = discover_download_sources(workspace_root, _workspace_download_refs(workspace_root))
    query_terms = extract_download_queries(candidate_sources)

    entries: list[dict[str, object]] = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "pdf_url": paper.pdf_url,
            "candidate_urls": [],
            "source_path": "",
            "status": "not_attempted",
            "relative_path": "",
            "bytes": 0,
            "error": "",
            "_paper": paper,
        }
        for paper in bundle.papers
    ]

    if not enabled:
        for entry in entries:
            entry["status"] = "disabled"
            entry["_paper"].download_status = "disabled"
    else:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "research-agent-platform"},
        ) as client:
            # Phase 1: resolve candidate URLs (may consult Unpaywall) and partition by limit.
            for entry in entries:
                paper = entry["_paper"]
                source = _match_source(paper, candidate_sources, query_terms)
                candidate_urls = _candidate_urls(paper, source)
                if unpaywall_email:
                    oa_url = await _resolve_unpaywall_pdf(
                        client, paper.identifiers.get("doi", ""), unpaywall_email
                    )
                    if oa_url and oa_url not in candidate_urls:
                        candidate_urls.append(oa_url)
                entry["candidate_urls"] = candidate_urls
                if source:
                    entry["source_path"] = source.source_path
                    if source.query:
                        paper.matched_queries.append(source.query)

            attemptable = [entry for entry in entries if entry["candidate_urls"]]
            for entry in entries:
                if not entry["candidate_urls"]:
                    entry["status"] = "no_public_pdf"
                    entry["_paper"].download_status = "no_public_pdf"
                    entry["_paper"].download_error = "No public PDF URL or DOI target was found."
                    entry["error"] = entry["_paper"].download_error
            to_fetch = attemptable[: max(0, limit)]
            for entry in attemptable[max(0, limit) :]:
                entry["status"] = "skipped_limit"
                entry["_paper"].download_status = "skipped_limit"
                entry["_paper"].download_error = f"Download limit reached ({limit})."
                entry["error"] = entry["_paper"].download_error

            # Phase 2: fetch selected papers concurrently.
            semaphore = asyncio.Semaphore(max(1, concurrency))

            async def _worker(entry: dict[str, object]) -> None:
                async with semaphore:
                    paper = entry["_paper"]
                    last_error = ""
                    for candidate_url in entry["candidate_urls"]:
                        try:
                            content = await _fetch_pdf(client, candidate_url, max_bytes=max_bytes)
                            filename = f"{paper.paper_id or 'paper'}_{_safe_stem(paper.title, fallback='untitled')}.pdf"
                            relative_path = f"bib/papers/{filename}"
                            if write_file is None:
                                target = workspace_root / relative_path
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(content)
                            else:
                                write_file(relative_path, content)
                            paper.download_status = "downloaded"
                            paper.downloaded_path = relative_path
                            paper.download_error = ""
                            entry["status"] = "downloaded"
                            entry["relative_path"] = relative_path
                            entry["bytes"] = len(content)
                            return
                        except (httpx.HTTPError, ValueError, OSError) as exc:
                            last_error = str(exc) or exc.__class__.__name__
                    paper.download_status = "invalid_pdf" if "not a PDF" in last_error else "failed"
                    paper.download_error = last_error or "Download failed."
                    entry["status"] = paper.download_status
                    entry["error"] = paper.download_error

            await asyncio.gather(*(_worker(entry) for entry in to_fetch))

    manifest = [{key: value for key, value in entry.items() if not key.startswith("_")} for entry in entries]
    counts: dict[str, int] = {}
    for entry in manifest:
        status = str(entry["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "enabled": enabled,
        "limit": limit,
        "max_mb": max_mb,
        "query_terms": query_terms,
        "download_sources": download_targets_markdown(candidate_sources),
        "counts": counts,
        "papers": manifest,
    }


async def download_public_pdfs_from_sources(
    bundle: LiteratureBundle,
    workspace_root: Path,
    source_refs: list[str],
    *,
    enabled: bool = True,
    limit: int = 30,
    max_mb: int = 50,
    timeout_seconds: float = 60.0,
    unpaywall_email: str = "",
    write_file: Callable[[str, bytes], object] | None = None,
) -> dict[str, object]:
    sources = discover_download_sources(workspace_root, source_refs)
    return await _download_bundle_with_sources(
        bundle,
        workspace_root,
        sources,
        enabled=enabled,
        limit=limit,
        max_mb=max_mb,
        timeout_seconds=timeout_seconds,
        unpaywall_email=unpaywall_email,
        write_file=write_file,
    )


async def _download_bundle_with_sources(
    bundle: LiteratureBundle,
    workspace_root: Path,
    sources: list[DownloadSource],
    *,
    enabled: bool,
    limit: int,
    max_mb: int,
    timeout_seconds: float,
    unpaywall_email: str = "",
    write_file: Callable[[str, bytes], object] | None = None,
) -> dict[str, object]:
    papers_root = workspace_root / "bib" / "papers"
    papers_root.mkdir(parents=True, exist_ok=True)
    max_bytes = max(1, max_mb) * 1024 * 1024
    manifest: list[dict[str, object]] = []
    attempted = 0
    query_terms = extract_download_queries(sources)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": "research-agent-platform"},
    ) as client:
        for paper in bundle.papers:
            entry = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "pdf_url": paper.pdf_url,
                "candidate_urls": [],
                "source_path": "",
                "status": "not_attempted",
                "relative_path": "",
                "bytes": 0,
                "error": "",
            }
            if not enabled:
                entry["status"] = "disabled"
                paper.download_status = "disabled"
                manifest.append(entry)
                continue
            if attempted >= max(0, limit):
                entry["status"] = "skipped_limit"
                paper.download_status = "skipped_limit"
                paper.download_error = f"Download limit reached ({limit})."
                entry["error"] = paper.download_error
                manifest.append(entry)
                continue

            source = _match_source(paper, sources, query_terms)
            candidate_urls = _candidate_urls(paper, source)
            if unpaywall_email:
                oa_url = await _resolve_unpaywall_pdf(client, paper.identifiers.get("doi", ""), unpaywall_email)
                if oa_url and oa_url not in candidate_urls:
                    candidate_urls.append(oa_url)
            entry["candidate_urls"] = candidate_urls
            if source:
                entry["source_path"] = source.source_path
                if source.query:
                    paper.matched_queries.append(source.query)
            if not candidate_urls:
                entry["status"] = "no_public_pdf"
                paper.download_status = "no_public_pdf"
                paper.download_error = "No public PDF URL or DOI target was found."
                entry["error"] = paper.download_error
                manifest.append(entry)
                continue

            attempted += 1
            downloaded = False
            last_error = ""
            for candidate_url in candidate_urls:
                try:
                    content = await _fetch_pdf(client, candidate_url, max_bytes=max_bytes)
                    filename = f"{paper.paper_id or 'paper'}_{_safe_stem(paper.title, fallback='untitled')}.pdf"
                    relative_path = f"bib/papers/{filename}"
                    if write_file is None:
                        target = workspace_root / relative_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                    else:
                        write_file(relative_path, content)
                    paper.download_status = "downloaded"
                    paper.downloaded_path = relative_path
                    paper.download_error = ""
                    entry["status"] = "downloaded"
                    entry["relative_path"] = relative_path
                    entry["bytes"] = len(content)
                    downloaded = True
                    break
                except (httpx.HTTPError, ValueError, OSError) as exc:
                    last_error = str(exc) or exc.__class__.__name__
            if not downloaded:
                paper.download_status = "invalid_pdf" if "not a PDF" in last_error else "failed"
                paper.download_error = last_error or "Download failed."
                entry["status"] = paper.download_status
                entry["error"] = paper.download_error
            manifest.append(entry)

    counts: dict[str, int] = {}
    for entry in manifest:
        status = str(entry["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "enabled": enabled,
        "limit": limit,
        "max_mb": max_mb,
        "query_terms": query_terms,
        "download_sources": download_targets_markdown(sources),
        "counts": counts,
        "papers": manifest,
    }


async def _fetch_pdf(client: httpx.AsyncClient, url: str, *, max_bytes: int) -> bytes:
    if not url:
        raise ValueError("Empty URL.")
    if url.lower().startswith("doi:"):
        url = _doi_to_resolver_url(url[4:])
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("File exceeds size limit.")
        chunks: list[bytes] = []
        total_bytes = 0
        async for chunk in response.aiter_bytes():
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError("File exceeds size limit.")
            chunks.append(chunk)
    content = b"".join(chunks)
    if not content.startswith(b"%PDF"):
        raise ValueError("Response is not a PDF file.")
    return content


def _match_source(
    paper: PaperRecord,
    sources: list[DownloadSource],
    query_terms: list[str],
) -> DownloadSource | None:
    text = f"{paper.title} {paper.abstract} {paper.venue}".casefold()
    doi = _normalize_doi(paper.identifiers.get("doi", ""))
    for source in sources:
        if source.doi and doi and _normalize_doi(source.doi).casefold() == doi.casefold():
            return source
        if source.url and source.url.casefold() in (paper.url.casefold(), paper.pdf_url.casefold()):
            return source
        if source.title and source.title.casefold() in text:
            return source
        if source.query and source.query.casefold() in text:
            return source
    if query_terms:
        joined = " ".join(query_terms).casefold()
        if joined and joined in text:
            return DownloadSource(source_path="", source_type="query", query=joined)
    return None


def _workspace_download_refs(workspace_root: Path) -> list[str]:
    refs: list[str] = []
    for directory_name in ("bib", "paper", "plan", "idea", "Content"):
        directory = workspace_root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".bib", ".enw", ".nbib", ".ris", ".md", ".txt", ".csv", ".tsv", ".json", ".xml", ".html", ".htm", ".pdf"}:
                refs.append(path.relative_to(workspace_root).as_posix())
    return refs
