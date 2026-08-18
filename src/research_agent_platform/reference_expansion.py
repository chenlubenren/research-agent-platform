from __future__ import annotations

import asyncio
import difflib
import hashlib
import ipaddress
import json
import re
import socket
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REFERENCE_HEADING_RE = re.compile(r"(?im)^\s*(?:references|bibliography|参考文献)\s*$")
REFERENCE_START_RE = re.compile(
    r"((?:19|20)\d{2}[a-z]?\.)\s+(?=[^\W\d_][^,]{1,60},\s)"
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
ARXIV_RE = re.compile(
    r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s<>]+", re.I)


@dataclass
class ReferenceRecord:
    reference_id: str
    raw_text: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    source_url: str = ""
    resolved_title: str = ""
    resolved_authors: list[str] = field(default_factory=list)
    resolved_year: int | None = None
    openalex_id: str = ""
    candidate_urls: list[str] = field(default_factory=list)
    status: str = "unresolved"
    source_relative_path: str = ""
    paper_id: str = ""
    reason: str = ""

    @property
    def display_title(self) -> str:
        return self.resolved_title or self.title or self.raw_text[:160]


@dataclass
class ReferenceExpansionResult:
    primary_source: str
    primary_paper_id: str
    extracted_count: int
    records: list[ReferenceRecord]
    manifest_relative_path: str
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_source": self.primary_source,
            "primary_paper_id": self.primary_paper_id,
            "extracted_count": self.extracted_count,
            "records": [asdict(record) for record in self.records],
            "manifest_relative_path": self.manifest_relative_path,
            "cached": self.cached,
        }


def extract_reference_records(pdf_path: str | Path, *, limit: int = 100) -> list[ReferenceRecord]:
    """Extract first-level references from a PDF without treating them as evidence."""

    text = _extract_pdf_text(Path(pdf_path))
    if not text:
        return []
    reference_text = _reference_section(text)
    if not reference_text:
        return []
    raw_entries = _split_reference_entries(reference_text)
    records: list[ReferenceRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries[: max(0, limit)], start=1):
        record = _reference_record(index, raw)
        identity = _reference_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    return records


async def expand_pdf_references(
    source_relative_path: str,
    workspace_root: str | Path,
    *,
    primary_paper_id: str,
    limit: int = 100,
    download_limit: int = 100,
    timeout_seconds: float = 20.0,
    max_pdf_mb: int = 20,
    max_total_mb: int = 50,
    max_total_bytes: int | None = None,
    cache: bool = True,
) -> ReferenceExpansionResult:
    """Resolve and download direct references for one uploaded PDF.

    Only public PDF candidates are downloaded. Metadata-only or failed entries remain
    explicit and are never converted into Wiki evidence.
    """

    workspace = Path(workspace_root).resolve()
    manifest_relative_path = f"wiki/reference-expansions/{primary_paper_id}.json"
    manifest_path = workspace / manifest_relative_path
    source_path = workspace / source_relative_path
    if cache and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return _result_from_dict(payload, cached=True)
        except (OSError, ValueError, TypeError, KeyError):
            pass

    records = extract_reference_records(source_path, limit=limit)
    if not records:
        result = ReferenceExpansionResult(
            primary_source=source_relative_path,
            primary_paper_id=primary_paper_id,
            extracted_count=0,
            records=[],
            manifest_relative_path=manifest_relative_path,
        )
        _write_manifest(manifest_path, result)
        return result

    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout_seconds,
        headers={"User-Agent": "research-agent-platform/0.1"},
    ) as client:
        await asyncio.gather(
            *(
                _resolve_record(record, client=client, semaphore=semaphore)
                for record in records
            )
        )
        downloadable: list[ReferenceRecord] = []
        for record in records:
            if record.status not in {"resolved", "unresolved"}:
                continue
            if len(downloadable) >= max(0, download_limit):
                record.status = "skipped_limit"
                record.reason = f"参考文献下载上限为 {download_limit}。"
                continue
            if not record.candidate_urls:
                record.status = "unavailable"
                record.reason = record.reason or "没有找到可验证的公开 PDF 地址。"
                continue
            downloadable.append(record)

        download_semaphore = asyncio.Semaphore(4)

        async def download_record(record: ReferenceRecord) -> tuple[ReferenceRecord, bytes, str]:
            async with download_semaphore:
                content, error = await _download_first_pdf(
                    client,
                    record.candidate_urls,
                    timeout_seconds=timeout_seconds,
                    max_bytes=max(1, max_pdf_mb) * 1024 * 1024,
                )
            return record, content, error

        download_results = await asyncio.gather(
            *(download_record(record) for record in downloadable)
        )
        known_hashes = _known_pdf_hashes(workspace)
        total_bytes = 0
        total_limit = (
            max(1, max_total_bytes)
            if max_total_bytes is not None
            else max(1, max_total_mb) * 1024 * 1024
        )
        for record, content, error in download_results:
            if not content:
                record.status = "unavailable"
                record.reason = error or "公开 PDF 下载失败。"
                continue
            if total_bytes + len(content) > total_limit:
                record.status = "skipped_total_limit"
                record.reason = f"参考文献总下载上限为 {max_total_mb} MB。"
                continue
            digest = hashlib.sha256(content).hexdigest()
            if digest in known_hashes:
                record.status = "duplicate"
                record.source_relative_path = known_hashes[digest]
                record.reason = "下载内容与已入库 PDF 重复。"
                continue
            filename = f"{record.reference_id}_{_safe_stem(record.display_title)}.pdf"
            relative_path = f"bib/references/{filename}"
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            total_bytes += len(content)
            known_hashes[digest] = relative_path
            record.status = "downloaded"
            record.source_relative_path = relative_path
            record.reason = "已下载并通过 PDF 文件头校验。"

    result = ReferenceExpansionResult(
        primary_source=source_relative_path,
        primary_paper_id=primary_paper_id,
        extracted_count=len(records),
        records=records,
        manifest_relative_path=manifest_relative_path,
    )
    _write_manifest(manifest_path, result)
    return result


def reference_catalog_markdown(results: list[ReferenceExpansionResult]) -> str:
    lines = [
        "# 参考文献扩展目录",
        "",
        "本目录只记录输入论文的一层直接参考文献。只有状态为 `downloaded` 或 `duplicate` 的条目才进入 Wiki 论文证据；其他条目仅保留引用元数据。",
        "",
    ]
    total = sum(len(result.records) for result in results)
    downloaded = sum(
        1
        for result in results
        for record in result.records
        if record.status in {"downloaded", "duplicate"}
    )
    lines.extend(
        [
            f"- 输入论文数: {len(results)}",
            f"- 直接参考文献数: {total}",
            f"- 已进入 Wiki 的参考论文数: {downloaded}",
            "- 说明: 未下载成功的条目不能被 Idea 当作已阅读证据。",
            "",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"## 主论文 `{result.primary_paper_id}`",
                "",
                f"- 输入文件: `{result.primary_source}`",
                f"- 解析数量: {result.extracted_count}",
                "",
                "| 编号 | 论文 | 状态 | Wiki 论文 | 备注 |",
                "|---|---|---|---|---|",
            ]
        )
        for record in result.records:
            paper = f"`{record.paper_id}`" if record.paper_id else "-"
            title = record.display_title.replace("|", "\\|")
            reason = (record.reason or "").replace("|", "\\|")[:160]
            lines.append(
                f"| {record.reference_id} | {title} | `{record.status}` | {paper} | {reason} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        try:
            import fitz

            document = fitz.open(str(path))
            return "\n".join(page.get_text() for page in document)
        except Exception:
            return ""


def _reference_section(text: str) -> str:
    match = REFERENCE_HEADING_RE.search(text)
    if not match:
        return ""
    section = text[match.end() :]
    section = re.split(r"(?im)^\s*(?:appendix|附录|a\.\s+proofs?)\b", section)[0]
    section = re.sub(r"(?m)^\s*Simple and Deep Graph Convolutional Networks\s*$", "", section)
    section = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", section)
    return section


def _split_reference_entries(section: str) -> list[str]:
    normalized = section.replace("\u00ad", "")
    normalized = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    boundaries = list(REFERENCE_START_RE.finditer(normalized))
    if not boundaries:
        return [normalized] if normalized else []
    entries: list[str] = []
    start = 0
    for boundary in boundaries:
        end = boundary.end(1)
        entry = normalized[start:end].strip(" .")
        if entry and not entry.casefold().startswith("references"):
            entries.append(entry)
        start = boundary.end()
    tail = normalized[start:].strip(" .")
    if tail and not tail.casefold().startswith("a. proofs"):
        entries.append(tail)
    return entries


def _reference_record(index: int, raw: str) -> ReferenceRecord:
    clean = _clean_reference_text(raw)
    year_match = list(re.finditer(r"\b((?:19|20)\d{2})[a-z]?\b", clean))
    year = int(year_match[-1].group(1)) if year_match else None
    doi_match = DOI_RE.search(clean)
    arxiv_match = ARXIV_RE.search(clean)
    url_match = URL_RE.search(clean)
    title = _extract_title(clean)
    return ReferenceRecord(
        reference_id=f"R{index:03d}",
        raw_text=clean,
        title=title,
        year=year,
        doi=_normalize_doi(doi_match.group(0)) if doi_match else "",
        arxiv_id=arxiv_match.group(1) if arxiv_match else "",
        source_url=_normalize_url(url_match.group(0)) if url_match else "",
        status="unresolved",
    )


def _extract_title(raw: str) -> str:
    marker = re.search(
        r"\.\s+(?:In(?=\s|[A-Z])\s*|arXiv\b\s*|Technical report\b|The\s+(?:Journal|handbook)\b|Data Science and Engineering\b|AI Magazine\b|ICLR\b|ICML\b|NeurIPS\b|KDD\b|AAAI\b|ACL\b|CVPR\b|UAI\b|JMLR\b)",
        raw,
    )
    year_tail = re.search(r",\s*(?:19|20)\d{2}[a-z]?\s*$", raw)
    title_end = marker.start() if marker else (year_tail.start() if year_tail else len(raw))
    prefix = raw[:title_end].strip(" .")
    starts = list(re.finditer(r"\.\s+(?=[A-Z](?:[a-z]|[A-Z]{2,}))", prefix))
    if starts:
        title = prefix[starts[-1].end() :]
    else:
        title = prefix
    title = re.sub(r"\s+", " ", title).strip(" .,")
    if len(title) < 8:
        return ""
    return title


async def _resolve_record(
    record: ReferenceRecord,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> None:
    direct_urls: list[str] = []
    if record.arxiv_id:
        direct_urls.append(f"https://arxiv.org/pdf/{record.arxiv_id}.pdf")
    if record.source_url:
        direct_urls.append(record.source_url)
    if record.doi:
        direct_urls.append(f"https://doi.org/{record.doi}")
    metadata = await _resolve_openalex_reference(record, client=client, semaphore=semaphore)
    if metadata:
        record.resolved_title = str(metadata.get("title") or "")
        record.resolved_authors = [str(item) for item in metadata.get("authors", []) if item]
        record.resolved_year = metadata.get("year")
        record.openalex_id = str(metadata.get("openalex_id") or "")
        record.candidate_urls.extend(
            str(url)
            for url in metadata.get("candidate_urls", [])
            if isinstance(url, str) and url
        )
    record.candidate_urls = _unique_urls([*direct_urls, *record.candidate_urls])
    if record.candidate_urls:
        record.status = "resolved"
        record.reason = "已找到候选公开来源，等待 PDF 校验。"
    else:
        record.status = "unresolved"
        record.reason = "未解析到公开来源。"


async def _resolve_openalex_reference(
    record: ReferenceRecord,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    query = record.doi or record.title or record.raw_text
    if not query:
        return {}
    params: dict[str, str] = {
        "per-page": "5",
        "select": "id,display_name,publication_year,doi,authorships,best_oa_location,primary_location",
    }
    if record.doi:
        params["filter"] = f"doi:{record.doi}"
    else:
        params["search"] = query[:240]
    try:
        async with semaphore:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        return {}
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return {}
    best: tuple[float, dict[str, Any]] | None = None
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("display_name") or "")
        score = _title_similarity(record.title or record.raw_text, title)
        year = item.get("publication_year")
        if record.year and isinstance(year, int):
            score += max(0.0, 0.15 - abs(record.year - year) * 0.03)
        if record.doi and _normalize_doi(str(item.get("doi") or "")) == record.doi:
            score += 1.0
        if best is None or score > best[0]:
            best = (score, item)
    if best is None or (not record.doi and best[0] < 0.8):
        return {}
    item = best[1]
    locations = [item.get("best_oa_location"), item.get("primary_location")]
    candidate_urls = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        pdf_url = location.get("pdf_url")
        landing_url = location.get("landing_page_url")
        if isinstance(pdf_url, str) and pdf_url:
            candidate_urls.append(pdf_url)
        if isinstance(landing_url, str) and landing_url:
            candidate_urls.append(landing_url)
    authorships = item.get("authorships") or []
    authors = [
        str(author.get("author", {}).get("display_name"))
        for author in authorships
        if isinstance(author, dict) and isinstance(author.get("author"), dict)
    ]
    return {
        "openalex_id": item.get("id", ""),
        "title": item.get("display_name", ""),
        "year": item.get("publication_year"),
        "authors": authors,
        "candidate_urls": _unique_urls(candidate_urls),
    }


async def _download_first_pdf(
    client: httpx.AsyncClient,
    urls: list[str],
    *,
    timeout_seconds: float,
    max_bytes: int,
) -> tuple[bytes, str]:
    last_error = ""
    for url in urls:
        try:
            current_url = url
            for _ in range(6):
                safe, reason = await _is_safe_public_url(current_url)
                if not safe:
                    last_error = reason
                    break
                async with client.stream("GET", current_url, timeout=timeout_seconds) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location", "")
                        if not location:
                            last_error = "重定向响应缺少 Location。"
                            break
                        current_url = urllib.parse.urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > max_bytes:
                        last_error = "PDF 响应超过单文件大小上限。"
                        break
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            last_error = "PDF 下载超过单文件大小上限。"
                            break
                        chunks.append(chunk)
                    else:
                        content = b"".join(chunks)
                        if not content.startswith(b"%PDF"):
                            last_error = "响应不是 PDF。"
                            break
                        return content, ""
                    break
        except (httpx.HTTPError, ValueError, OSError) as exc:
            last_error = str(exc) or exc.__class__.__name__
    return b"", last_error or "没有可用下载地址。"


async def _is_safe_public_url(value: str) -> tuple[bool, str]:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return False, "URL 无法解析。"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False, "只允许公网 HTTP/HTTPS URL。"
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False, "禁止访问本机地址。"
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False, "URL 主机无法解析。"
    if not addresses:
        return False, "URL 主机没有可用地址。"
    for item in addresses:
        address = ipaddress.ip_address(item[4][0].split("%")[0])
        if not address.is_global:
            return False, "禁止访问内网、回环、链路本地或保留地址。"
    return True, ""


def _known_pdf_hashes(workspace: Path) -> dict[str, str]:
    known: dict[str, str] = {}
    for path in [*workspace.glob("*/uploads/*.pdf"), *workspace.glob("bib/**/*.pdf")]:
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            known[digest] = path.relative_to(workspace).as_posix()
        except OSError:
            continue
    return known


def _write_manifest(path: Path, result: ReferenceExpansionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _result_from_dict(payload: dict[str, Any], *, cached: bool) -> ReferenceExpansionResult:
    records = [ReferenceRecord(**record) for record in payload.get("records", [])]
    return ReferenceExpansionResult(
        primary_source=str(payload.get("primary_source", "")),
        primary_paper_id=str(payload.get("primary_paper_id", "")),
        extracted_count=int(payload.get("extracted_count", len(records))),
        records=records,
        manifest_relative_path=str(payload.get("manifest_relative_path", "")),
        cached=cached,
    )


def _reference_identity(record: ReferenceRecord) -> str:
    return record.doi.casefold() or record.arxiv_id.casefold() or _normalize_title(record.title or record.raw_text)


def _title_similarity(left: str, right: str) -> float:
    left_norm = _normalize_title(left)
    right_norm = _normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def _normalize_title(value: str) -> str:
    value = value.casefold().replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def _clean_reference_text(value: str) -> str:
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def _normalize_doi(value: str) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).rstrip(".,;)")


def _normalize_url(value: str) -> str:
    return value.strip().rstrip(".,;)")


def _unique_urls(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", value).strip(" ._")
    return (value or "untitled")[:100]
