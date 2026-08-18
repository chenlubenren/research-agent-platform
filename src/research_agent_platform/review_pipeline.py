from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .connectors.scholar import LiteratureBundle, PaperRecord
from .config import config


class ReviewEvidenceError(RuntimeError):
    pass


def clean_review_topic(objective: str) -> str:
    topic = re.sub(r"^\s*/review\b", "", objective, flags=re.I).strip()
    topic = re.sub(r"--[\w-]+(?:=|\s+)\S+", "", topic).strip()
    topic = re.sub(
        r"^(?:请|请帮我|帮我)?(?:写|做|整理|生成|调研|查找|检索)(?:一个|一份|一下)?(?:当前|最新)?",
        "",
        topic,
    ).strip()
    topic = re.sub(
        r"(?:相关的?|领域的?|方向的?)?(?:文献)?(?:综述|研究综述|研究进展|进展与展望|survey|review)\s*$",
        "",
        topic,
        flags=re.I,
    ).strip(" ，,。.;；")
    return topic or objective.strip()


def parse_review_queries(objective: str, research_brief: str) -> list[str]:
    topic = clean_review_topic(objective)
    candidates = [topic]
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"(?mi)^\s*(?:[-*+]\s*)?Q\d+\s*[:：]\s*[`\"']?(.+?)[`\"']?\s*$",
            research_brief,
        )
    )
    if re.search(r"[A-Za-z]", topic):
        candidates.extend([f"{topic} review", f"{topic} systematic review"])
    return _deduplicate(candidates)[:6]


def build_frozen_local_corpus_bundle(
    query: str,
    workspace_root: Path,
    source_refs: list[str],
    *,
    queries: list[str],
) -> LiteratureBundle:
    metadata = _local_corpus_metadata(workspace_root, source_refs)
    evidence_by_paper: dict[str, list[dict]] = {}
    for relative in source_refs:
        path = workspace_root / relative
        if path.suffix.lower() != ".jsonl" or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            paper_id = str(record.get("paper_id") or "").strip()
            text = str(record.get("text") or "").strip()
            if paper_id and text:
                evidence_by_paper.setdefault(paper_id, []).append(record)

    papers: list[PaperRecord] = []
    for index, (source_id, evidence) in enumerate(evidence_by_paper.items(), start=1):
        item = metadata.get(source_id, {})
        papers.append(
            PaperRecord(
                paper_id=f"P{index:03d}",
                title=str(item.get("title") or source_id),
                year=_as_year(item.get("year")),
                abstract=" ".join(str(record.get("text") or "").strip() for record in evidence[:3])[:1800],
                authors=_as_authors(item.get("authors")),
                url=str(item.get("source_url") or item.get("url") or ""),
                venue=str(item.get("venue") or item.get("status") or ""),
                citation_count=None,
                sources=["frozen_local_corpus"],
                identifiers={
                    key: value
                    for key, value in {
                        "source_paper_id": source_id,
                        "bib_key": str(evidence[0].get("bib_key") or ""),
                        "doi": str(item.get("doi") or ""),
                    }.items()
                    if value
                },
                relevance_score=1.0,
                matched_queries=queries or [query],
                verification_status="local_full_text_evidence",
            )
        )
    count = len(papers)
    return LiteratureBundle(
        query=query,
        queries=queries or [query],
        papers=papers,
        provider_status={"local_corpus": "frozen user-provided evidence"},
        quality={
            "status": "adequate" if count >= max(1, config.review_minimum_sources) else "insufficient",
            "candidate_count": count,
            "relevant_count": count,
            "traceable_count": count,
            "minimum_for_synthesis": config.review_minimum_sources,
            "source_mode": "frozen_local_corpus",
        },
    )


def _local_corpus_metadata(workspace_root: Path, source_refs: list[str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for relative in source_refs:
        path = workspace_root / relative
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                records = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict) and record.get("paper_id"):
                        metadata[str(record["paper_id"])] = record
        elif path.suffix.lower() == ".csv" and path.is_file():
            with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle), start=1):
                    metadata.setdefault(f"paper-{index:03d}", row)
    return metadata


def _as_year(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_authors(value: object) -> list[str]:
    if not value:
        return []
    return [author.strip() for author in str(value).split(";") if author.strip()]


def review_quality_markdown(
    bundle: LiteratureBundle,
    *,
    local_sources: list[str],
    local_candidates: list[str] | None = None,
) -> str:
    quality = bundle.quality
    minimum_sources = max(1, config.review_minimum_sources)
    total_sources = int(bundle.quality.get("relevant_count", len(bundle.papers))) + len(local_sources)
    traceable_sources = int(bundle.quality.get("traceable_count", 0)) + len(local_sources)
    enough = review_evidence_is_sufficient(bundle, local_source_count=len(local_sources))
    lines = [
        "# Literature Retrieval Quality Report",
        "",
        f"- Topic: {bundle.query}",
        f"- Gate: {'PASS' if enough else 'FAIL'}",
        f"- Required minimum sources: {minimum_sources}",
        f"- Admitted sources: {total_sources}",
        f"- Traceable sources: {traceable_sources}",
        f"- Retrieval status: {quality.get('status', 'unknown')}",
        f"- Candidate records: {quality.get('candidate_count', 0)}",
        f"- Relevant records before cap: {quality.get('eligible_count', quality.get('relevant_count', 0))}",
        f"- Relevant records retained: {quality.get('relevant_count', 0)}",
        f"- Relevant records omitted by cap: {quality.get('truncated_count', 0)}",
        f"- Irrelevant records excluded: {bundle.excluded_count}",
        f"- Traceable records: {quality.get('traceable_count', 0)}",
        f"- Local candidates discovered: {len(local_candidates or local_sources)}",
        f"- Local sources with extracted evidence: {len(local_sources)}",
        "",
        "## Query Plan",
    ]
    lines.extend(f"- {query}" for query in bundle.queries)
    lines.extend(["", "## Provider Coverage"])
    lines.extend(f"- {provider}: {status}" for provider, status in bundle.provider_status.items())
    lines.extend(["", "## Local Sources"])
    lines.extend(f"- `{source}`" for source in local_sources)
    if not local_sources:
        lines.append("- None")
    lines.extend(["", "## Gate Decision"])
    if enough:
        lines.append("- Evidence is sufficient for a traceable synthesis. Coverage limitations must still be disclosed.")
    else:
        lines.append(
            "- Evidence is insufficient for a formal review. Do not generate a literature synthesis, evidence map, or research-gap claim from this retrieval."
        )
        lines.append("- Add local papers or restore another scholarly provider, then rerun `/review`.")
    lines.extend(["", query_effectiveness_markdown(query_effectiveness_payload(bundle)).rstrip()])
    return "\n".join(lines) + "\n"


def query_effectiveness_payload(bundle: LiteratureBundle) -> dict:
    slots: dict[str, dict] = {
        query: {"query": query, "retained": 0, "providers": set(), "paper_ids": []}
        for query in bundle.queries
    }
    for paper in bundle.papers:
        for query in paper.matched_queries or ["(unmatched)"]:
            slot = slots.setdefault(
                query,
                {"query": query, "retained": 0, "providers": set(), "paper_ids": []},
            )
            slot["retained"] += 1
            slot["providers"].update(paper.sources)
            slot["paper_ids"].append(paper.paper_id)
    total = max(1, len(bundle.papers))
    rows = []
    for slot in slots.values():
        rows.append(
            {
                "query": slot["query"],
                "retained": slot["retained"],
                "retained_share": round(slot["retained"] / total, 3),
                "providers": sorted(slot["providers"]),
                "paper_ids": list(dict.fromkeys(slot["paper_ids"])),
            }
        )
    rows.sort(key=lambda item: (item["retained"], item["query"]), reverse=True)
    return {
        "candidate_count": int(bundle.quality.get("candidate_count", len(bundle.papers))),
        "retained_count": len(bundle.papers),
        "excluded_count": bundle.excluded_count,
        "per_query": rows,
        "note": "Per-query values count retained records matched to each query; one paper may match multiple queries.",
    }


def query_effectiveness_markdown(metrics: dict) -> str:
    lines = [
        "## Query Effectiveness",
        "",
        f"- Candidate records: {metrics.get('candidate_count', 0)}",
        f"- Retained records: {metrics.get('retained_count', 0)}",
        f"- Excluded records: {metrics.get('excluded_count', 0)}",
        "",
        "| Query | retained | share | providers |",
        "|---|---:|---:|---|",
    ]
    for row in metrics.get("per_query", []):
        providers = ", ".join(row.get("providers", [])) or "-"
        lines.append(
            f"| {row['query']} | {row['retained']} | {row['retained_share']} | {providers} |"
        )
    return "\n".join(lines) + "\n"


def review_directions_markdown(bundle: LiteratureBundle, *, maximum: int = 5) -> str:
    metrics = query_effectiveness_payload(bundle)
    lines = [
        "# Retrieved Research Directions",
        "",
        "These directions are grounded in the retained papers and use retrieval facets as a deterministic fallback; they are not independent novelty claims.",
        "",
    ]
    populated = [row for row in metrics["per_query"] if row["paper_ids"]][: max(1, maximum)]
    if not populated:
        return "\n".join(lines + ["- No retained direction could be formed.", ""])
    for index, row in enumerate(populated, start=1):
        lines.extend(
            [
                f"## D{index}: {row['query']}",
                "",
                f"- Retained papers: {row['retained']}",
                "- Representative IDs: "
                + ", ".join(f"`{paper_id}`" for paper_id in row["paper_ids"][:5]),
                "- Boundary: inspect titles/abstracts/full text before treating this facet as a coherent research theme.",
                "",
            ]
        )
    return "\n".join(lines)


def review_evidence_is_sufficient(
    bundle: LiteratureBundle,
    *,
    local_source_count: int = 0,
    minimum_sources: int | None = None,
) -> bool:
    relevant_count = int(bundle.quality.get("relevant_count", len(bundle.papers)))
    traceable_count = int(bundle.quality.get("traceable_count", 0))
    required = max(1, minimum_sources if minimum_sources is not None else config.review_minimum_sources)
    return relevant_count + local_source_count >= required and traceable_count + local_source_count >= required


def build_review_quality_reports(workspace_root: Path) -> tuple[dict, dict]:
    bundle_path = workspace_root / "bib" / "LITERATURE_SEARCH.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8")) if bundle_path.exists() else {}
    papers = payload.get("papers") if isinstance(payload.get("papers"), list) else []
    known_ids = {str(paper.get("paper_id", "")) for paper in papers if paper.get("paper_id")}
    traceable_ids = {
        str(paper.get("paper_id", ""))
        for paper in papers
        if paper.get("paper_id") and paper.get("verification_status") == "traceable_identifier"
    }
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    local_sources = {
        str(source)
        for source in quality.get("local_evidence_sources", [])
        if isinstance(source, str) and source
    }
    file_citations: dict[str, list[str]] = {}
    file_reference_entries: dict[str, list[str]] = {}
    file_local_citations: dict[str, list[str]] = {}
    review_text = ""
    for relative in (
        "bib/LITERATURE_REVIEW.md",
        "bib/EVIDENCE_MAP.md",
        "bib/RESEARCH_GAPS.md",
    ):
        path = workspace_root / relative
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        body, references = _split_markdown_references(text)
        if relative == "bib/LITERATURE_REVIEW.md":
            review_text = body
        file_citations[relative] = sorted(_extract_stable_citation_ids(body, known_ids))
        file_reference_entries[relative] = sorted(_extract_stable_citation_ids(references, known_ids))
        file_local_citations[relative] = sorted(source for source in local_sources if source in body)
    used_ids = set(identifier for citations in file_citations.values() for identifier in citations)
    used_local_sources = set(source for citations in file_local_citations.values() for source in citations)
    unknown_ids = sorted(used_ids - known_ids)
    uncited_ids = sorted(known_ids - used_ids)
    claim_coverage = _assess_review_claim_citation_coverage(review_text, known_ids, local_sources)
    source_count = len(known_ids) + len(local_sources)
    cited_source_count = len(used_ids & known_ids) + len(used_local_sources)
    traceable_source_count = len(traceable_ids) + len(local_sources)
    coverage_ratio = round(cited_source_count / source_count, 3) if source_count else 0.0
    traceable_ratio = round(traceable_source_count / source_count, 3) if source_count else 0.0
    resolution_status = "pass" if (known_ids or local_sources) and not unknown_ids else "needs_attention"
    corpus_coverage_status = "pass" if source_count and coverage_ratio >= 0.4 else "needs_attention"
    citation_audit = {
        "known_paper_ids": sorted(known_ids),
        "file_citations": file_citations,
        "file_reference_entries": file_reference_entries,
        "known_local_sources": sorted(local_sources),
        "file_local_citations": file_local_citations,
        "unknown_paper_ids": unknown_ids,
        "uncited_paper_ids": uncited_ids,
        "resolution_status": resolution_status,
        "corpus_coverage_status": corpus_coverage_status,
        "claim_coverage": claim_coverage,
        "semantic_support_status": "not_deterministically_verified",
        "status": (
            "pass"
            if resolution_status == "pass"
            and corpus_coverage_status == "pass"
            and claim_coverage["status"] == "pass"
            else "needs_attention"
        ),
        "note": (
            "Resolution checks whether stable IDs exist; claim coverage checks whether likely citable claims have "
            "paragraph-local citations; corpus coverage checks admitted-source use. Semantic support strength "
            "(strong, partial, background, or limiting) still requires evidence-aware review."
        ),
    }
    coverage_report = {
        "retrieval_quality": quality,
        "paper_count": source_count,
        "external_paper_count": len(known_ids),
        "local_source_count": len(local_sources),
        "cited_paper_count": cited_source_count,
        "reference_coverage": coverage_ratio,
        "traceable_paper_ratio": traceable_ratio,
        "provider_status": payload.get("provider_status", {}),
        "status": (
            "pass"
            if source_count >= max(1, config.review_minimum_sources)
            and not unknown_ids
            and coverage_ratio >= 0.4
            and traceable_ratio >= 0.8
            else "needs_attention"
        ),
        "limitations": [
            "Metadata and abstracts do not replace full-text verification.",
            "Coverage reflects configured providers and local uploads only.",
        ],
    }
    return citation_audit, coverage_report


def _split_markdown_references(text: str) -> tuple[str, str]:
    match = re.search(r"(?im)^#{1,6}\s+(?:references|bibliography|参考文献)\s*$", text)
    if match is None:
        return text, ""
    return text[: match.start()], text[match.end() :]


def _extract_stable_citation_ids(text: str, known_ids: set[str]) -> set[str]:
    cited = {identifier for identifier in known_ids if re.search(rf"(?<![\w-]){re.escape(identifier)}(?![\w-])", text)}
    cited.update(
        re.findall(r"(?<![\w-])([A-Za-z][A-Za-z0-9_-]*\d{2,})(?![\w-])", text)
    )
    return cited


def _assess_review_claim_citation_coverage(
    text: str,
    known_ids: set[str],
    local_sources: set[str],
) -> dict:
    claim_cues = re.compile(
        r"(?:\b(?:report(?:s|ed)?|find(?:s|ings)?|found|show(?:s|ed)?|demonstrat(?:e|es|ed)|"
        r"yield(?:s|ed)?|outperform(?:s|ed)?|associat(?:e|es|ed)|increas(?:e|es|ed)|"
        r"decreas(?:e|es|ed)|accuracy|macro-f1|gpu-hours?|seeds?|records?|studies|participants?|patients?)\b|%)",
        re.I,
    )
    units: list[dict[str, str | int]] = []
    for block_index, block in enumerate(re.split(r"\n\s*\n", text), start=1):
        compact = " ".join(line.strip() for line in block.splitlines() if not line.lstrip().startswith("#"))
        if not compact or not claim_cues.search(compact):
            continue
        citations = sorted(_extract_stable_citation_ids(compact, known_ids))
        local = sorted(source for source in local_sources if source in compact)
        units.append(
            {
                "block": block_index,
                "excerpt": compact[:240],
                "citation_count": len(citations) + len(local),
            }
        )
    uncited = [unit for unit in units if unit["citation_count"] == 0]
    return {
        "status": "pass" if units and not uncited else "needs_attention",
        "citable_claim_unit_count": len(units),
        "cited_claim_unit_count": len(units) - len(uncited),
        "uncited_claim_count": len(uncited),
        "uncited_claims": uncited,
        "method": "Conservative paragraph-local heuristic; semantic entailment is not inferred.",
    }


def _deduplicate(values: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split()).strip(" `\"'，,。.;；")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            results.append(cleaned)
            seen.add(key)
    return results
