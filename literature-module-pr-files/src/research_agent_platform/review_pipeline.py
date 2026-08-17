from __future__ import annotations

import json
import re
from pathlib import Path

from .connectors.scholar import LiteratureBundle
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
    if re.search(r"[\u4e00-\u9fff]", topic):
        candidates.extend([f"{topic} 综述", f"{topic} 研究进展"])
    limit = max(1, config.review_query_limit)
    return _deduplicate(candidates)[:limit]


# Per-domain "key" indexes; if these fail, coverage is degraded even when the gate passes.
DOMAIN_KEY_SOURCES: dict[str, list[str]] = {
    "cs": ["arxiv", "dblp", "semantic_scholar", "openalex", "crossref"],
    "bio": ["pubmed", "europepmc", "pmc", "openalex", "crossref"],
    "med": ["pubmed", "europepmc", "pmc", "openalex", "crossref"],
    "": ["openalex", "crossref", "semantic_scholar"],
}


def _reports_zero_records(token: str) -> bool:
    return bool(re.search(r"\b0\s+records\b", token)) or "(0)" in token


def _provider_is_healthy(status: str) -> bool:
    token = str(status or "").strip().casefold()
    if not token or not token.startswith("ok"):
        return False
    return not _reports_zero_records(token)


def _provider_is_failing(status: str) -> bool:
    token = str(status or "").strip().casefold()
    if not token:
        return True
    if "disabled" in token or "not configured" in token or "not attempted" in token or "inactive" in token:
        return False  # intentionally off / not run: not a health failure
    if token.startswith("ok"):
        return _reports_zero_records(token)
    return True  # error / timeout / failed / unexpected


def source_health(provider_status: dict[str, str] | None, domain: str = "") -> dict:
    """Assess coverage for a domain's key indexes.

    Coverage is 'degraded' when a key index is present but failing (e.g. DBLP=error,
    Semantic Scholar=0 records) or when no key index is healthy. Missing (never-run)
    sources are reported but do not, by themselves, degrade coverage.
    """

    key_sources = DOMAIN_KEY_SOURCES.get((domain or "").strip().casefold(), DOMAIN_KEY_SOURCES[""])
    status_map = {str(name).strip().casefold(): str(value) for name, value in (provider_status or {}).items()}
    healthy: list[str] = []
    degraded: list[str] = []
    missing: list[str] = []
    for source in key_sources:
        status = status_map.get(source)
        if status is None:
            missing.append(source)
        elif _provider_is_healthy(status):
            healthy.append(source)
        elif _provider_is_failing(status):
            degraded.append(source)
    coverage_state = "degraded" if (degraded or not healthy) else "ok"
    return {
        "domain": domain or "general",
        "key_sources": key_sources,
        "healthy_sources": healthy,
        "degraded_sources": degraded,
        "missing_sources": missing,
        "coverage_state": coverage_state,
    }


def preprint_ratio(bundle: LiteratureBundle) -> float:
    papers = bundle.papers
    if not papers:
        return 0.0
    preprints = 0
    for paper in papers:
        blob = f"{' '.join(paper.sources)} {paper.venue}".casefold()
        if "arxiv" in blob or "preprint" in blob or "biorxiv" in blob or "ssrn" in blob:
            preprints += 1
    return round(preprints / len(papers), 3)


def review_quality_markdown(
    bundle: LiteratureBundle,
    *,
    local_sources: list[str],
    local_candidates: list[str] | None = None,
    query_metrics: dict | None = None,
) -> str:
    quality = bundle.quality
    minimum_sources = max(1, config.review_minimum_sources)
    total_sources = int(bundle.quality.get("relevant_count", len(bundle.papers))) + len(local_sources)
    traceable_sources = int(bundle.quality.get("traceable_count", 0)) + len(local_sources)
    enough = review_evidence_is_sufficient(bundle, local_source_count=len(local_sources))
    health = quality.get("source_health") if isinstance(quality.get("source_health"), dict) else None
    coverage_state = str(quality.get("coverage_state", (health or {}).get("coverage_state", "unknown")))
    if enough and coverage_state == "degraded":
        gate_label = "PASS (retrieval_pass_with_degraded_coverage)"
    elif enough:
        gate_label = "PASS"
    else:
        gate_label = "FAIL"
    lines = [
        "# Literature Retrieval Quality Report",
        "",
        f"- Topic: {bundle.query}",
        f"- Gate: {gate_label}",
        f"- Coverage state: {coverage_state}",
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
    if health:
        lines.extend(
            [
                "",
                "## Coverage State",
                f"- State: {coverage_state}",
                f"- Domain: {health.get('domain', 'general')}",
                f"- Key sources: {', '.join(health.get('key_sources', [])) or 'none'}",
                f"- Healthy key sources: {', '.join(health.get('healthy_sources', [])) or 'none'}",
                f"- Degraded key sources: {', '.join(health.get('degraded_sources', [])) or 'none'}",
                f"- Missing key sources: {', '.join(health.get('missing_sources', [])) or 'none'}",
                f"- Preprint ratio: {quality.get('preprint_ratio', 0.0)}",
            ]
        )
        if coverage_state == "degraded":
            lines.append(
                "- Note: 关键索引失效或空返回，'traceable' 覆盖率不能等同于领域覆盖充分；综述须显式声明该局限。"
            )
    tiers = quality.get("screen_tiers") if isinstance(quality.get("screen_tiers"), dict) else None
    if tiers:
        lines.extend(
            [
                "",
                "## Screening Tiers",
                f"- Core (counted): {tiers.get('core', 0)}",
                f"- Transferable (counted): {tiers.get('transferable', 0)}",
                f"- Background (recorded, not counted): {tiers.get('background', 0)}",
                f"- Uncertain (queued, not counted): {tiers.get('uncertain', 0)}",
                f"- Semantic rerank mode: {quality.get('rerank_mode') or 'lexical'}",
            ]
        )
    screen_error = str(quality.get("screen_error", "") or "").strip()
    if screen_error:
        lines.extend(["", "## Screening Degraded", f"- screening degraded: {screen_error}"])
    lines.extend(["", "## Provider Coverage"])
    lines.extend(f"- {provider}: {status}" for provider, status in bundle.provider_status.items())
    lines.extend(["", "## Local Sources"])
    lines.extend(f"- `{source}`" for source in local_sources)
    if not local_sources:
        lines.append("- None")
    if query_metrics:
        from .review_metrics import query_effectiveness_markdown

        lines.extend(["", query_effectiveness_markdown(query_metrics).rstrip()])
    if any(getattr(paper, "screen_decision", "") for paper in bundle.papers):
        from .review_screening import screening_markdown

        lines.extend(["", screening_markdown(bundle.papers).rstrip()])
    lines.extend(["", "## Gate Decision"])
    if enough:
        lines.append("- Evidence is sufficient for a traceable synthesis. Coverage limitations must still be disclosed.")
    else:
        lines.append(
            "- Evidence is insufficient for a formal review. Do not generate a literature synthesis, evidence map, or research-gap claim from this retrieval."
        )
        lines.append("- Add local papers or restore another scholarly provider, then rerun `/review`.")
    return "\n".join(lines) + "\n"


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
    file_local_citations: dict[str, list[str]] = {}
    for relative in (
        "bib/LITERATURE_REVIEW.md",
        "bib/EVIDENCE_MAP.md",
        "bib/RESEARCH_GAPS.md",
    ):
        path = workspace_root / relative
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        file_citations[relative] = sorted(set(re.findall(r"\bP\d{3}\b", text)))
        file_local_citations[relative] = sorted(source for source in local_sources if source in text)
    used_ids = set(identifier for citations in file_citations.values() for identifier in citations)
    used_local_sources = set(source for citations in file_local_citations.values() for source in citations)
    unknown_ids = sorted(used_ids - known_ids)
    uncited_ids = sorted(known_ids - used_ids)
    citation_audit = {
        "known_paper_ids": sorted(known_ids),
        "file_citations": file_citations,
        "known_local_sources": sorted(local_sources),
        "file_local_citations": file_local_citations,
        "unknown_paper_ids": unknown_ids,
        "uncited_paper_ids": uncited_ids,
        "status": "pass" if (known_ids or local_sources) and not unknown_ids else "needs_attention",
        "note": "ID resolution is deterministic; semantic claim support still requires author verification.",
    }
    source_count = len(known_ids) + len(local_sources)
    cited_source_count = len(used_ids & known_ids) + len(used_local_sources)
    traceable_source_count = len(traceable_ids) + len(local_sources)
    coverage_ratio = round(cited_source_count / source_count, 3) if source_count else 0.0
    traceable_ratio = round(traceable_source_count / source_count, 3) if source_count else 0.0
    health = quality.get("source_health") if isinstance(quality.get("source_health"), dict) else {}
    coverage_state = str(quality.get("coverage_state", health.get("coverage_state", "unknown")))
    coverage_report = {
        "retrieval_quality": quality,
        "coverage_state": coverage_state,
        "degraded_sources": health.get("degraded_sources", []),
        "missing_key_sources": health.get("missing_sources", []),
        "preprint_ratio": quality.get("preprint_ratio", 0.0),
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


def evidence_sufficiency_gate(
    papers: list[dict],
    *,
    directions: list[dict] | None = None,
    selected_ids: list[str] | None = None,
    coverage_state: str = "unknown",
    min_core: int = 8,
) -> dict:
    """Decide whether the evidence base supports 'likely' research-gap claims (P5).

    Requires enough core evidence, coverage of each selected subtopic, and non-degraded
    (or explicitly disclosed) source coverage. When unmet, callers should downgrade any
    "领域空白" claim to a "candidate gap / insufficient evidence" statement.
    """

    def _counted(paper: dict) -> bool:
        return paper.get("screen_decision") == "keep" and paper.get("screen_tier") in {"core", "transferable"}

    def _core(paper: dict) -> bool:
        return paper.get("screen_decision") == "keep" and paper.get("screen_tier") == "core"

    screened = any(paper.get("screen_decision") for paper in papers)
    core_count = sum(1 for paper in papers if _core(paper))
    counted_count = sum(1 for paper in papers if _counted(paper))
    reasons: list[str] = []

    # When screening never ran we cannot claim tier-based sufficiency; fall back to counted set.
    effective_core = core_count if screened else counted_count
    if effective_core < max(1, min_core):
        reasons.append(f"核心证据仅 {effective_core} 篇，低于阈值 {min_core}")

    uncovered: list[str] = []
    if directions:
        counted_ids = {str(paper.get("paper_id")) for paper in papers if _counted(paper)}
        selected = set(selected_ids or [])
        for direction in directions:
            did = str(direction.get("id", ""))
            if selected and did not in selected:
                continue
            candidate_ids = {str(pid) for pid in direction.get("candidate_paper_ids", [])}
            if candidate_ids and not (candidate_ids & counted_ids):
                uncovered.append(did)
        if uncovered:
            reasons.append("以下选定方向缺乏被采纳(core/transferable)证据: " + ", ".join(uncovered))

    if coverage_state == "degraded":
        reasons.append("关键索引覆盖降级，研究空白须显式声明该限制")

    sufficient = not reasons
    return {
        "sufficient": sufficient,
        "gap_label": "likely" if sufficient else "candidate",
        "core_count": core_count,
        "counted_count": counted_count,
        "min_core": min_core,
        "uncovered_directions": uncovered,
        "coverage_state": coverage_state,
        "screened": screened,
        "reasons": reasons,
    }


def evidence_sufficiency_directive(gate: dict) -> str:
    """Prompt directive telling the gap stage how strong its claims may be."""

    if gate.get("sufficient"):
        return (
            "证据充分性门：PASS。core 证据充足且选定方向均有被采纳证据，可输出 'likely research gap'，"
            "但仍需对每条空白给出支持证据 ID。\n\n"
        )
    reasons = "；".join(gate.get("reasons", [])) or "证据不足"
    return (
        "证据充分性门：FAIL（" + reasons + "）。只能输出 'candidate gap / insufficient evidence'，"
        "严禁把'未检索到'升级为'领域空白/首创'。请显式标注证据不足与覆盖限制，并建议补充检索方向。\n\n"
    )


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
