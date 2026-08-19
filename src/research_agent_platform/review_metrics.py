"""Query-level and screening-level observability for the /review pipeline.

These metrics separate *retrieval* quality (how well queries recall relevant work)
from *screening* quality (how precisely the LLM keeps the on-scope subset):

  - retrieval_precision_before_screening = kept / candidate
  - screening_precision_after           = counted(core+transferable) / kept
  - per-query: unique_recall -> kept -> counted -> core, and Query Yield = counted / unique_recall

The report is written to ``bib/QUERY_YIELD.json`` and rendered into a
``## Query Effectiveness`` section of ``RETRIEVAL_QUALITY.md`` so a follow-up round
can up-weight high-yield queries and prune low-yield ones.
"""

from __future__ import annotations

from .connectors.scholar import LiteratureBundle, PaperRecord

_COUNTED_TIERS = {"core", "transferable"}


def _is_kept(paper: PaperRecord) -> bool:
    return paper.screen_decision != "drop" and paper.screen_tier != "exclude"


def _is_counted(paper: PaperRecord) -> bool:
    return paper.screen_decision == "keep" and paper.screen_tier in _COUNTED_TIERS


def _is_core(paper: PaperRecord) -> bool:
    return paper.screen_decision == "keep" and paper.screen_tier == "core"


def compute_review_metrics(bundle: LiteratureBundle) -> dict:
    papers = list(bundle.papers)
    candidate = len(papers)
    kept = [p for p in papers if _is_kept(p)]
    counted = [p for p in papers if _is_counted(p)]
    core = [p for p in papers if _is_core(p)]

    slots: dict[str, dict] = {}
    for paper in papers:
        matched = paper.matched_queries or ["(unmatched)"]
        for query in matched:
            slot = slots.setdefault(
                query,
                {"unique_recall": 0, "kept": 0, "counted": 0, "core": 0, "providers": set(), "core_paper_ids": []},
            )
            slot["unique_recall"] += 1
            if _is_kept(paper):
                slot["kept"] += 1
            if _is_counted(paper):
                slot["counted"] += 1
            if _is_core(paper):
                slot["core"] += 1
                slot["core_paper_ids"].append(paper.paper_id)
            slot["providers"].update(paper.sources)

    per_query = []
    for query, slot in slots.items():
        recall = slot["unique_recall"]
        per_query.append(
            {
                "query": query,
                "unique_recall": recall,
                "kept": slot["kept"],
                "counted": slot["counted"],
                "core": slot["core"],
                "query_yield": round(slot["counted"] / recall, 3) if recall else 0.0,
                "providers": sorted(slot["providers"]),
                "core_paper_ids": slot["core_paper_ids"],
            }
        )
    per_query.sort(key=lambda item: (item["query_yield"], item["counted"], item["unique_recall"]), reverse=True)

    return {
        "search_round": bundle.quality.get("search_round", ""),
        "screened": bool(bundle.quality.get("screened")),
        "screen_error": bundle.quality.get("screen_error", ""),
        "candidate_count": candidate,
        "kept_count": len(kept),
        "counted_count": len(counted),
        "core_count": len(core),
        "retrieval_precision_before_screening": round(len(kept) / candidate, 3) if candidate else 0.0,
        "screening_precision_after": round(len(counted) / len(kept), 3) if kept else 0.0,
        "per_query": per_query,
    }


def query_effectiveness_markdown(metrics: dict) -> str:
    lines = [
        "## Query Effectiveness",
        "",
        f"- Candidates: {metrics.get('candidate_count', 0)}",
        f"- Kept (after screening): {metrics.get('kept_count', 0)}",
        f"- Counted core+transferable: {metrics.get('counted_count', 0)} (core {metrics.get('core_count', 0)})",
        f"- Retrieval precision (kept/candidate): {metrics.get('retrieval_precision_before_screening', 0.0)}",
        f"- Screening precision (counted/kept): {metrics.get('screening_precision_after', 0.0)}",
        "",
        "| Query | recall | kept | counted | core | yield | providers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics.get("per_query", []):
        providers = ", ".join(row.get("providers", [])) or "-"
        lines.append(
            f"| {row['query']} | {row['unique_recall']} | {row['kept']} | "
            f"{row['counted']} | {row['core']} | {row['query_yield']} | {providers} |"
        )
    return "\n".join(lines) + "\n"
