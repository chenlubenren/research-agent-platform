from __future__ import annotations

from research_agent_platform.connectors.scholar import LiteratureBundle, PaperRecord
from research_agent_platform.review_metrics import (
    compute_review_metrics,
    query_effectiveness_markdown,
)


def _paper(pid: str, *, queries, decision: str, tier: str, relevance: float = 0.5) -> PaperRecord:
    return PaperRecord(
        title=pid,
        year=2024,
        abstract=pid,
        authors=["A"],
        url=f"https://example.org/{pid}",
        venue="V",
        citation_count=1,
        sources=["OpenAlex"],
        identifiers={},
        paper_id=pid,
        relevance_score=relevance,
        matched_queries=list(queries),
        verification_status="traceable_identifier",
        screen_decision=decision,
        screen_tier=tier,
    )


def test_compute_review_metrics_separates_retrieval_and_screening() -> None:
    papers = [
        _paper("P001", queries=["q_core"], decision="keep", tier="core"),
        _paper("P002", queries=["q_core"], decision="keep", tier="transferable"),
        _paper("P003", queries=["q_noise"], decision="drop", tier="exclude"),
        _paper("P004", queries=["q_bg"], decision="keep", tier="background"),
    ]
    bundle = LiteratureBundle(
        query="t", papers=papers, provider_status={}, queries=["q_core", "q_noise", "q_bg"],
        quality={"screened": True, "search_round": "focused"},
    )
    metrics = compute_review_metrics(bundle)
    assert metrics["candidate_count"] == 4
    assert metrics["kept_count"] == 3  # core + transferable + background
    assert metrics["counted_count"] == 2  # core + transferable
    assert metrics["core_count"] == 1
    # kept/candidate = 3/4, counted/kept = 2/3
    assert metrics["retrieval_precision_before_screening"] == 0.75
    assert metrics["screening_precision_after"] == round(2 / 3, 3)
    by_query = {row["query"]: row for row in metrics["per_query"]}
    assert by_query["q_core"]["query_yield"] == 1.0  # both counted
    assert by_query["q_noise"]["query_yield"] == 0.0
    assert by_query["q_core"]["core"] == 1
    md = query_effectiveness_markdown(metrics)
    assert "## Query Effectiveness" in md and "Query Yield" not in md  # table header uses 'yield'
    assert "q_core" in md
