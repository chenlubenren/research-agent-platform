from __future__ import annotations

from research_agent_platform.review_pipeline import (
    evidence_sufficiency_directive,
    evidence_sufficiency_gate,
)


def _paper(pid: str, decision: str, tier: str, directions=None) -> dict:
    return {
        "paper_id": pid,
        "screen_decision": decision,
        "screen_tier": tier,
        "direction_ids": list(directions or []),
    }


def test_gate_passes_with_enough_core_and_covered_directions() -> None:
    papers = [_paper(f"P{i:03d}", "keep", "core") for i in range(1, 9)]
    directions = [{"id": "D1", "candidate_paper_ids": ["P001", "P002"]}]
    gate = evidence_sufficiency_gate(
        papers, directions=directions, selected_ids=["D1"], coverage_state="ok", min_core=8
    )
    assert gate["sufficient"] is True
    assert gate["gap_label"] == "likely"
    assert "PASS" in evidence_sufficiency_directive(gate)


def test_gate_fails_when_core_below_threshold() -> None:
    papers = [_paper("P001", "keep", "core"), _paper("P002", "keep", "transferable")]
    gate = evidence_sufficiency_gate(papers, coverage_state="ok", min_core=8)
    assert gate["sufficient"] is False
    assert gate["gap_label"] == "candidate"
    directive = evidence_sufficiency_directive(gate)
    assert "FAIL" in directive and "candidate gap" in directive


def test_gate_fails_on_uncovered_direction_and_degraded_coverage() -> None:
    papers = [_paper(f"P{i:03d}", "keep", "core") for i in range(1, 9)]
    # D2 selected but none of its candidate papers are in the counted set.
    directions = [
        {"id": "D1", "candidate_paper_ids": ["P001"]},
        {"id": "D2", "candidate_paper_ids": ["P999"]},
    ]
    gate = evidence_sufficiency_gate(
        papers, directions=directions, selected_ids=["D1", "D2"], coverage_state="degraded", min_core=8
    )
    assert gate["sufficient"] is False
    assert "D2" in gate["uncovered_directions"]
    assert any("覆盖降级" in reason for reason in gate["reasons"])


def test_gate_uses_counted_when_screening_absent() -> None:
    # No screen decisions => fall back to counted set (here empty) and fail.
    papers = [{"paper_id": "P001", "screen_decision": "", "screen_tier": ""}]
    gate = evidence_sufficiency_gate(papers, min_core=1)
    assert gate["screened"] is False
    assert gate["sufficient"] is False
