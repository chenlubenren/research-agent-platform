from __future__ import annotations

import asyncio
from types import SimpleNamespace

from research_agent_platform import agent as agent_module
from research_agent_platform import reference_expansion
from research_agent_platform.agent import ResearchAgentService
from research_agent_platform.config import config
from research_agent_platform.connectors.scholar import LiteratureBundle, PaperRecord
from research_agent_platform.publication_quality import review_report_markdown
from research_agent_platform.memory.store import ResearchWikiStore
from research_agent_platform.review_pipeline import (
    query_effectiveness_payload,
    review_directions_markdown,
)


def test_role_reviews_use_dedicated_model_and_surface_failures(monkeypatch):
    calls: list[str | None] = []

    async def fake_generate_text(*, system_prompt, user_prompt, model=None, temperature=0.3):
        calls.append(model)
        if "Role focus: Assess domain" in user_prompt:
            raise RuntimeError("review upstream unavailable")
        return '{"findings":[]}'

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(config, "upstream_model", "default-model")
    monkeypatch.setattr(config, "upstream_review_model", "review-model")

    service = object.__new__(ResearchAgentService)
    reviews = asyncio.run(
        service._run_role_reviews(
            SimpleNamespace(),
            "paper/FINAL_PAPER.md",
            "# Manuscript\n\nA bounded claim.",
        )
    )

    assert len(calls) == 6
    assert set(calls) == {"review-model"}
    assert ResearchAgentService._role_review_status(reviews["domain"]) == "unavailable"
    assert ResearchAgentService._role_review_status(reviews["editor"]) == "available"


def test_review_report_marks_incomplete_review_as_needing_attention():
    report = review_report_markdown(
        {
            "manuscript_path": "paper/FINAL_PAPER.md",
            "review_status": "needs_attention",
            "unavailable_roles": ["domain"],
            "decision": "REVIEW_UNAVAILABLE",
            "findings": [],
        }
    )

    assert "- Status: needs_attention" in report
    assert "must not be treated as a clean review result" in report
    assert "domain" in report


def test_idea_gate_blocks_insufficient_cross_paper_evidence(tmp_path):
    service = object.__new__(ResearchAgentService)
    task = SimpleNamespace(artifact_root=str(tmp_path))

    content, gate = service._apply_idea_novelty_gate(
        task,
        "# Final Idea\n\n## Falsifiable Prediction\n\n可证伪：如果指标不改善则否定假设。\n",
    )

    assert gate["status"] == "blocked_preliminary"
    assert gate["full_text_papers"] == 0
    assert "## 创新性判定" in content


def test_wiki_coverage_counts_only_pdf_summary_with_evidence_id(tmp_path):
    source = tmp_path / "paper" / "uploads" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n")
    store = ResearchWikiStore(tmp_path)
    paper = store.paper_for_source("paper/uploads/paper.pdf")
    store.ensure_pdf_copy(paper)
    store.write_summary(paper, "# Paper\n\n## 研究问题\n\n- 未确认。")

    assert store.coverage_report()["full_text_papers"] == 0

    store.write_summary(
        paper,
        "# Paper\n\n## 可复用证据\n\n- `PE-ABCDEF1234` | 第 1 页 | evidence",
    )
    assert store.coverage_report()["full_text_papers"] == 1


def test_query_metrics_and_directions_use_retained_paper_ids():
    papers = [
        PaperRecord(
            paper_id="P001",
            title="Paper one",
            year=2025,
            abstract="Graph agent evidence.",
            authors=["A"],
            url="https://example.test/p1",
            venue="Test",
            citation_count=1,
            sources=["openalex"],
            matched_queries=["graph agents"],
        ),
        PaperRecord(
            paper_id="P002",
            title="Paper two",
            year=2026,
            abstract="Agent memory evidence.",
            authors=["B"],
            url="https://example.test/p2",
            venue="Test",
            citation_count=2,
            sources=["semantic_scholar"],
            matched_queries=["graph agents", "agent memory"],
        ),
    ]
    bundle = LiteratureBundle(
        query="agents",
        queries=["graph agents", "agent memory"],
        papers=papers,
        provider_status={"openalex": "ok"},
        quality={"candidate_count": 4},
    )

    metrics = query_effectiveness_payload(bundle)
    directions = review_directions_markdown(bundle)

    assert metrics["per_query"][0]["retained"] == 2
    assert "`P001`" in directions
    assert "`P002`" in directions


def test_reference_expansion_rejects_loopback_urls():
    safe, reason = asyncio.run(reference_expansion._is_safe_public_url("http://127.0.0.1/paper.pdf"))

    assert safe is False
    assert "禁止访问" in reason
