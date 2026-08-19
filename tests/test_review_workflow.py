from __future__ import annotations

import json
from pathlib import Path

from research_agent_platform import agent as agent_module
from research_agent_platform.agent import ResearchAgentService
from research_agent_platform.config import config
from research_agent_platform.connectors.scholar import LiteratureBundle, PaperRecord

from .conftest import run


def _disable_review_pauses(monkeypatch) -> None:
    # These legacy end-to-end tests exercise the retrieval/gate path, not the two
    # human-in-the-loop stages; run them non-interactively so no checkpoint pauses.
    monkeypatch.setattr(config, "review_clarify_enabled", False)
    monkeypatch.setattr(config, "review_direction_selection_enabled", False)
    # Pin the source-count gate so the assertions stay deterministic regardless of any
    # local .env override (e.g. REVIEW_MINIMUM_SOURCES tuned for production runs).
    monkeypatch.setattr(config, "review_minimum_sources", 10)
    monkeypatch.setattr(config, "review_recommended_sources", 15)


def _bundle(query: str, count: int) -> LiteratureBundle:
    papers = [
        PaperRecord(
            title=f"{query} study {index}",
            year=2018 + index,
            abstract=f"Evidence for {query}, method {index}, and limitations.",
            authors=[f"Author {index}"],
            url=f"https://doi.org/10.1000/{index}",
            venue="Research Journal",
            citation_count=index,
            sources=["OpenAlex"],
            identifiers={"doi": f"10.1000/{index}"},
            paper_id=f"P{index:03d}",
            relevance_score=0.9,
            matched_queries=[query],
            verification_status="traceable_identifier",
        )
        for index in range(1, count + 1)
    ]
    return LiteratureBundle(
        query=query,
        queries=[query, f"{query} review"],
        papers=papers,
        provider_status={"openalex": f"ok ({count} records)"},
        quality={
            "status": "adequate" if count >= 10 else "insufficient",
            "candidate_count": count,
            "relevant_count": count,
            "minimum_for_synthesis": 10,
            "recommended_for_review": 15,
            "traceable_count": count,
            "provider_success_count": 1,
        },
    )


def test_review_stops_before_synthesis_when_evidence_is_insufficient(
    service: ResearchAgentService,
    monkeypatch,
) -> None:
    _disable_review_pauses(monkeypatch)

    async def insufficient(query, **_kwargs):
        return _bundle(query, 2)

    monkeypatch.setattr(service.scholar, "search_bundle", insufficient)
    result = run(service.chat(None, "/review 写一个当前海绵城市相关的综述"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "failed"
    assert task is not None and task.status == "failed"
    workspace = Path(task.artifact_root)
    assert (workspace / "bib" / "RESEARCH_BRIEF.md").exists()
    assert (workspace / "bib" / "LITERATURE_SEARCH.json").exists()
    assert "Gate: FAIL" in (workspace / "bib" / "RETRIEVAL_QUALITY.md").read_text(encoding="utf-8")
    assert not (workspace / "bib" / "LITERATURE_REVIEW.md").exists()
    assert not (workspace / "bib" / "EVIDENCE_MAP.md").exists()
    assert not (workspace / "bib" / "RESEARCH_GAPS.md").exists()
    assert not (workspace / "bib" / "CITATION_AUDIT.json").exists()


def test_nine_traceable_sources_do_not_satisfy_ten_source_gate(
    service: ResearchAgentService,
    monkeypatch,
) -> None:
    _disable_review_pauses(monkeypatch)

    async def insufficient(query, **_kwargs):
        return _bundle(query, 9)

    monkeypatch.setattr(service.scholar, "search_bundle", insufficient)
    result = run(service.chat(None, "/review sponge city"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "failed"
    assert task is not None and task.status == "failed"
    quality = Path(task.artifact_root, "bib", "RETRIEVAL_QUALITY.md").read_text(encoding="utf-8")
    assert "Required minimum sources: 10" in quality
    assert "Admitted sources: 9" in quality
    assert "Gate: FAIL" in quality


def test_review_completes_with_stable_id_audit(
    service: ResearchAgentService,
    monkeypatch,
) -> None:
    _disable_review_pauses(monkeypatch)

    async def review_text(*, system_prompt, user_prompt, model=None, temperature=0.3):
        if "Current stage: Research Brief" in user_prompt:
            return "# Brief\n\n## Search Strategy\nQ1: sponge city\nQ2: green stormwater infrastructure review"
        if "Current stage: Literature Synthesis" in user_prompt:
            return "# Review\n\nEvidence agrees [P001] [P002] [P003] [P004] [P005] [P006] [P007] [P008] [P009] [P010]."
        if "Current stage: Evidence Map" in user_prompt:
            return "# Evidence Map\n\nSupported claim [P001] and unknown source [P999]."
        return "# Research Gaps\n\nSupported gap [P002] [P003]."

    monkeypatch.setattr(agent_module, "generate_text", review_text)
    result = run(service.chat(None, "/review sponge city"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "completed"
    assert task is not None
    workspace = Path(task.artifact_root)
    audit = json.loads((workspace / "bib" / "CITATION_AUDIT.json").read_text(encoding="utf-8"))
    coverage = json.loads((workspace / "bib" / "REVIEW_COVERAGE.json").read_text(encoding="utf-8"))
    assert audit["unknown_paper_ids"] == ["P999"]
    assert audit["status"] == "needs_attention"
    assert coverage["paper_count"] == 10
    assert coverage["reference_coverage"] == 1.0


def test_ten_extracted_local_papers_can_satisfy_review_gate(
    service: ResearchAgentService,
    monkeypatch,
) -> None:
    _disable_review_pauses(monkeypatch)
    session = service.store.create_session()
    uploads = Path(session.workspace_root, "paper", "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    for index in range(1, 11):
        (uploads / f"paper-{index}.txt").write_text(
            f"海绵城市 paper {index}. Method, result, and limitation.",
            encoding="utf-8",
        )
    (uploads / "unrelated.txt").write_text(
        "Contemporary film aesthetics and ritual literature.",
        encoding="utf-8",
    )

    async def no_external_evidence(query, **_kwargs):
        return _bundle(query, 0)

    monkeypatch.setattr(service.scholar, "search_bundle", no_external_evidence)
    result = run(service.chat(session.session_id, "/review 海绵城市"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "completed"
    assert task is not None
    workspace = Path(task.artifact_root)
    quality = (workspace / "bib" / "RETRIEVAL_QUALITY.md").read_text(encoding="utf-8")
    search = json.loads((workspace / "bib" / "LITERATURE_SEARCH.json").read_text(encoding="utf-8"))
    assert "Gate: PASS" in quality
    assert "Required minimum sources: 10" in quality
    assert "Local candidates discovered: 11" in quality
    assert "Local sources with extracted evidence: 10" in quality
    assert len(search["quality"]["local_evidence_sources"]) == 10


def test_new_review_archives_previous_outputs_before_failed_retrieval(
    service: ResearchAgentService,
    monkeypatch,
) -> None:
    _disable_review_pauses(monkeypatch)
    first = run(service.chat(None, "/review sponge city"))
    first_task = service.get_task(first["task_id"])
    assert first_task is not None
    workspace = Path(first_task.artifact_root)
    assert (workspace / "bib" / "LITERATURE_REVIEW.md").exists()

    async def insufficient(query, **_kwargs):
        return _bundle(query, 0)

    monkeypatch.setattr(service.scholar, "search_bundle", insufficient)
    second = run(service.chat(first_task.session_id, "/review quantum networking"))
    second_task = service.get_task(second["task_id"])

    assert second["status"] == "failed"
    assert second_task is not None
    archive = workspace / "bib" / "archive" / f"prior-to-{second_task.task_id}"
    assert (archive / "LITERATURE_REVIEW.md").exists()
    assert not (workspace / "bib" / "LITERATURE_REVIEW.md").exists()
    assert not (workspace / "bib" / "EVIDENCE_MAP.md").exists()
    assert not (workspace / "bib" / "RESEARCH_GAPS.md").exists()
