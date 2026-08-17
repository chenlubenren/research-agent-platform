from __future__ import annotations

import json
from pathlib import Path

from research_agent_platform import agent as agent_module
from research_agent_platform import paper_pipeline as paper_pipeline_module
from research_agent_platform.agent import ResearchAgentService
from research_agent_platform.models import UploadBatchRecord
from research_agent_platform.paper_pipeline import (
    build_paper_quality_reports,
    collect_paper_evidence,
    resolve_write_source_config,
    select_venue_profile,
)

from .conftest import run


def _prepare_write_workspace(session) -> None:
    workspace = Path(session.workspace_root)
    result = workspace / "figures" / "results.csv"
    plan = workspace / "plan" / "EXPERIMENT_PLAN.md"
    bib = workspace / "bib" / "references.bib"
    result.parent.mkdir(parents=True, exist_ok=True)
    plan.parent.mkdir(parents=True, exist_ok=True)
    bib.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("model,accuracy\nours,0.91\nbaseline,0.84\n", encoding="utf-8")
    plan.write_text("# Experiment Plan\n\nEvaluate accuracy against the baseline.", encoding="utf-8")
    bib.write_text("@article{smith2025, title={Grounded Research}}", encoding="utf-8")
    session.upload_batches.append(
        UploadBatchRecord(relative_paths=["figures/results.csv", "bib/references.bib"])
    )


def test_write_source_set_combines_latest_uploads_and_workspace(service: ResearchAgentService):
    session = service.store.create_session()
    _prepare_write_workspace(session)
    service.store.save_session(session)

    config = resolve_write_source_config(
        "/write 写论文",
        Path(session.workspace_root),
        session.upload_batches,
    )

    assert config.resolved_scope == "session"
    assert "figures/results.csv" in config.source_refs
    assert "plan/EXPERIMENT_PLAN.md" in config.source_refs
    assert config.upload_batch_id == session.upload_batches[-1].upload_batch_id


def test_collect_paper_evidence_assigns_stable_ids(tmp_path: Path):
    source = tmp_path / "plan" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Result\n\nAccuracy is 0.91.", encoding="utf-8")

    first = collect_paper_evidence(tmp_path, ["plan/notes.md"])
    second = collect_paper_evidence(tmp_path, ["plan/notes.md"])

    assert first == second
    assert first[0]["evidence_id"].startswith("PE-")
    assert first[0]["source_path"] == "plan/notes.md"


def test_collect_paper_evidence_prioritizes_future_work_pages(tmp_path: Path, monkeypatch):
    source = tmp_path / "paper" / "uploads" / "research.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        paper_pipeline_module,
        "_extract_source_blocks",
        lambda _path: [
            (1, "Introduction and problem statement. " * 35, "document_text"),
            (2, "Method details and equations. " * 35, "document_text"),
            (3, "Additional implementation details. " * 35, "document_text"),
            (8, "Conclusion. Interesting directions for future work include attention. " * 18, "document_text"),
        ],
    )

    records = collect_paper_evidence(
        tmp_path,
        ["paper/uploads/research.pdf"],
        total_limit=2600,
        query="Discussion limitations and future work",
        prioritize_research_sections=True,
    )

    assert [record["page"] for record in records][:2] == [8, 1]
    assert "future work" in records[0]["excerpt"].lower()


def test_write_evidence_cache_is_bound_to_task_source_set(service: ResearchAgentService):
    session = service.store.create_session()
    workspace = Path(session.workspace_root)
    first_source = workspace / "plan" / "first.md"
    second_source = workspace / "plan" / "second.md"
    first_source.parent.mkdir(parents=True, exist_ok=True)
    first_source.write_text("first evidence", encoding="utf-8")
    second_source.write_text("second evidence", encoding="utf-8")
    first_task = service.store.list_tasks(session.session_id)
    assert first_task == []

    from research_agent_platform.models import TaskRun, WriteSourceConfig

    task = TaskRun(
        session_id=session.session_id,
        command="/write",
        objective="write",
        route_source="explicit",
        workflow_title="Paper Writing Workflow",
        artifact_root=session.workspace_root,
        write_source=WriteSourceConfig(
            resolved_scope="selected",
            source_refs=["plan/second.md"],
        ),
    )
    Path(workspace, "paper").mkdir(parents=True, exist_ok=True)
    Path(workspace, "Content").mkdir(parents=True, exist_ok=True)
    Path(workspace, "paper", "PAPER_EVIDENCE_MAP.json").write_text(
        json.dumps([{"evidence_id": "PE-OLD", "source_path": "plan/first.md"}]),
        encoding="utf-8",
    )
    Path(workspace, "Content", "PAPER_EVIDENCE_METADATA.json").write_text(
        json.dumps({"task_id": "old-task", "source_refs": ["plan/first.md"]}),
        encoding="utf-8",
    )

    records = service._paper_evidence_records(task)

    assert records[0]["source_path"] == "plan/second.md"


def test_write_runs_evidence_review_revision_and_delivery_gates(
    service: ResearchAgentService, monkeypatch
):
    session = service.store.create_session()
    _prepare_write_workspace(session)
    service.store.save_session(session)

    async def generate_paper(*, system_prompt, user_prompt, model=None, temperature=0.3):
        if "Current stage: Paper Plan" in user_prompt:
            return (
                "# Paper Plan\n\n## Target Story\nEvidence-grounded result.\n\n## Submission Target\nUnspecified.\n\n"
                "## Section Outline\nStandard paper.\n\n## Section Responsibilities and Paragraph Jobs\nOne claim per paragraph.\n\n"
                "## Claim to Evidence Map\n- Accuracy -> frozen PE evidence.\n\n## Terminology and Claim Boundaries\nBounded.\n\n"
                "## Writing Risks\nMissing robustness.\n\n## Questions for Human Review\nNone.\n\n## Decision Required\nNone."
            )
        if "Current stage: Narrative Report" in user_prompt:
            return (
                "# Narrative\n\n## Problem Statement\nProblem.\n\n## Core Claim\nAccuracy improved.\n\n"
                "## Method Summary\nMethod.\n\n## Key Results\n0.91 vs 0.84.\n\n"
                "## Evidence and Citation Boundaries\nUse frozen evidence.\n\n## Limitations\nRobustness.\n\n## Open Gaps\nMore seeds."
            )
        if "Current stage: Draft Sections" in user_prompt:
            return _paper_markdown("Draft title", "Draft evidence wording")
        if "Current stage: Paper Self Review" in user_prompt:
            return (
                "# Self Review\n\n## Quality Scores\n- Evidence: 4\n\n## Major Issues\n- M1: Clarify evidence wording.\n\n"
                "## Minor Issues\n- None.\n\n## Claim and Evidence Findings\n- Grounded.\n\n"
                "## Citation Findings\n- smith2025 resolves.\n\n## Structure and Venue Findings\n- Complete.\n\n"
                "## Revision Actions\n- M1 revise wording."
            )
        if "Current stage: Paper Revision" in user_prompt:
            return _paper_markdown("Revised title", "Revised evidence wording")
        return "# Artifact\n\nGenerated."

    monkeypatch.setattr(agent_module, "generate_text", generate_paper)
    result = run(service.chat(session.session_id, "/write --source workspace 写论文 word pdf"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "completed"
    assert result["checkpoint"] is None
    assert task is not None and task.write_source is not None
    assert Path(task.artifact_root, "paper", "PAPER_EVIDENCE_MAP.json").exists()
    assert Path(task.artifact_root, "paper", "PAPER_SELF_REVIEW.md").exists()
    assert Path(task.artifact_root, "paper", "PAPER_REVISED.md").exists()
    assert Path(task.artifact_root, "paper", "PAPER_REVISED.docx").exists()
    assert Path(task.artifact_root, "paper", "PAPER_REVISED.pdf").exists()
    selection = json.loads(
        Path(task.artifact_root, "Content", "PAPER_SOURCE_SELECTION.json").read_text(encoding="utf-8")
    )
    assert selection["source_boundary"] == "frozen_at_task_start"
    delivery = json.loads(
        Path(task.artifact_root, "paper", "PAPER_DELIVERY_REPORT.json").read_text(encoding="utf-8")
    )
    citation = json.loads(
        Path(task.artifact_root, "paper", "CITATION_AUDIT.json").read_text(encoding="utf-8")
    )
    assert delivery["status"] == "pass"
    assert citation["unknown_keys"] == []


def test_quality_gate_reports_unknown_citation(tmp_path: Path):
    manuscript = tmp_path / "paper" / "PAPER_REVISED.md"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text(_paper_markdown("Title", "Text").replace("smith2025", "unknown2026"), encoding="utf-8")
    (tmp_path / "bib").mkdir()
    (tmp_path / "bib" / "references.bib").write_text(
        "@article{smith2025, title={Known}}", encoding="utf-8"
    )

    citation, delivery = build_paper_quality_reports(
        tmp_path,
        "paper/PAPER_REVISED.md",
        [],
        select_venue_profile("research paper"),
    )

    assert citation["unknown_keys"] == ["unknown2026"]
    assert delivery["status"] == "needs_attention"


def _paper_markdown(title: str, wording: str) -> str:
    return (
        f"# {title}\n\n## Abstract\n{wording}.\n\n## Introduction\nProblem context [@smith2025].\n\n"
        "## Related Work\nPrior work [@smith2025].\n\n## Method\nMethod description.\n\n"
        "## Experiments or Results\nAccuracy is 0.91 versus 0.84.\n\n## Limitations\nRobustness remains limited.\n\n"
        "## Conclusion\nBounded conclusion.\n\n## References\n- [@smith2025]\n\n## Unresolved Author Inputs\nNone."
    )
