from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from research_agent_platform import agent as agent_module
from research_agent_platform.agent import ResearchAgentService
from research_agent_platform.config import config
from research_agent_platform.connectors.seafile import SeafileSyncResult, SeafileWorkspaceSync
from research_agent_platform.graphs.workflows import workflow_registry
from research_agent_platform.models import ArtifactRecord, ChatSession, CloudWorkspaceState, TaskRun
from research_agent_platform.router.intent import route_message


def test_reorganized_workflow_boundaries():
    workflows = workflow_registry()

    assert [stage.name for stage in workflows["/review"].stage_definitions] == [
        "research_brief",
        "literature_synthesis",
        "evidence_map",
        "research_gaps",
    ]
    assert [stage.name for stage in workflows["/idea"].stage_definitions] == [
        "idea_candidates",
        "idea_verification",
        "final_idea",
    ]
    assert [stage.name for stage in workflows["/plan"].stage_definitions] == [
        "blueprint",
        "experiment_plan",
        "execution_checklist",
    ]
    assert [stage.name for stage in workflows["/rebuttal"].stage_definitions] == [
        "rebuttal_intake",
        "review_to_paper_map",
        "response_strategy",
        "rebuttal_draft",
        "revision_plan",
        "revised_manuscript",
        "revision_ledger",
    ]
    assert [stage.name for stage in workflows["/write"].stage_definitions] == [
        "paper_evidence",
        "paper_plan",
        "narrative_report",
        "draft_sections",
        "paper_self_review",
        "paper_revision",
    ]
    assert [stage.name for stage in workflows["/fig"].stage_definitions] == [
        "figure_contract",
        "figure_design",
        "figure_render_and_qa",
        "figure_delivery",
    ]


def test_router_separates_literature_review_and_peer_review():
    literature = asyncio.run(route_message("帮我找文献并写一份文献综述"))
    rebuttal = asyncio.run(route_message("请分析审稿意见并回复审稿人"))
    idea = asyncio.run(route_message("围绕这个方向提出三个创新点"))
    plan = asyncio.run(route_message("给这个选题制定实验方案"))

    assert literature is not None and literature.command == "/review"
    assert rebuttal is not None and rebuttal.command == "/rebuttal"
    assert idea is not None and idea.command == "/idea"
    assert plan is not None and plan.command == "/plan"


def test_seafile_workspace_sync_is_incremental(tmp_path: Path):
    workspace = tmp_path / "agent-workspace" / "local" / "session_demo"
    source = workspace / "paper" / "draft.md"
    source.parent.mkdir(parents=True)
    source.write_text("first version", encoding="utf-8")
    uploads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Token test-token"
        if request.url.path == "/api2/repos/repo-1/dir/":
            return httpx.Response(200, json=[])
        if request.url.path == "/api2/repos/repo-1/upload-link/":
            return httpx.Response(200, json="https://cloud.example/upload/repo-1")
        if request.url.path == "/upload/repo-1":
            uploads.append(request.content.decode("latin-1"))
            return httpx.Response(200, json={"id": "file-id"})
        if request.url.path == "/api/v2.1/share-links/":
            return httpx.Response(200, json=[{"link": "https://cloud.example/d/shared-session"}])
        raise AssertionError(f"Unexpected Seafile request: {request.method} {request.url}")

    sync = SeafileWorkspaceSync(
        enabled=True,
        base_url="https://cloud.example",
        api_token="test-token",
        repo_id="repo-1",
        remote_root="research-agent",
        transport=httpx.MockTransport(handler),
    )

    first = asyncio.run(sync.sync_workspace(workspace, user_id="local", session_id="session_demo"))
    second = asyncio.run(sync.sync_workspace(workspace, user_id="local", session_id="session_demo"))

    assert first.status == "synced"
    assert first.remote_path == "/research-agent/local/session_demo"
    assert first.uploaded_files == 1
    assert first.preview_url == "https://cloud.example/d/shared-session"
    assert second.uploaded_files == 0
    assert len(uploads) == 1
    state = json.loads((workspace / "Content" / "CLOUD_SYNC.json").read_text(encoding="utf-8"))
    assert state["file_signatures"]["paper/draft.md"]


def test_completed_reply_includes_cloud_delivery_link(service: ResearchAgentService):
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
        download_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    assert service.store.list_tasks(session.session_id) == []

    run = TaskRun(
        session_id=session.session_id,
        command="/fig",
        objective="figure",
        route_source="explicit",
        workflow_title="Figure Generation Workflow",
        status="completed",
        artifact_root=session.workspace_root,
    )
    run.artifacts.append(
        ArtifactRecord(
            name="figure.png",
            kind="image",
            relative_path="figures/figure.png",
            absolute_path=f"{session.workspace_root}\\figures\\figure.png",
            url_path="/workspace-files/figures/figure.png",
            description="final figure",
        )
    )
    service.store.save_task(run)

    reply = service.runtime.build_reply(run.task_id, service.workflows["/fig"])

    assert "最终产物已写入本会话工作区" in reply["text"]
    assert "清华网盘预览/下载链接" in reply["text"]


def test_required_cloud_delivery_fails_without_share_link(service: ResearchAgentService, monkeypatch):
    monkeypatch.setattr(config, "cloud_delivery_required", True)
    session = service.store.create_session()
    run = TaskRun(
        session_id=session.session_id,
        command="/fig",
        objective="figure",
        route_source="explicit",
        workflow_title="Figure Generation Workflow",
        status="completed",
        artifact_root=session.workspace_root,
    )
    service.store.save_task(run)

    service.enforce_cloud_delivery(run, CloudWorkspaceState(status="error", error="share link unavailable"))

    failed = service.get_task(run.task_id)
    assert failed is not None
    assert failed.status == "failed"
    assert "清华网盘交付失败" in failed.error


def test_runtime_completed_reply_uses_final_artifact_path(service: ResearchAgentService):
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    task = TaskRun(
        session_id=session.session_id,
        command="/write",
        objective="write",
        route_source="explicit",
        workflow_title="Paper Writing Workflow",
        status="completed",
        artifact_root=session.workspace_root,
    )
    task.artifacts.append(
        ArtifactRecord(
            name="paper.md",
            kind="report",
            relative_path="paper/PAPER_REVISED.md",
            absolute_path=f"{session.workspace_root}\\paper\\PAPER_REVISED.md",
            url_path="/workspace-files/paper/PAPER_REVISED.md",
            description="final paper",
        )
    )
    service.store.save_task(task)

    reply = service.runtime.build_reply(task.task_id, service.workflows["/write"])

    assert "最终产物位于" in reply["text"]
    assert "paper/PAPER_REVISED.md" in reply["text"]
    assert "任务记录位于" not in reply["text"]


def test_workflow_completes_only_after_cloud_link_is_ready(service: ResearchAgentService, monkeypatch):
    observed_statuses: list[str] = []

    async def delayed_cloud_sync(task):
        latest = service.get_task(task.task_id)
        observed_statuses.append(latest.status if latest else "missing")
        session = service.store.load_session(task.session_id)
        assert session is not None
        session.cloud_workspace = CloudWorkspaceState(
            status="synced",
            configured=True,
            preview_url="https://cloud.example/d/ready",
        )
        service.store.save_session(session)
        return session.cloud_workspace

    monkeypatch.setattr(service, "sync_task_workspace", delayed_cloud_sync)
    result = asyncio.run(service.chat(None, "/plan 制定实验方案"))

    assert "running" in observed_statuses
    assert result["status"] == "completed"
    assert "https://cloud.example/d/ready" in result["text"]


def test_completed_reply_without_artifacts_omits_cloud_delivery_link(service: ResearchAgentService):
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    run = TaskRun(
        session_id=session.session_id,
        command="/chat",
        objective="hello",
        route_source="chat",
        workflow_title="Chat Response",
        status="completed",
        artifact_root=session.workspace_root,
        response_text="这是最终回复。",
    )
    service.store.save_task(run)

    reply = service._build_reply(run, text="这是最终回复。")

    assert "清华网盘预览/下载链接" not in reply["text"]


def test_cloud_sync_progress_includes_workspace_link(service: ResearchAgentService, monkeypatch):
    session = service.store.create_session()
    task = TaskRun(
        session_id=session.session_id,
        command="/chat",
        objective="hello",
        route_source="chat",
        workflow_title="Chat Response",
        status="running",
        artifact_root=session.workspace_root,
    )
    service.store.save_task(task)

    async def fake_sync_workspace(*args, **kwargs):
        return SeafileSyncResult(
            status="synced",
            remote_path="/research-agent/local/session",
            share_url="https://cloud.example/d/session-link",
            preview_url="https://cloud.example/d/session-link",
            download_url="https://cloud.example/d/session-link",
            synced_files=1,
            uploaded_files=1,
        )

    monkeypatch.setattr(service.cloud, "sync_workspace", fake_sync_workspace)
    asyncio.run(service.sync_task_workspace(task))

    synced = service.get_task(task.task_id)
    assert synced is not None
    assert any("清华网盘工作区：https://cloud.example/d/session-link" in item for item in synced.progress_log)


def test_plan_does_not_trigger_literature_search(service: ResearchAgentService, monkeypatch):
    calls = 0

    async def unexpected_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("/plan must reuse evidence instead of launching literature search")

    monkeypatch.setattr(service.scholar, "search_bundle", unexpected_search)
    result = asyncio.run(service.chat(None, "/plan 制定实验方案"))

    assert result["status"] == "completed"
    assert calls == 0


def test_idea_reuses_existing_review_evidence(service: ResearchAgentService, monkeypatch):
    session = service.store.create_session()
    workspace = Path(session.workspace_root)
    evidence = workspace / "bib" / "EVIDENCE_MAP.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("# Evidence Map\n\nGrounded evidence.", encoding="utf-8")
    calls = 0

    async def unexpected_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("/idea must reuse existing review evidence")

    monkeypatch.setattr(service.scholar, "search_bundle", unexpected_search)
    result = asyncio.run(service.chat(session.session_id, "/idea 提出一个创新方向"))

    assert result["status"] == "completed"
    assert calls == 0


def test_figure_prompt_uses_session_context(service: ResearchAgentService, monkeypatch):
    session = service.store.create_session()
    workspace = Path(session.workspace_root)
    note = workspace / "paper" / "PAPER_DRAFT.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("urban fringe topic", encoding="utf-8")
    task = TaskRun(
        session_id=session.session_id,
        command="/fig",
        objective="draw a figure",
        route_source="explicit",
        workflow_title="Figure Generation Workflow",
        artifact_root=session.workspace_root,
    )
    service.store.save_session(session)
    service.store.save_task(task)

    captured: dict[str, str] = {}

    async def fake_generate_text(*, system_prompt: str, user_prompt: str, model=None, temperature=None):
        captured["user_prompt"] = user_prompt
        return "Figure prompt"

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)
    prompt = asyncio.run(service._build_figure_render_prompt(task, "inventory", "briefs"))

    assert prompt == "Figure prompt"
    assert "urban fringe topic" in captured["user_prompt"]


def test_presentation_pipeline_passes_session_context(service: ResearchAgentService, monkeypatch):
    session = service.store.create_session()
    workspace = Path(session.workspace_root)
    (workspace / "presentation").mkdir(parents=True, exist_ok=True)
    (workspace / "Content").mkdir(parents=True, exist_ok=True)
    (workspace / "presentation" / "SLIDE_CONTENT.md").write_text(
        "## Slide 1: Result\n### Main Message\n- Show the urban fringe topic.\n",
        encoding="utf-8",
    )
    (workspace / "presentation" / "SLIDES_OUTLINE.md").write_text(
        "## Slide 1: Result\n### Main Message\n- Show the urban fringe topic.\n",
        encoding="utf-8",
    )
    task = TaskRun(
        session_id=session.session_id,
        command="/present",
        objective="make a ppt",
        route_source="explicit",
        workflow_title="Presentation Workflow",
        artifact_root=session.workspace_root,
    )
    service.store.save_session(session)
    service.store.save_task(task)

    recorded: dict[str, str] = {}

    def fake_build_slide_prompt(slide, template, mode, asset=None):
        recorded["slide_title"] = slide.title
        return "slide prompt"

    monkeypatch.setattr(agent_module, "build_slide_prompt", fake_build_slide_prompt)
    asyncio.run(service._write_presentation_delivery_artifacts(task))

    assert recorded["slide_title"] == "Result"
