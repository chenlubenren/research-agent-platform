from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from research_agent_platform import api as api_module
from research_agent_platform import agent as agent_module
from research_agent_platform.models import ArtifactRecord, CloudWorkspaceState, TaskRun


BLOCKING_PRESENTATION_DECISION = (
    "\n\n## Decision Required\n"
    "Blocking: Yes\n"
    "Question: Choose the external disclosure scope.\n"
    "Why user input is necessary: This controls whether an unpublished result is disclosed.\n"
    "Option A: Include the unpublished result.\n"
    "Option B: Exclude the unpublished result.\n"
    "Recommended Default: Option B.\n"
)


def test_openai_compatible_chat_completion_routes_to_task(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "research-agent-platform",
            "messages": [{"role": "user", "content": "/present 做一个中文汇报"}],
            "metadata": {"session_id": None, "user_id": "qingxiaoda-user"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["x_agent_task"]["status"] == "completed"
    assert payload["x_agent_task"]["task_id"]

    task = service.get_task(payload["x_agent_task"]["task_id"])
    assert task is not None
    assert task.user_id == "qingxiaoda-user"
    assert f"agent-workspace/local/{task.session_id}" in task.artifact_root.replace("\\", "/")
    assert "agent-workspace/qingxiaoda-user/" not in task.artifact_root.replace("\\", "/")


def test_openai_compatible_chat_completion_streams_done_marker(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "research-agent-platform",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data: " in body
    assert "chat.completion.chunk" in body
    assert "你好，我是科研智能体 Research Agent" in body
    assert "data: [DONE]" in body


def test_openai_compatible_chat_completion_handles_greeting_without_upstream_call(
    service, monkeypatch
):
    monkeypatch.setattr(api_module, "agent", service)

    async def fail_generate_text(**kwargs):
        raise AssertionError("greeting probe should not call upstream text generation")

    monkeypatch.setattr(agent_module, "generate_text", fail_generate_text)
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert "科研智能体" in payload["choices"][0]["message"]["content"]


def test_openai_compatible_base_path_has_discovery_response(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.get("/v1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["base_path"] == "/v1"
    assert "/v1/chat/completions" in payload["endpoints"]


def test_openai_text_chat_adds_one_time_intro_and_hides_server_path(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "你好，我是科研智能体 Research Agent" in content
    assert "/v1" not in content or "工作区" in content
    assert "/agent-workspace/" not in content
    assert response.json()["x_agent_task"]["artifact_root"] == ""


def test_openai_compatible_endpoints_require_public_api_key_when_configured(
    service, monkeypatch
):
    monkeypatch.setattr(api_module, "agent", service)
    monkeypatch.setattr(api_module.config, "public_api_key", "secret-key")
    client = TestClient(api_module.app)

    missing = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    allowed = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-key"},
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert missing.status_code == 401
    assert allowed.status_code == 200


def test_openai_compatible_responses_and_task_files(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "research-agent-platform",
            "input": "/present 做一个中文汇报",
            "metadata": {"session_id": "session-1"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    task_id = payload["metadata"]["task_id"]
    files = client.get(f"/api/tasks/{task_id}/files").json()

    assert payload["metadata"]["status"] == "completed"
    assert files["task_id"] == task_id
    assert any(item["relative_path"].endswith("SLIDES_OUTLINE.md") for item in files["files"])
    assert not any(item["relative_path"].endswith("Content/slides_outline_checkpoint.md") for item in files["files"])
    assert any(item["relative_path"].startswith("presentation/") for item in files["files"])


def test_session_file_upload_creates_session_and_classifies_files(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/api/session/files",
        data={"user_id": "local", "target": "auto"},
        files=[
            ("files", ("result.png", b"png-data", "image/png")),
            ("files", ("draft.pdf", b"pdf-data", "application/pdf")),
            ("files", ("train.py", b"print('ok')", "text/x-python")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "local"
    assert payload["upload_batch_id"].startswith("upload_")
    session = service.store.load_session(payload["session_id"])
    assert session is not None
    assert session.upload_batches[-1].upload_batch_id == payload["upload_batch_id"]
    assert set(session.upload_batches[-1].relative_paths) == {
        "figures/uploads/result.png",
        "paper/uploads/draft.pdf",
        "code/uploads/train.py",
    }
    assert f"local/{payload['session_id']}" in payload["workspace_root"].replace("\\", "/")
    paths = {item["relative_path"] for item in payload["files"]}
    assert paths == {
        "figures/uploads/result.png",
        "paper/uploads/draft.pdf",
        "code/uploads/train.py",
    }
    for item in payload["files"]:
        assert item["size"] > 0
        assert Path(item["absolute_path"]).exists()


def test_create_session_initializes_workspace(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post("/api/sessions", json={"user_id": "local"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("session_")
    workspace = Path(payload["workspace_root"])
    assert workspace.exists()
    for directory in (
        "bib",
        "plan",
        "idea",
        "code",
        "figures",
        "paper",
        "presentation",
        "rebuttal",
        "wiki",
        "Content",
        "logs",
    ):
        assert (workspace / directory).is_dir()


def test_session_file_upload_reuses_session_and_preserves_duplicate_names(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    first = client.post(
        "/api/session/files",
        data={"target": "plan"},
        files={"files": ("notes.txt", b"first", "text/plain")},
    ).json()
    second = client.post(
        "/api/session/files",
        data={"session_id": first["session_id"], "target": "plan"},
        files={"files": ("notes.txt", b"second", "text/plain")},
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["session_id"] == first["session_id"]
    assert first["files"][0]["relative_path"] == "plan/uploads/notes.txt"
    assert payload["files"][0]["relative_path"] == "plan/uploads/notes_2.txt"


def test_bib_pdf_upload_is_stored_under_papers(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/api/session/files",
        data={"target": "bib"},
        files={"files": ("downloaded.pdf", b"%PDF-1.7\npaper", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"][0]["relative_path"] == "bib/papers/downloaded.pdf"
    assert Path(payload["files"][0]["absolute_path"]).read_bytes().startswith(b"%PDF")


def test_auto_upload_routes_paper_id_pdf_to_bib_papers(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/api/session/files",
        data={"target": "auto"},
        files={"files": ("P001_method.pdf", b"%PDF-1.7\npaper", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["files"][0]["relative_path"] == "bib/papers/P001_method.pdf"


def test_session_file_upload_rejects_invalid_target(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.post(
        "/api/session/files",
        data={"target": "outside"},
        files={"files": ("notes.txt", b"content", "text/plain")},
    )

    assert response.status_code == 400


def test_manual_session_cloud_sync_endpoint(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)
    session = service.store.create_session()

    response = client.post(f"/api/sessions/{session.session_id}/sync")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session.session_id
    assert payload["cloud_workspace"]["status"] == "disabled"


def test_approve_returns_before_background_completion(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)

    async def scenario():
        original_generate_text = agent_module.generate_text

        async def generate_with_choices(*, system_prompt, user_prompt, model=None, temperature=0.3):
            content = await original_generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
            )
            if "Current stage: Slides Outline" in user_prompt:
                return content + BLOCKING_PRESENTATION_DECISION
            return content

        monkeypatch.setattr(agent_module, "generate_text", generate_with_choices)
        start = await service.chat(None, "/present 做一个中文汇报")
        task_id = start["task_id"]
        release = asyncio.Event()
        original_approve = service.approve_task

        async def delayed_approve(approved_task_id: str, feedback: str = ""):
            await release.wait()
            return await original_approve(approved_task_id, feedback)

        monkeypatch.setattr(service, "approve_task", delayed_approve)
        response = await api_module.api_approve(task_id, {"feedback": ""})

        assert response["status"] == "running"
        assert response["checkpoint"] is None
        assert "后台" in response["text"]
        assert task_id in api_module.background_tasks
        assert not api_module.background_tasks[task_id].done()

        release.set()
        await api_module.background_tasks[task_id]
        task = service.get_task(task_id)
        assert task is not None
        assert task.status == "completed"

    asyncio.run(scenario())


def test_background_failure_is_persisted(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)

    async def scenario():
        original_generate_text = agent_module.generate_text

        async def generate_with_choices(*, system_prompt, user_prompt, model=None, temperature=0.3):
            content = await original_generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
            )
            if "Current stage: Slides Outline" in user_prompt:
                return content + BLOCKING_PRESENTATION_DECISION
            return content

        monkeypatch.setattr(agent_module, "generate_text", generate_with_choices)
        start = await service.chat(None, "/present 做一个中文汇报")
        task_id = start["task_id"]

        async def failing_approve(approved_task_id: str, feedback: str = ""):
            raise RuntimeError("image provider unavailable")

        monkeypatch.setattr(service, "approve_task", failing_approve)
        response = await api_module.api_approve(task_id, {"feedback": ""})
        assert response["status"] == "running"
        await api_module.background_tasks[task_id]

        payload = api_module.task_status_payload(task_id)
        assert payload["status"] == "failed"
        assert payload["error"] == "image provider unavailable"
        assert any("image provider unavailable" in item for item in payload["progress"])

    asyncio.run(scenario())


def test_task_status_payload_supports_progress_polling(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    original_generate_text = agent_module.generate_text

    async def generate_with_choices(*, system_prompt, user_prompt, model=None, temperature=0.3):
        content = await original_generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
        if "Current stage: Slides Outline" in user_prompt:
            return content + BLOCKING_PRESENTATION_DECISION
        return content

    monkeypatch.setattr(agent_module, "generate_text", generate_with_choices)
    start = asyncio.run(service.chat(None, "/present 做一个中文汇报"))

    payload = api_module.task_status_payload(start["task_id"])

    assert payload["status"] == "waiting_human"
    assert payload["current_stage_name"] == "slides_outline"
    assert payload["checkpoint"]["title"] == "Presentation Outline Approval"
    assert payload["progress"]
    assert payload["artifacts"]


def test_task_status_payload_appends_cloud_link_to_response_text(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    task = TaskRun(
        session_id=session.session_id,
        command="/chat",
        objective="hello",
        route_source="chat",
        workflow_title="Chat Response",
        status="completed",
        artifact_root=session.workspace_root,
        response_text="这是最终回复。",
    )
    task.artifacts.append(
        ArtifactRecord(
            name="report.md",
            kind="report",
            relative_path="report.md",
            absolute_path=f"{session.workspace_root}\\report.md",
            url_path="/workspace-files/report.md",
            description="final report",
        )
    )
    service.store.save_task(task)

    payload = api_module.task_status_payload(task.task_id)

    assert "清华网盘预览/下载链接" in payload["response_text"]
    assert "https://cloud.example/d/session-link" in payload["response_text"]
    assert payload["text"] == payload["response_text"]


def test_task_status_payload_omits_cloud_link_without_artifacts(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    task = TaskRun(
        session_id=session.session_id,
        command="/chat",
        objective="hello",
        route_source="chat",
        workflow_title="Chat Response",
        status="completed",
        artifact_root=session.workspace_root,
        response_text="这是最终回复。",
    )
    service.store.save_task(task)

    payload = api_module.task_status_payload(task.task_id)

    assert "清华网盘预览/下载链接" not in payload["response_text"]
    assert payload["text"] == "这是最终回复。"


def test_task_status_payload_omits_cloud_link_when_artifacts_missing(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    session = service.store.create_session()
    session.cloud_workspace = CloudWorkspaceState(
        status="synced",
        configured=True,
        preview_url="https://cloud.example/d/session-link",
    )
    service.store.save_session(session)
    task = TaskRun(
        session_id=session.session_id,
        command="/chat",
        objective="hello",
        route_source="chat",
        workflow_title="Chat Response",
        status="completed",
        artifact_root=session.workspace_root,
        response_text="这是最终回复。",
    )
    service.store.save_task(task)

    payload = api_module.task_status_payload(task.task_id)

    assert "清华网盘预览/下载链接" not in payload["response_text"]


def test_chat_page_contains_task_polling_and_restore(service, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    client = TestClient(api_module.app)

    response = client.get("/chat")

    assert response.status_code == 200
    assert "setTimeout(pollTask, 1500)" in response.text
    assert "restoreTaskState();" in response.text
    assert "已批准，正在后台继续生成" in response.text
    assert 'id="new-session"' in response.text
    assert 'fetch("/api/sessions"' in response.text
    assert 'id="attach-file"' in response.text
    assert 'id="composer" class="composer"' in response.text
    assert 'composer.addEventListener("drop"' in response.text
    assert 'formData.append("target", "auto")' in response.text
    assert 'className = "msg system"' in response.text
    assert 'class="sidebar"' not in response.text
    assert 'id="drop-zone"' not in response.text
    assert 'id="upload-target"' not in response.text

