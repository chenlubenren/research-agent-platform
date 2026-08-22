from __future__ import annotations

import asyncio
import json

from research_agent_platform import api as api_module
from research_agent_platform.agent import ResearchAgentService

from .conftest import run


def test_agent_chat_returns_running_task_and_sse_delivers_events(service: ResearchAgentService, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)

    async def scenario():
        response = await api_module.api_agent_chat(
            {"session_id": None, "message": "/present 做一个中文汇报", "user_id": "local"}
        )
        task_id = response["task_id"]
        assert response["status"] == "running"
        assert "科研智能体(ResearchAgent)" in response["text"]
        running_task = api_module.background_tasks[task_id]

        stream_response = await api_module.api_task_events(task_id)
        events = []
        async for chunk in stream_response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            if "data: " not in chunk:
                continue
            event_name = chunk.split("event: ", 1)[1].split("\n", 1)[0]
            payload = json.loads(chunk.split("data: ", 1)[1].split("\n", 1)[0])
            events.append((event_name, payload))
            if event_name == "done":
                break

        await running_task
        assert events[0][0] == "snapshot"
        assert any(name == "progress" for name, _ in events)
        assert any(payload.get("event", {}).get("kind") == "model" for name, payload in events if name == "progress")
        assert events[-1][0] == "done"
        assert events[-1][1]["task"]["status"] in {"completed", "waiting_human", "failed"}

    asyncio.run(scenario())


def test_cloud_config_endpoint_exposes_disabled_reason(service: ResearchAgentService, monkeypatch):
    monkeypatch.setattr(api_module, "agent", service)
    payload = api_module.api_cloud_config()

    assert payload["enabled"] is False
    assert payload["configured"] is False
    assert "CLOUD_SYNC_ENABLED" in payload["hint"]


def test_plain_question_is_answered_without_background_file_task(
    service: ResearchAgentService, monkeypatch
):
    monkeypatch.setattr(api_module, "agent", service)

    async def no_workflow_route(*args, **kwargs):
        return "chat"

    monkeypatch.setattr("research_agent_platform.router.intent.generate_text", no_workflow_route)

    async def scenario():
        response = await api_module.api_agent_chat(
            {"session_id": None, "message": "请解释当前研究工作区如何组织", "user_id": "local"}
        )
        assert response["task_id"] == ""
        assert response["status"] == "idle"
        assert "科研智能体(ResearchAgent)" in response["text"]
        assert not api_module.background_tasks
        assert service.list_tasks(response["session_id"]) == []

    asyncio.run(scenario())


def test_short_artifact_keyword_is_not_enough_to_start_background_task(
    service: ResearchAgentService, monkeypatch
):
    monkeypatch.setattr(api_module, "agent", service)

    async def answer_question(**_kwargs):
        return "PPT 是演示文稿。"

    monkeypatch.setattr("research_agent_platform.agent.generate_text", answer_question)

    async def scenario():
        response = await api_module.api_agent_chat(
            {"session_id": None, "message": "PPT", "user_id": "local"}
        )
        assert response["task_id"] == ""
        assert response["status"] == "idle"
        assert service.list_tasks(response["session_id"]) == []

    asyncio.run(scenario())


