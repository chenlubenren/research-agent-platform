from __future__ import annotations

import asyncio
import json
from pathlib import Path

from reportlab.pdfgen import canvas

from research_agent_platform import agent as agent_module
from research_agent_platform import upstream as upstream_module
from research_agent_platform.agent import ResearchAgentService
from research_agent_platform.config import config
from research_agent_platform.memory.store import ResearchWikiStore


def _write_test_pdf(path: Path, text: str = "Graph neural network limitation and future work") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(path))
    document.drawString(72, 760, text)
    document.save()


def _stage_markdown(user_prompt: str, *, invalid_reference: bool = False) -> str:
    if "Current stage: Idea Candidates" in user_prompt:
        return (
            "# Idea Candidates\n\n"
            "# Problem Frame\n- Topic.\n\n"
            "# Candidate Ideas\n- A\n- B\n- C\n\n"
            "# Comparative Assessment\n- A best.\n\n"
            "# Recommended Candidate\n- A\n\n"
            "# Risks and Unknowns\n- Evidence coverage.\n\n"
            "# Decision Required\nNone."
        )
    if "Current stage: Idea Verification" in user_prompt:
        return (
            "# Idea Verification\n\n"
            "# Candidate Under Review\n- A\n\n"
            "# Closest Prior Work\n- Prior work.\n\n"
            "# Novelty Stress Test\n- Narrow the claim.\n\n"
            "# Feasibility Stress Test\n- Feasible.\n\n"
            "# Disconfirming Evidence\n- None confirmed.\n\n"
            "# Unresolved Questions\n- Data coverage.\n\n"
            "# Verification Verdict\n- Keep with caveats."
        )
    if "Current stage: Final Idea" in user_prompt:
        evidence = "[P001] and [P999]" if invalid_reference else "available Wiki evidence"
        return (
            "# 最终研究 Idea：示例方向\n\n"
            "## 一句话研究 Idea\n- 用一个最小机制解决目标问题。\n\n"
            "## 研究背景与核心问题\n- Topic.\n\n"
            "## 现有研究不足与可切入空白\n- Gap.\n\n"
            "## 核心假设与方法思路\n- A.\n\n"
            "## 预期创新与学术价值\n- B.\n\n"
            f"## 可证伪预测\n- C based on {evidence}.\n\n"
            "## 证据依据\n- Current evidence.\n\n"
            "## 适用边界、风险与不确定性\n- F.\n\n"
            "## 交给实验方案模块的下一步\n- Ready."
        )
    if "Using the template and the idea artifacts" in user_prompt:
        return "# Research Contract\n\n- Ready for /plan."
    return "# Wiki Artifact\n\n- Grounded summary."


def test_wiki_copies_pdf_and_creates_one_summary_per_paper(service: ResearchAgentService):
    session = service.store.create_session()
    source_pdf = Path(session.workspace_root, "paper", "uploads", "gnn-paper.pdf")
    _write_test_pdf(source_pdf)

    first = asyncio.run(service.chat(session.session_id, "/wiki 收录这篇 GNN 论文"))
    second = asyncio.run(service.chat(session.session_id, "/wiki 更新论文知识"))

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    paper_directories = list(Path(session.workspace_root, "wiki", "papers").glob("P-*"))
    assert len(paper_directories) == 1
    assert Path(paper_directories[0], "source.pdf").exists()
    summary = Path(paper_directories[0], "summary.md").read_text(encoding="utf-8")
    assert "## 作者讨论与局限" in summary
    assert "## 作者提出的未来工作" in summary
    assert Path(session.workspace_root, "wiki", "index.md").exists()
    query_pack = Path(session.workspace_root, "wiki", "query_pack.md")
    assert query_pack.exists()
    assert len(query_pack.read_text(encoding="utf-8")) <= 8000


def test_idea_routes_each_stage_to_configured_model(
    service: ResearchAgentService,
    monkeypatch,
):
    monkeypatch.setattr(config, "aris_repo_root", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setattr(config, "idea_generator_model", "generator-model")
    monkeypatch.setattr(config, "idea_critic_model", "critic-model")
    monkeypatch.setattr(config, "idea_final_model", "final-model")
    stage_models: dict[str, str | None] = {}
    stage_system_prompts: list[str] = []
    stage_system_prompts_by_name: dict[str, str] = {}
    stage_user_prompts: dict[str, str] = {}

    async def capture_models(*, system_prompt, user_prompt, model=None, temperature=0.3):
        for stage_name in ("Idea Candidates", "Idea Verification", "Final Idea"):
            if f"Current stage: {stage_name}" in user_prompt:
                stage_models[stage_name] = model
                stage_system_prompts.append(system_prompt)
                stage_system_prompts_by_name[stage_name] = system_prompt
                stage_user_prompts[stage_name] = user_prompt
        return _stage_markdown(user_prompt)

    monkeypatch.setattr(agent_module, "generate_text", capture_models)
    session = service.store.create_session()
    source_pdf = Path(session.workspace_root, "paper", "uploads", "prompt-safety.pdf")
    _write_test_pdf(source_pdf)
    result = asyncio.run(service.chat(session.session_id, "/idea 生成一个 GNN 研究方向"))

    assert result["status"] == "completed"
    assert stage_models == {
        "Idea Candidates": "generator-model",
        "Idea Verification": "critic-model",
        "Final Idea": "final-model",
    }
    assert len(stage_system_prompts) == 3
    assert all("Simplified Chinese" in prompt for prompt in stage_system_prompts)
    assert all("PRD excerpt" not in prompt for prompt in stage_system_prompts)
    assert "Locked recommended candidate" in stage_user_prompts["Idea Verification"]
    assert "Locked recommended candidate" in stage_user_prompts["Final Idea"]
    assert "The critic and finalizer must work on this exact candidate" in stage_user_prompts["Final Idea"]
    assert "Locked critic verdict" in stage_user_prompts["Final Idea"]
    assert all("%PDF-" not in prompt for prompt in stage_user_prompts.values())
    assert "strong graduate and doctoral researchers" in stage_user_prompts["Final Idea"]
    assert "## 一句话研究 Idea" not in stage_user_prompts["Final Idea"]
    assert "- 一句话研究 Idea" in stage_user_prompts["Final Idea"]
    assert "strongest plausible rejection argument" in stage_user_prompts["Final Idea"]
    assert "交给实验方案模块的下一步" in stage_system_prompts_by_name["Final Idea"]
    task = service.get_task(result["task_id"])
    final_idea = Path(task.artifact_root, "idea", "FINAL_IDEA.md").read_text(encoding="utf-8")
    assert "## 一句话研究 Idea" in final_idea
    assert "## 现有研究不足与可切入空白" in final_idea
    assert "## 适用边界、风险与不确定性" in final_idea
    assert "## Problem Anchor" not in final_idea
    trace = json.loads(Path(task.artifact_root, "Content", "IDEA_TRACE.json").read_text(encoding="utf-8"))
    assert [stage["model_role"] for stage in trace["stages"]] == [
        "idea_generator",
        "idea_critic",
        "idea_finalizer",
    ]


def test_kimi_k2_temperature_is_normalized(monkeypatch):
    captured_payload: dict = {}

    async def capture_request(method, path, payload=None, **_kwargs):
        captured_payload.update(payload or {})
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(upstream_module, "_request", capture_request)
    result = asyncio.run(
        upstream_module.chat_completions(
            {
                "model": "Kimi-K2.6",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.35,
            }
        )
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured_payload["model"] == "Kimi-K2.6"
    assert captured_payload["temperature"] == 1.0


def test_generated_markdown_keeps_one_complete_document():
    duplicated = (
        "# Final\n\n## Problem Anchor\nShort.\n\n## Method Thesis\nShort.\n"
        "\n```markdown\n# Final Again\n\n## Problem Anchor\n"
        + ("Long. " * 100)
        + "\n\n## Method Thesis\nLong.\n```"
    )

    normalized = ResearchAgentService._normalize_generated_markdown(
        duplicated,
        ["Problem Anchor", "Method Thesis"],
    )

    assert normalized.count("## Problem Anchor") == 1
    assert "Final Again" not in normalized
    assert "```markdown" not in normalized


def test_idea_stage_model_falls_back_to_upstream_model(
    service: ResearchAgentService,
    monkeypatch,
):
    monkeypatch.setattr(config, "idea_generator_model", "unavailable-generator")
    monkeypatch.setattr(config, "upstream_model", "default-model")
    calls: list[str | None] = []

    async def fail_then_fallback(*, system_prompt, user_prompt, model=None, temperature=0.3):
        calls.append(model)
        if model == "unavailable-generator":
            raise RuntimeError("model unavailable")
        return _stage_markdown(user_prompt)

    monkeypatch.setattr(agent_module, "generate_text", fail_then_fallback)
    result = asyncio.run(service.chat(None, "/idea 测试模型回退"))
    task = service.get_task(result["task_id"])
    trace = json.loads(Path(task.artifact_root, "Content", "IDEA_TRACE.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert calls[:2] == ["unavailable-generator", "default-model"]
    assert trace["stages"][0]["fallback_used"] is True
    assert trace["stages"][0]["model"] == "default-model"


def test_final_idea_marks_unknown_evidence_ids(
    service: ResearchAgentService,
    monkeypatch,
):
    session = service.store.create_session()
    evidence_map = Path(session.workspace_root, "bib", "EVIDENCE_MAP.md")
    evidence_map.parent.mkdir(parents=True, exist_ok=True)
    evidence_map.write_text("# Evidence Map\n\n- [P001] verified paper.\n", encoding="utf-8")

    async def generate_with_invalid_id(*, system_prompt, user_prompt, model=None, temperature=0.3):
        return _stage_markdown(user_prompt, invalid_reference=True)

    monkeypatch.setattr(agent_module, "generate_text", generate_with_invalid_id)
    result = asyncio.run(service.chat(session.session_id, "/idea 基于现有证据生成方向"))
    task = service.get_task(result["task_id"])
    final_idea = Path(task.artifact_root, "idea", "FINAL_IDEA.md").read_text(encoding="utf-8")
    trace = json.loads(Path(task.artifact_root, "Content", "IDEA_TRACE.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert "UNVERIFIED(P999)" in final_idea
    assert trace["evidence_validation"]["invalid_ids"] == ["P999"]


def test_final_idea_adds_known_evidence_page(service: ResearchAgentService, monkeypatch):
    session = service.store.create_session()
    evidence_map = Path(session.workspace_root, "bib", "EVIDENCE_MAP.md")
    evidence_map.parent.mkdir(parents=True, exist_ok=True)
    evidence_map.write_text("# Evidence Map\n\n- [PE-ABCDEF1234, p.8] future work.\n", encoding="utf-8")

    async def generate_with_page(*, system_prompt, user_prompt, model=None, temperature=0.3):
        content = _stage_markdown(user_prompt)
        if "Current stage: Final Idea" in user_prompt:
            content = content.replace("available Wiki evidence", "[PE-ABCDEF1234]")
        return content

    monkeypatch.setattr(agent_module, "generate_text", generate_with_page)
    result = asyncio.run(service.chat(session.session_id, "/idea 基于现有证据生成方向"))
    task = service.get_task(result["task_id"])
    final_idea = Path(task.artifact_root, "idea", "FINAL_IDEA.md").read_text(encoding="utf-8")
    trace = json.loads(Path(task.artifact_root, "Content", "IDEA_TRACE.json").read_text(encoding="utf-8"))

    assert "[PE-ABCDEF1234, p.8]" in final_idea
    assert trace["evidence_validation"]["page_linked_ids"] == ["PE-ABCDEF1234"]
    assert trace["evidence_validation"]["unpaged_ids"] == []


def test_wiki_query_pack_prioritizes_gap_sections(tmp_path: Path):
    summary = Path(tmp_path, "wiki", "papers", "P-ABCDEF123456", "summary.md")
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "# Example Paper\n\n"
        "## 基本信息\n\n- 原始文件: `paper/uploads/example.pdf`\n\n"
        "## 核心方法\n\n" + ("方法背景。" * 500) + "\n\n"
        "## 作者讨论与局限\n\n- 作者明确指出现有方法在深层训练时不稳定。[PE-1111111111, p.7]\n\n"
        "## 作者提出的未来工作\n\n- 作者建议未来结合注意力机制。[PE-2222222222, p.8]\n\n"
        "## 对 Idea 生成的提示\n\n- 优先验证作者明确提出的注意力方向。\n",
        encoding="utf-8",
    )

    query_pack = ResearchWikiStore(tmp_path).query_pack("研究 Gap future work", character_limit=1800)

    assert "作者提出的未来工作" in query_pack
    assert "PE-2222222222" in query_pack
    assert "作者讨论与局限" in query_pack


def test_idea_auto_ingests_uploaded_pdf_and_writes_back_to_wiki(
    service: ResearchAgentService,
    monkeypatch,
):
    session = service.store.create_session()
    source_pdf = Path(session.workspace_root, "paper", "uploads", "direct-idea.pdf")
    _write_test_pdf(source_pdf, "Discussion identifies a graph generalization limitation")

    async def generate_idea(*, system_prompt, user_prompt, model=None, temperature=0.3):
        return _stage_markdown(user_prompt)

    monkeypatch.setattr(agent_module, "generate_text", generate_idea)
    result = asyncio.run(service.chat(session.session_id, "/idea 基于上传论文提出方向"))
    task = service.get_task(result["task_id"])

    assert result["status"] == "completed"
    paper_directories = list(Path(task.artifact_root, "wiki", "papers").glob("P-*"))
    assert len(paper_directories) == 1
    assert Path(paper_directories[0], "source.pdf").exists()
    assert Path(paper_directories[0], "summary.md").exists()
    assert Path(task.artifact_root, "wiki", "ideas", f"{task.task_id}.md").exists()
    relations = Path(task.artifact_root, "wiki", "relations.jsonl").read_text(encoding="utf-8")
    assert "idea_based_on" in relations
