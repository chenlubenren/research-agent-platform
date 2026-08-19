from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from .connectors import ScholarSearchService, SeafileWorkspaceSync
from .connectors.scholar import LiteratureBundle, PaperRecord, merge_paper_records
from .artifacts.store import ArtifactStore
from .config import config
from .context import discover_pdf_sources, resolve_local_research_context
from .document_exports import (
    detect_write_formats,
    markdown_to_docx_bytes,
    markdown_to_latex,
    markdown_to_pdf_bytes,
)
from .graphs.runtime import LangGraphWorkflowRuntime
from .graphs.workflows import StageDefinition, WorkflowDefinition, workflow_registry
from .figure_pipeline import NoRenderableDataError, render_code_figure, select_data_sources
from .institutional_access import build_institutional_handoff, institutional_handoff_markdown
from .literature_downloads import download_public_pdfs, download_public_pdfs_from_sources
from .literature_sources import (
    discover_download_sources,
    download_targets_markdown,
    resolve_download_source_config,
)
from .memory.wiki import WikiService
from .memory.store import ResearchWikiStore, WikiPaper
from .models import (
    ApprovalCheckpoint,
    ChatSession,
    CloudWorkspaceState,
    MessageRecord,
    ProgressEvent,
    TaskRun,
    utc_now,
)
from .paper_pipeline import (
    build_paper_quality_reports,
    collect_paper_evidence,
    paper_evidence_prompt,
    paper_evidence_to_json,
    resolve_write_source_config,
    select_venue_profile,
)
from .presentation import (
    SlideRender,
    assemble_mixed_deck,
    build_slide_prompt,
    collect_workspace_evidence,
    compose_slide_preview,
    crop_slide_image,
    enforce_presentation_page_mix,
    extract_presentation_assets,
    parse_slide_content,
    parse_speaker_notes,
    presentation_design_spec_to_json,
    presentation_assets_from_json,
    presentation_assets_to_json,
    reconcile_slide_specs,
    resolve_presentation_source_config,
    select_slide_asset,
    select_presentation_template,
    slide_specs_to_json,
    speaker_notes_to_json,
    template_manifest,
)
from .rebuttal import (
    RebuttalInputError,
    build_rebuttal_closure_report,
    rebuttal_inputs_markdown,
    resolve_rebuttal_source_config,
)
from .review_clarify import (
    build_clarify_draft,
    build_clarify_pass2,
    clarified_scope_markdown,
    clarify_pass1_markdown,
    default_scope_card,
)
from .review_directions import (
    build_directions_markdown,
    directions_from_facets,
    directions_from_json,
    directions_to_json,
    discover_review_directions,
    focused_query_plan,
    focused_scope_markdown,
    interpret_selection,
)
from .review_pipeline import (
    ReviewEvidenceError,
    build_review_quality_reports,
    clean_review_topic,
    core_backfill_queries,
    extract_scope_artifact_names,
    inherited_domain_arg,
    parse_review_domains,
    parse_review_queries,
    preprint_ratio,
    resolve_key_source_domain,
    review_evidence_is_sufficient,
    review_quality_markdown,
    source_health,
)
from .review_screening import (
    parse_exclusion_terms,
    screen_papers,
    screening_gate_counts,
)
from .router.intent import (
    RouteDecision,
    explicit_route,
    is_approval_message,
    is_stop_message,
    route_message,
)
from .state.store import StateStore
from .upstream import configured_model_for_role, embed_texts, generate_image, generate_text


def _recount_screen_tiers(papers: list[PaperRecord]) -> dict:
    """Recount tier totals from the records themselves.

    Needed after a supplementary retrieval round merges newly screened papers into an
    already-screened pool, where the original per-call summary no longer covers the set.
    """

    counts = {"core": 0, "transferable": 0, "background": 0, "unscreened": 0}
    for paper in papers:
        if paper.screen_tier == "unscreened":
            counts["unscreened"] += 1
        elif paper.screen_decision == "keep" and paper.screen_tier in counts:
            counts[paper.screen_tier] += 1
        elif paper.screen_decision == "uncertain" and paper.screen_tier == "background":
            counts["background"] += 1
    counts["kept"] = sum(1 for paper in papers if paper.screen_decision == "keep")
    counts["dropped"] = sum(1 for paper in papers if paper.screen_decision == "drop")
    counts["uncertain"] = sum(1 for paper in papers if paper.screen_decision == "uncertain")
    return counts


def _discovery_fingerprint(scope_text: str, queries: list[str]) -> str:
    """Stable digest of the inputs that determine a discovery round's result.

    Re-running direction_selection with the same clarified scope and query plan should reuse
    the persisted bundle rather than re-hit the scholarly APIs; a scope edit changes this digest
    and forces a genuine re-run.
    """

    payload = "\u0001".join([scope_text.strip(), *[query.strip() for query in queries]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CLOUD_DELIVERY_SYSTEM_POLICY = (
    "The platform mirrors the complete session workspace to the configured Tsinghua Seafile library "
    "when the workspace is initialized and whenever a new artifact is generated. Cloud delivery is "
    "part of task completion: do not claim that an output has been delivered unless synchronization "
    "succeeds and the user-facing response includes the cloud preview/download URL. Never request, "
    "expose, or write cloud credentials into research artifacts."
)


class ResearchAgentService:
    def __init__(self) -> None:
        self.store = StateStore(config.state_root, config.artifact_root)
        self.artifacts = ArtifactStore(config.artifact_root)
        self.wiki = WikiService()
        self.scholar = ScholarSearchService(
            config.scholar_request_timeout_seconds,
            wos_api_base_url=config.wos_api_base_url,
            wos_api_key=config.wos_api_key,
            wos_default_db=config.wos_default_db,
            cnki_search_endpoint=config.cnki_search_endpoint,
            cnki_search_method=config.cnki_search_method,
            cnki_api_key=config.cnki_api_key,
            cnki_auth_header=config.cnki_auth_header,
            cnki_auth_scheme=config.cnki_auth_scheme,
            semantic_scholar_api_key=config.semantic_scholar_api_key,
            minimum_for_synthesis=config.review_minimum_sources,
            recommended_for_review=config.review_recommended_sources,
        )
        self.cloud = SeafileWorkspaceSync(
            enabled=config.cloud_sync_enabled,
            base_url=config.seafile_base_url,
            api_token=config.seafile_api_token,
            username=config.seafile_username,
            password=config.seafile_password,
            repo_id=config.seafile_repo_id,
            repo_name=config.seafile_repo_name,
            remote_root=config.seafile_remote_root,
            create_share_links=config.seafile_share_links,
            share_password=config.seafile_share_password,
            timeout_seconds=config.request_timeout_seconds,
            sync_retries=config.seafile_sync_retries,
        )
        self._cloud_sync_jobs: dict[str, asyncio.Task] = {}
        self._cloud_sync_pending: set[str] = set()
        self.workflows = workflow_registry()
        self.runtime = LangGraphWorkflowRuntime(self, self.workflows)

    async def chat(self, session_id: str | None, message: str, user_id: str | None = None) -> dict:
        session = await self._prepare_chat_session(session_id, message, user_id)
        session_context = self._session_context(session)

        waiting_task = self._find_waiting_task(session)
        if waiting_task is not None:
            if session.active_task_id != waiting_task.task_id:
                session.active_task_id = waiting_task.task_id
                self.store.save_session(session)
            reply = await self._handle_waiting_task(session, waiting_task, message)
        else:
            direct_reply = self._direct_chat_response("".join(message.lower().split()))
            if direct_reply is not None:
                reply = await self._chat_reply(session)
            else:
                route = await route_message(message, session_context)
                if route is None:
                    reply = await self._chat_reply(session)
                else:
                    reply = await self._start_task(session, message, route)

        latest_session = self.store.load_session(session.session_id) or session
        latest_session.history.append(MessageRecord(role="assistant", content=reply["text"]))
        self.store.save_session(latest_session)
        return {"session_id": latest_session.session_id, **reply}

    async def start_chat(
        self,
        session_id: str | None,
        message: str,
        user_id: str | None = None,
        *,
        sync_workspace: bool = True,
    ) -> dict:
        session = await self._prepare_chat_session(session_id, message, user_id, sync_workspace=sync_workspace)
        waiting_task = self._find_waiting_task(session)
        if waiting_task is not None:
            if session.active_task_id != waiting_task.task_id:
                session.active_task_id = waiting_task.task_id
                self.store.save_session(session)
            reply = await self._handle_waiting_task(session, waiting_task, message)
        else:
            route = explicit_route(message)
            if route is None:
                task = await self._create_chat_task(session, message, sync_workspace=sync_workspace)
                reply = self._build_reply(
                    task,
                    text=(
                        f"已接收问题，任务 `{task.task_id}` 正在后台处理。"
                        "路由、模型调用和回答完成状态将通过 SSE 推送。"
                    ),
                )
            else:
                task = await self._create_task(session, message, route, sync_workspace=sync_workspace)
                if task.status == "failed":
                    reply = self._build_reply(task, text=task.error or task.summary)
                else:
                    reply = self._build_reply(
                        task,
                        text="已经接收到您的请求，后台正在工作，请稍后...",
                    )
        latest_session = self.store.load_session(session.session_id) or session
        latest_session.history.append(MessageRecord(role="assistant", content=reply["text"]))
        self.store.save_session(latest_session)
        return {"session_id": latest_session.session_id, **reply}

    async def execute_chat_task(self, task_id: str) -> dict:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        session = self.store.load_session(task.session_id)
        if not session:
            raise ValueError(f"Unknown session: {task.session_id}")
        session_context = self._session_context(session)

        # A free-text reply reaching the async chat pipeline while another workflow is paused at a
        # human checkpoint must resume that workflow, not spawn a new task. _create_chat_task has
        # already overwritten active_task_id, so recover the suspended task by scanning the session.
        waiting_task = self._find_waiting_task(session, exclude_task_id=task.task_id)
        if waiting_task is not None:
            self._log_progress(
                task,
                f"检测到任务 {waiting_task.task_id} 正在等待人工输入，本条消息转交该任务继续，不新建任务",
                kind="router",
            )
            task.status = "completed"
            task.current_stage_name = "redirected"
            task.summary = f"消息已转交等待中的任务 {waiting_task.task_id} 继续。"
            self.store.save_task(task)
            session.active_task_id = waiting_task.task_id
            self.store.save_session(session)
            return await self._handle_waiting_task(session, waiting_task, task.objective)

        self._log_progress(task, "正在分析请求类型", kind="router")
        direct_reply = self._direct_chat_response("".join(task.objective.lower().split()))
        route = None if direct_reply is not None else await route_message(task.objective, session_context)
        if route is not None:
            task.command = route.command
            task.route_source = route.source
            task.workflow_title = self.workflows[route.command].title
            task.objective = self._strip_command(task.objective, route.command, route.command)
            self._configure_task_inputs(task, session)
            if task.status == "failed":
                return self._build_reply(task, text=task.error or task.summary)
            self.store.save_task(task)
            self._log_progress(
                task,
                f"已路由到 {route.command} | 来源: {route.source} | 原因: {route.reason}",
                kind="router",
            )
            return await self.execute_task(task_id)

        if direct_reply is not None:
            response = direct_reply
        else:
            self._log_progress(task, "正在调用上游模型生成回答", kind="model")
            response = (await self._chat_reply(session, latest_message=task.objective))["text"]
            self._log_progress(task, "上游模型已返回回答", kind="model")

        task.status = "completed"
        task.current_stage_name = "response"
        task.summary = "回答已生成。"
        task.response_text = response
        self._log_progress(task, "回答已生成", kind="response")
        session.active_task_id = None
        session.history.append(MessageRecord(role="assistant", content=response))
        self.store.save_task(task)
        self.store.save_session(session)
        await self.sync_task_workspace(task)
        return self._build_reply(task, text=response)

    async def execute_task(self, task_id: str) -> dict:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        if task.command == "/chat":
            return await self.execute_chat_task(task_id)
        workflow = self.workflows[task.command]
        try:
            result = await self.runtime.start_task(task, workflow)
        except ReviewEvidenceError as exc:
            self.record_task_failure(task.task_id, exc)
            failed_task = self.store.load_task(task.task_id) or task
            await self.sync_task_workspace(failed_task)
            result = self._build_reply(failed_task, text=str(exc))
        latest_session = self.store.load_session(task.session_id)
        if latest_session:
            latest_session.history.append(MessageRecord(role="assistant", content=result["text"]))
            self.store.save_session(latest_session)
        latest_task = self.store.load_task(task_id)
        if latest_task:
            latest_task.response_text = result["text"]
            self.store.save_task(latest_task)
        return result

    async def _prepare_chat_session(
        self, session_id: str | None, message: str, user_id: str | None, *, sync_workspace: bool = True
    ) -> ChatSession:
        session = self.store.get_or_create_session(session_id, user_id or "local")
        workspace_root = self.artifacts.session_root(
            user_id=session.user_id,
            session_id=session.session_id,
        )
        session.workspace_root = str(workspace_root.resolve())
        if sync_workspace and config.cloud_sync_enabled and session.cloud_workspace.status != "synced":
            await self.sync_session_workspace(session)
        session.history.append(MessageRecord(role="user", content=message))
        self.store.save_session(session)
        return session

    async def approve_task(self, task_id: str, feedback: str = "") -> dict:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        session = self.store.load_session(task.session_id)
        if not session:
            raise ValueError(f"Unknown session: {task.session_id}")
        result = await self._resume_after_approval(session, task, approved=True, feedback=feedback)
        latest_session = self.store.load_session(session.session_id) or session
        latest_session.history.append(MessageRecord(role="assistant", content=result["text"]))
        self.store.save_session(latest_session)
        return {"session_id": latest_session.session_id, **result}

    async def reject_task(self, task_id: str, feedback: str) -> dict:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        session = self.store.load_session(task.session_id)
        if not session:
            raise ValueError(f"Unknown session: {task.session_id}")
        result = await self._resume_after_approval(session, task, approved=False, feedback=feedback)
        latest_session = self.store.load_session(session.session_id) or session
        latest_session.history.append(MessageRecord(role="assistant", content=result["text"]))
        self.store.save_session(latest_session)
        return {"session_id": latest_session.session_id, **result}

    async def continue_task(self, task_id: str) -> dict:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        session = self.store.load_session(task.session_id)
        if not session:
            raise ValueError(f"Unknown session: {task.session_id}")
        workflow = self.workflows[task.command]
        result = await self.runtime.continue_task(task, workflow)
        latest_session = self.store.load_session(session.session_id) or session
        latest_session.history.append(MessageRecord(role="assistant", content=result["text"]))
        self.store.save_session(latest_session)
        return {"session_id": latest_session.session_id, **result}

    def mark_task_scheduled(self, task_id: str, action: str) -> TaskRun:
        task = self.store.load_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        task.status = "running"
        task.error = ""
        task.summary = "审批已接收，正在后台继续生成。" if action == "approve" else "修改意见已接收，正在后台重新生成。"
        self._log_progress(task, task.summary)
        self.store.save_task(task)
        return task

    def record_task_failure(self, task_id: str, error: BaseException) -> None:
        task = self.store.load_task(task_id)
        if not task:
            return
        if isinstance(error, asyncio.CancelledError):
            task.status = "running"
            task.error = ""
            task.summary = "服务停止时后台任务被中断，将在服务启动后从断点恢复。"
            self._log_progress(task, task.summary)
            self.store.save_task(task)
            return
        detail = str(error).strip() or error.__class__.__name__
        task.status = "failed"
        task.error = detail
        task.summary = f"任务执行失败：{detail}"
        self._log_progress(task, task.summary)
        self.store.save_task(task)
        session = self.store.load_session(task.session_id)
        if session and session.active_task_id == task.task_id:
            session.active_task_id = None
            self.store.save_session(session)

    def get_task(self, task_id: str) -> TaskRun | None:
        return self.store.load_task(task_id)

    def list_tasks(self, session_id: str | None = None) -> list[TaskRun]:
        return self.store.list_tasks(session_id)

    def _find_waiting_task(self, session: ChatSession, *, exclude_task_id: str = "") -> TaskRun | None:
        """The task, if any, this session has paused at a human checkpoint.

        ``active_task_id`` is the fast path, but creating a follow-up chat task overwrites it and
        orphans a suspended ``/review``; a free-text reply would then be re-routed as a brand-new
        task and the review would hang forever. Falling back to a scan keeps a checkpoint reply
        attached to its own workflow even after ``active_task_id`` was clobbered.
        """

        if session.active_task_id and session.active_task_id != exclude_task_id:
            active = self.store.load_task(session.active_task_id)
            if active and active.status == "waiting_human":
                return active
        for candidate in self.store.list_tasks(session.session_id):
            if candidate.task_id == exclude_task_id:
                continue
            if candidate.status == "waiting_human":
                return candidate
        return None

    async def _chat_reply(self, session: ChatSession, latest_message: str | None = None) -> dict:
        latest_message = latest_message if latest_message is not None else session.history[-1].content if session.history else ""
        direct_reply = self._direct_chat_response(latest_message)
        if direct_reply:
            return {
                "text": direct_reply,
                "task_id": "",
                "status": "idle",
                "command": "",
                "workflow_title": "",
                "artifact_root": "",
                "artifacts": [],
                "progress": [],
                "checkpoint": None,
            }
        artifact_reply = self._artifact_location_reply(session, latest_message)
        if artifact_reply is not None:
            return {
                "text": artifact_reply,
                "task_id": "",
                "status": "idle",
                "command": "",
                "workflow_title": "",
                "artifact_root": "",
                "artifacts": [],
                "progress": [],
                "checkpoint": None,
            }
        history = session.history[-8:]
        if latest_message and (not history or history[-1].content != latest_message):
            history = [*history, MessageRecord(role="user", content=latest_message)]
        session_context = self._session_context(session)
        system_prompt = (
            "You are Research Agent Platform, a research workflow assistant rather than a generic AI chatbot. "
            "Reply in Chinese when the user writes Chinese. Keep answers concise, useful, and concrete. "
            "If the user asks who you are, explicitly say you are a 科研智能体 / Research Agent for literature review, "
            "idea discovery, experiment planning, paper drafting, rebuttal, and research memory. Mention commands such as "
            "/review, /idea, /plan, /code, /write, /rebuttal, /fig, /present, and /wiki when relevant. "
            "Do not describe yourself as a generic assistant. Do not create a workflow task unless the user explicitly requests one. "
            f"{CLOUD_DELIVERY_SYSTEM_POLICY}"
        )
        if session_context:
            system_prompt += f"\n\nSession memory:\n{session_context}"
        content = await generate_text(
            system_prompt=system_prompt,
            user_prompt="\n".join(f"{item.role}: {item.content}" for item in history),
            temperature=0.2,
        )
        return {
            "text": content
            or (
                "我是科研智能体 Research Agent，不是通用助手。"
                "我可以处理文献梳理、选题、实验规划、论文写作、审稿回复和研究记忆，也支持 "
                "/review、/idea、/plan、/code、/write、/rebuttal、/fig、/present、/wiki 等工作流。"
            ),
            "task_id": "",
            "status": "idle",
            "command": "",
            "workflow_title": "",
            "artifact_root": "",
            "artifacts": [],
            "progress": [],
            "checkpoint": None,
        }

    def _artifact_location_reply(self, session: ChatSession, message: str) -> str | None:
        normalized = "".join(message.lower().split())
        if not normalized:
            return None
        if not self._looks_like_artifact_location_question(normalized):
            return None

        session_root = Path(session.workspace_root).resolve() if session.workspace_root else None
        tasks = [
            task
            for task in self.store.list_tasks(session.session_id)
            if task.artifact_root and Path(task.artifact_root).exists()
        ]
        if not tasks and session_root is None:
            return None

        figure_task = next(
            (
                task
                for task in tasks
                if task.command == "/fig"
                or any(artifact.relative_path.startswith("figures/generated/") for artifact in task.artifacts)
            ),
            None,
        )
        if figure_task:
            root = Path(figure_task.artifact_root).resolve()
            figure_path = root / "figures" / "generated" / "FIGURE_01.png"
            if figure_path.exists():
                return (
                    f"图已输出到 `{figure_path}`。\n"
                    "具体文件名是 `figures/generated/FIGURE_01.png`。\n\n"
                    f"这个会话的工作区根目录是 `{root}`，图片默认保存在 `figures/generated/` 下。"
                )
            return (
                f"图产物在 `{root}` 的 `figures/generated/` 目录里。\n"
                "如果你要找具体文件，可以去看 `FIGURE_01.png` 和 `FIGURE_DELIVERY.json`。"
            )

        if session_root is not None:
            return (
                f"这个会话的工作区根目录是 `{session_root}`。\n"
                "如果是图产物，默认看 `figures/generated/`；如果是论文产物，默认看 `paper/`。"
            )
        return None

    def _session_context(self, session: ChatSession, *, task_limit: int = 3, artifact_limit: int = 3) -> str:
        tasks = self.store.list_tasks(session.session_id)
        if not tasks:
            return ""
        blocks: list[str] = []
        total_chars = 0
        for task in tasks[:task_limit]:
            lines = [
                f"## Task {task.task_id}",
                f"Command: {task.command}",
                f"Objective: {task.objective}",
                f"Status: {task.status}",
            ]
            if task.summary:
                lines.append(f"Summary: {task.summary}")
            if task.response_text:
                lines.append(f"Response: {task.response_text[:800]}")
            for artifact in task.artifacts[:artifact_limit]:
                excerpt = self._artifact_excerpt(task, artifact.relative_path)
                if excerpt:
                    lines.append(f"### {artifact.relative_path}")
                    lines.append(excerpt[:1000])
            block = "\n".join(lines)
            total_chars += len(block)
            if total_chars > 9000:
                break
            blocks.append(block)
        return "\n\n".join(blocks)

    def _looks_like_artifact_location_question(self, normalized_message: str) -> bool:
        location_terms = (
            "where",
            "saved",
            "save",
            "output",
            "download",
            "link",
            "path",
            "file",
            "folder",
            "directory",
            "artifact",
            "在哪里",
            "在哪",
            "哪里",
            "哪儿",
            "输出到",
            "保存到",
            "放到",
            "存到",
            "路径",
            "位置",
            "下载",
            "链接",
            "文件",
            "文件夹",
            "目录",
            "产物",
        )
        artifact_terms = (
            "figure",
            "image",
            "plot",
            "chart",
            "diagram",
            "artifact",
            "output",
            "paper",
            "docx",
            "pdf",
            "ppt",
            "png",
            "svg",
            "图",
            "图片",
            "图像",
            "图表",
            "示意图",
            "流程图",
            "产物",
            "论文",
            "文档",
            "文件",
            "结果",
        )
        followup_terms = (
            "要求的",
            "刚才",
            "上个",
            "上一",
            "前面",
            "生成的",
            "输出的",
            "保存的",
            "that",
            "the",
            "last",
            "previous",
            "generated",
        )
        has_location = any(term in normalized_message for term in location_terms)
        has_artifact = any(term in normalized_message for term in artifact_terms)
        has_followup = any(term in normalized_message for term in followup_terms)
        return has_location and has_artifact and (has_followup or len(normalized_message) <= 80)

    def _direct_chat_response(self, message: str) -> str | None:
        normalized = "".join(message.lower().split())
        if self._is_greeting_message(normalized):
            return (
                "你好，我是科研智能体 Research Agent。"
                "我可以帮你做文献、选题、实验、写作和审稿回复。"
                "要直接开始，可以输入 /review、/idea、/plan、/code、/write、/rebuttal、/fig、/present 或 /wiki。"
            )
        if self._is_identity_question(normalized):
            return (
                "我是科研智能体 Research Agent，不是通用聊天助手。\n\n"
                "我主要用于文献梳理、选题发现、实验规划、代码实现协同、论文写作、审稿回复和研究记忆管理。\n\n"
                "你可以直接提问，也可以用 /review、/idea、/plan、/code、/write、/rebuttal、/fig、/present、/wiki 启动对应科研流程。"
                "进入多阶段任务后，我会显示路由结果、阶段进度、待审批 checkpoint 和产物文件路径。"
            )
        if self._is_capability_question(normalized):
            return (
                "我更适合科研工作流，不是泛用闲聊机器人。\n\n"
                "常见用法是：/review 做文献调研，/idea 做选题，/plan 做实验方案，/code 做实现与实验执行，"
                "/write 产出论文草稿，/rebuttal 处理审稿意见，/wiki 回看本地研究记忆。"
            )
        return None

    def _is_identity_question(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        keywords = (
            "你是谁",
            "你是干什么的",
            "介绍一下你自己",
            "自我介绍",
            "whoareyou",
            "whatareyou",
            "introduceyourself",
        )
        return any(keyword in normalized_message for keyword in keywords)

    def _is_greeting_message(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        greetings = (
            "hello",
            "hi",
            "hey",
            "你好",
            "您好",
            "嗨",
            "在吗",
            "在不在",
        )
        return len(normalized_message) <= 24 and any(
            normalized_message.startswith(greeting) for greeting in greetings
        )

    def _is_capability_question(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        keywords = (
            "你会什么",
            "你能做什么",
            "可以做什么",
            "怎么用你",
            "whatcanyoudo",
            "howtouseyou",
        )
        return any(keyword in normalized_message for keyword in keywords)

    async def _handle_waiting_task(self, session: ChatSession, task: TaskRun, message: str) -> dict:
        if is_stop_message(message):
            task.status = "failed"
            task.error = "Stopped by user during human checkpoint."
            session.active_task_id = None
            self.store.save_task(task)
            return self._build_reply(
                task,
                text=f"任务 `{task.task_id}` 已停止。已有产物保留在 `{task.artifact_root}`。",
            )

        if is_approval_message(message):
            return await self._resume_after_approval(session, task, approved=True, feedback="")

        return await self._resume_after_approval(session, task, approved=False, feedback=message)

    async def _start_task(
        self,
        session: ChatSession,
        message: str,
        route: RouteDecision,
        *,
        sync_workspace: bool = True,
    ) -> dict:
        task = await self._create_task(session, message, route, sync_workspace=sync_workspace)
        if task.status == "failed":
            return self._build_reply(task, text=task.error or task.summary)
        try:
            return await self.execute_task(task.task_id)
        except ReviewEvidenceError as exc:
            self.record_task_failure(task.task_id, exc)
            failed_task = self.store.load_task(task.task_id) or task
            await self.sync_task_workspace(failed_task)
            return self._build_reply(failed_task, text=str(exc))

    async def _create_task(
        self,
        session: ChatSession,
        message: str,
        route: RouteDecision,
        *,
        sync_workspace: bool = True,
    ) -> TaskRun:
        workflow = self.workflows[route.command]
        task = TaskRun(
            session_id=session.session_id,
            user_id=session.user_id,
            command=workflow.command,
            objective=self._strip_command(message, workflow.command, route.command),
            route_source=route.source,
            workflow_title=workflow.title,
        )
        task_root = self.artifacts.task_root(
            task.task_id,
            user_id=task.user_id,
            session_id=task.session_id,
        )
        task.artifact_root = str(task_root.resolve())
        if sync_workspace:
            await self.sync_session_workspace(session, task=task)
        self._configure_task_inputs(task, session, task_root=task_root)
        self._log_progress(task, f"已路由到 {route.command} | 来源: {route.source} | 原因: {route.reason}")
        session.active_task_id = task.task_id
        self.store.save_task(task)
        self.store.save_session(session)
        return task

    def _configure_task_inputs(
        self,
        task: TaskRun,
        session: ChatSession,
        *,
        task_root: Path | None = None,
    ) -> None:
        root = task_root or Path(task.artifact_root)
        if task.command == "/present":
            task.presentation_source = resolve_presentation_source_config(
                task.objective,
                root,
                session.upload_batches,
                source_limit=config.presentation_source_limit,
            )
        if task.command == "/write":
            task.write_source = resolve_write_source_config(
                task.objective,
                root,
                session.upload_batches,
                source_limit=config.write_source_limit,
            )
        if task.command == "/download":
            task.download_source = resolve_download_source_config(
                task.objective,
                root,
                session.upload_batches,
                source_limit=config.review_download_limit,
            )
        if task.command == "/rebuttal":
            try:
                task.rebuttal_source = resolve_rebuttal_source_config(
                task.objective,
                root,
                session.upload_batches,
            )
            except RebuttalInputError as exc:
                task.status = "failed"
                task.error = str(exc)
                task.summary = str(exc)
                self._log_progress(task, f"Rebuttal input validation failed: {exc}")
                session.active_task_id = None
                self.store.save_task(task)
                self.store.save_session(session)
                return task

    async def _create_chat_task(
        self,
        session: ChatSession,
        message: str,
        *,
        sync_workspace: bool = True,
    ) -> TaskRun:
        task_root = self.artifacts.task_root(
            f"chat-{session.session_id}",
            user_id=session.user_id,
            session_id=session.session_id,
        )
        task = TaskRun(
            session_id=session.session_id,
            user_id=session.user_id,
            command="/chat",
            objective=message,
            route_source="chat",
            workflow_title="Chat Response",
            artifact_root=str(task_root.resolve()),
            current_stage_name="routing",
        )
        if sync_workspace:
            await self.sync_session_workspace(session, task=task)
        self._log_progress(task, "问题已接收，准备分析请求类型", kind="router")
        session.active_task_id = task.task_id
        self.store.save_task(task)
        self.store.save_session(session)
        return task

    async def _resume_after_approval(
        self, session: ChatSession, task: TaskRun, *, approved: bool, feedback: str
    ) -> dict:
        workflow = self.workflows[task.command]
        if self.runtime.has_graph_state(task.task_id):
            return await self.runtime.resume_task(
                task,
                workflow,
                approved=approved,
                feedback=feedback,
            )

        if not task.approvals:
            raise ValueError(f"Task {task.task_id} has no pending checkpoint.")
        checkpoint = task.approvals[-1]

        if approved:
            checkpoint.status = "approved"
            checkpoint.feedback = feedback
            checkpoint.resolved_at = checkpoint.resolved_at or checkpoint.created_at
            task.status = "running"
            self._log_progress(task, f"已批准 checkpoint: {checkpoint.title}")
            task.current_stage_index = checkpoint.stage_index + 1
            if task.current_stage_index < len(workflow.stage_definitions):
                task.current_stage_name = workflow.stage_definitions[task.current_stage_index].name
            else:
                task.current_stage_name = ""
            self.store.save_task(task)
            return await self._run_task(task, workflow, revision_feedback=feedback)

        checkpoint.status = "rejected"
        checkpoint.feedback = feedback
        checkpoint.resolved_at = checkpoint.resolved_at or checkpoint.created_at
        task.status = "running"
        self._log_progress(task, f"已打回 checkpoint: {checkpoint.title} | 反馈: {feedback}")
        self.store.save_task(task)
        return await self._rerun_checkpoint_stage(task, workflow, feedback)

    async def _rerun_checkpoint_stage(
        self, task: TaskRun, workflow: WorkflowDefinition, feedback: str
    ) -> dict:
        stage_index = task.approvals[-1].stage_index
        stage = workflow.stage_definitions[stage_index]
        self._log_progress(task, f"正在根据反馈重生成阶段: {stage.title}")
        artifact = await self._execute_stage(task, workflow, stage, feedback)
        task.artifacts.append(artifact)
        checkpoint = self._make_checkpoint(task, stage, feedback)
        task.approvals.append(checkpoint)
        task.status = "waiting_human"
        task.current_stage_index = stage_index
        task.current_stage_name = stage.name
        task.summary = f"{stage.title} regenerated with human feedback."
        self._log_progress(task, f"阶段已重生成: {stage.title}")
        self.store.save_task(task)
        await self.sync_task_workspace(task)
        return self._build_reply(
            task,
            text=f"{stage.title} 已根据反馈重生成。请检查更新后的产物，确认后继续，或再次打回。",
            checkpoint=checkpoint,
        )

    async def _run_task(self, task: TaskRun, workflow: WorkflowDefinition, revision_feedback: str) -> dict:
        starting_index = task.current_stage_index
        for stage_index in range(task.current_stage_index, len(workflow.stage_definitions)):
            stage = workflow.stage_definitions[stage_index]
            task.current_stage_index = stage_index
            task.current_stage_name = stage.name
            self._log_progress(task, f"阶段开始: {stage.title}")
            self.store.save_task(task)
            artifact = await self._execute_stage(
                task,
                workflow,
                stage,
                revision_feedback if stage_index == starting_index else "",
            )
            task.artifacts.append(artifact)
            self._log_progress(task, f"阶段完成: {stage.title} -> {artifact.relative_path}")
            await self.sync_task_workspace(task)
            if self._stage_requires_checkpoint(task, stage):
                checkpoint = self._make_checkpoint(task, stage, revision_feedback)
                task.approvals.append(checkpoint)
                task.status = "waiting_human"
                task.summary = f"Waiting for human approval at {stage.title}."
                self._log_progress(task, f"等待人工审批: {checkpoint.title}")
                self.store.save_task(task)
                return self._build_reply(
                    task,
                    text=f"{stage.title} 已完成，正在等待你的审核。你可以批准继续，或填写修改意见后打回本阶段。",
                    checkpoint=checkpoint,
                )

        if task.command == "/idea":
            await self._write_research_contract(task)
        if task.command == "/fig":
            task.artifacts.extend(await self._write_figure_delivery_artifacts(task))
        if task.command == "/write":
            task.artifacts.extend(await self._write_delivery_artifacts(task))
        if task.command == "/rebuttal":
            task.artifacts.extend(self._write_rebuttal_delivery_artifacts(task))
        if task.command == "/review":
            task.artifacts.extend(self._write_review_delivery_artifacts(task))
        if task.command == "/present":
            await self._write_presentation_delivery_artifacts(task)
        wiki_note = self._finalize_task_record(task, workflow)
        self.store.save_task(task)
        cloud_workspace = await self.sync_task_workspace(task)
        self.enforce_cloud_delivery(task, cloud_workspace)
        if task.status != "failed":
            self._mark_task_completed(task, workflow)
        session = self.store.load_session(task.session_id)
        if session:
            session.active_task_id = None
            self.store.save_session(session)
        return self._build_reply(
            task,
            text=(
                f"{workflow.title} 已完成。所有研究产物均保存在本会话工作区 `{task.artifact_root}`。"
                f"任务记录位于 `{wiki_note.relative_path}`。"
            ),
        )

    async def _execute_stage(
        self, task: TaskRun, workflow: WorkflowDefinition, stage: StageDefinition, revision_feedback: str
    ):
        support_artifacts, support_context = await self._prepare_stage_support(task, stage)
        for artifact in support_artifacts:
            self._upsert_task_artifact(task, artifact)
        if task.command == "/review" and stage.name == "clarify":
            artifact = await self._run_review_clarify(task, stage, revision_feedback)
            self.artifacts._write_manifest_for_root(Path(task.artifact_root))
            return artifact
        if task.command == "/review" and stage.name == "direction_selection":
            artifact = await self._run_direction_selection(task, stage, revision_feedback)
            self.artifacts._write_manifest_for_root(Path(task.artifact_root))
            return artifact
        if task.command == "/rebuttal" and stage.name == "rebuttal_intake":
            if task.rebuttal_source is None:
                raise RebuttalInputError("/rebuttal input SourceSet is missing.")
            content = rebuttal_inputs_markdown(task.rebuttal_source)
            artifact = self._write_text(
                task,
                stage.artifact_path,
                content,
                kind=stage.artifact_kind,
                description=f"{workflow.title} / {stage.title}",
            )
            self.artifacts._write_manifest_for_root(Path(task.artifact_root))
            return artifact
        if task.command == "/write" and stage.name == "paper_evidence":
            records = self._paper_evidence_records(task)
            artifact = self._write_text(
                task,
                stage.artifact_path,
                paper_evidence_to_json(records),
                kind=stage.artifact_kind,
                description=f"{workflow.title} / {stage.title}",
            )
            metadata_artifact = self._write_text(
                task,
                "Content/PAPER_EVIDENCE_METADATA.json",
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "source_refs": list(task.write_source.source_refs if task.write_source else []),
                        "evidence_count": len(records),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                kind="note",
                description="Cache identity for the frozen paper evidence map.",
            )
            task.artifacts.append(metadata_artifact)
            self.artifacts._write_manifest_for_root(Path(task.artifact_root))
            return artifact
        prompt = self._build_stage_prompt(task, workflow, stage, revision_feedback)
        if support_context:
            prompt["user"] = support_context + "\n" + prompt["user"]
        self._log_progress(task, f"正在调用上游模型生成阶段内容: {stage.title}", kind="model")
        selected_model = configured_model_for_role(stage.model_role)
        fallback_model = config.upstream_model or None
        fallback_used = False
        started = time.monotonic()
        try:
            content = await generate_text(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                model=selected_model,
                temperature=0.35,
            )
        except Exception:
            if (
                task.command != "/idea"
                or not fallback_model
                or not selected_model
                or fallback_model == selected_model
            ):
                raise
            fallback_used = True
            self._log_progress(
                task,
                f"阶段模型不可用，回退到默认模型: {stage.title}",
                kind="model",
            )
            content = await generate_text(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                model=fallback_model,
                temperature=0.35,
            )
        if not content.strip():
            raise RuntimeError(f"Empty content from upstream for stage {stage.name}")
        content = self._normalize_generated_markdown(content, stage.required_sections)
        evidence_validation: dict | None = None
        if task.command == "/idea" and stage.name == "final_idea":
            content, evidence_validation = self._validate_idea_evidence(task, content)
        if task.command == "/idea":
            self._record_idea_trace(
                task,
                stage,
                selected_model=fallback_model if fallback_used else selected_model,
                elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                fallback_used=fallback_used,
                evidence_validation=evidence_validation,
            )
        self._log_progress(task, f"上游模型已返回阶段内容: {stage.title}", kind="model")
        artifact = self._write_text(
            task,
            stage.artifact_path,
            content,
            kind=stage.artifact_kind,
            description=f"{workflow.title} / {stage.title}",
        )
        self.artifacts._write_manifest_for_root(Path(task.artifact_root))
        return artifact

    def _stage_requires_checkpoint(self, task: TaskRun, stage: StageDefinition) -> bool:
        if not config.enable_hitl or not stage.hitl:
            return False
        objective = task.objective.lower()
        explicit_review = (
            r"(?:等待|等我).{0,12}(?:确认|审核|批准)",
            r"(?:先|必须先)(?:让我)?(?:确认|审核|批准).{0,12}(?:再|然后)(?:继续|生成|执行)",
            r"(?:确认|审核|批准).{0,12}(?:后再|之后再|再继续|再生成|再执行)",
            r"\b(?:wait for|require)\s+(?:my\s+)?(?:review|approval)\b",
            r"\bdo not continue (?:before|until)\s+(?:my\s+)?(?:review|approval)\b",
        )
        if any(re.search(pattern, objective, re.I) for pattern in explicit_review):
            return True
        if not stage.hitl:
            return False
        return bool(self._blocking_decision_section(task, stage))

    def _blocking_decision_section(self, task: TaskRun, stage: StageDefinition) -> str:
        artifact_path = Path(task.artifact_root) / stage.artifact_path
        if not artifact_path.exists():
            return ""
        content = artifact_path.read_text(encoding="utf-8", errors="ignore")
        sections = re.finditer(
            r"(?:^|\n)#{1,6}\s*(?:Decision Required|需要用户决策|待用户选择)\s*\n"
            r"(.*?)(?=\n#{1,6}\s|\Z)",
            content,
            re.I | re.S,
        )
        for section in sections:
            decision = section.group(1).strip()
            decision_plain = re.sub(r"[`*_]", "", decision).strip()
            if re.match(r"(?is)^none(?:\s|[.!。]|$)", decision_plain):
                continue
            normalized = re.sub(r"[`*_\s.。:：-]", "", decision).lower()
            if normalized in {"", "none", "no", "n/a", "na", "无", "无需", "不需要"}:
                continue
            blocking = re.search(
                r"(?mi)^\s*(?:[-*+]\s*)?(?:blocking|是否阻塞|必须由用户决定)\s*[:：]\s*"
                r"(?:yes|true|是|需要|必须)\s*[.!。]?\s*$",
                decision,
            )
            reason = re.search(
                r"(?mi)^\s*(?:[-*+]\s*)?(?:why user (?:input is necessary|must decide)|"
                r"用户必须决定的原因|必须由用户决定的原因|决策原因)\s*[:：]\s*(\S.+)$",
                decision,
            )
            recommended_default = re.search(
                r"(?mi)^\s*(?:[-*+]\s*)?(?:recommended default|推荐默认(?:方案)?)\s*[:：]\s*(\S.+)$",
                decision,
            )
            named_options = set(
                match.lower()
                for match in re.findall(
                    r"(?mi)^\s*(?:[-*+]\s*)?(?:option|方案|选项)\s*"
                    r"([A-Z一二三四五六七八九十\d]+)\s*[:：.-]\s*\S+.*$",
                    decision,
                )
            )
            if blocking and reason and recommended_default and len(named_options) >= 2:
                return decision
        return ""

    async def _prepare_stage_support(self, task: TaskRun, stage: StageDefinition) -> tuple[list, str]:
        support_artifacts: list = []
        support_context_parts: list[str] = []
        if task.command == "/present":
            presentation_artifacts, presentation_context = self._prepare_presentation_support(task)
            support_artifacts.extend(presentation_artifacts)
            if presentation_context:
                support_context_parts.append(presentation_context)
            return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/rebuttal":
            rebuttal_artifacts, rebuttal_context = self._prepare_rebuttal_support(task)
            support_artifacts.extend(rebuttal_artifacts)
            if rebuttal_context:
                support_context_parts.append(rebuttal_context)
            return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/write":
            write_artifacts, write_context = self._prepare_write_support(task)
            support_artifacts.extend(write_artifacts)
            if write_context:
                support_context_parts.append(write_context)
            return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/download":
            download_artifacts, download_context = await self._prepare_download_support(task, stage)
            support_artifacts.extend(download_artifacts)
            if download_context:
                support_context_parts.append(download_context)
            return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/review":
            return await self._prepare_review_support(task, stage)
        wiki_has_evidence = False
        if task.command in {"/wiki", "/idea"}:
            wiki_artifacts, wiki_context, wiki_has_evidence = await self._prepare_research_wiki_support(
                task,
                stage,
            )
            support_artifacts.extend(wiki_artifacts)
            if wiki_context:
                support_context_parts.append(wiki_context)
            if task.command == "/wiki":
                return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/idea" and stage.name in {"idea_verification", "final_idea"}:
            candidate_lock = self._idea_candidate_lock(task)
            if candidate_lock:
                support_context_parts.append(candidate_lock)
        if task.command not in {"/review", "/idea"}:
            return support_artifacts, "\n\n".join(support_context_parts)
        if task.command == "/idea":
            evidence_paths = (
                Path(task.artifact_root) / "bib" / "EVIDENCE_MAP.md",
                Path(task.artifact_root) / "bib" / "LITERATURE_REVIEW.md",
            )
            existing_evidence = next((path for path in evidence_paths if path.exists()), None)
            if existing_evidence:
                support_context_parts.append(self._file_excerpt(str(existing_evidence), 5000))
                return support_artifacts, "\n\n".join(support_context_parts)
            if wiki_has_evidence:
                return support_artifacts, "\n\n".join(support_context_parts)
        query = self._search_query_for_task(task.objective)
        if not query:
            return support_artifacts, "\n\n".join(support_context_parts)
        md_path = Path(task.artifact_root) / "bib" / "LITERATURE_SEARCH.md"
        json_path = Path(task.artifact_root) / "bib" / "LITERATURE_SEARCH.json"
        if md_path.exists() and json_path.exists():
            support_context_parts.append(self._file_excerpt(str(md_path), 5000))
            return support_artifacts, "\n\n".join(support_context_parts)
        bundle = await self.scholar.search_bundle(query, per_source_limit=config.scholar_results_per_source)
        md_artifact = self._write_text(
            task,
            "bib/LITERATURE_SEARCH.md",
            bundle.to_markdown(),
            kind="note",
            description="Aggregated scholarly search results for the task.",
        )
        json_artifact = self._write_text(
            task,
            "bib/LITERATURE_SEARCH.json",
            bundle.to_json(),
            kind="note",
            description="Structured scholarly search results for the task.",
        )
        support_artifacts.extend([md_artifact, json_artifact])
        support_context_parts.append(bundle.prompt_excerpt())
        return support_artifacts, "\n\n".join(support_context_parts)

    def _idea_candidate_lock(self, task: TaskRun) -> str:
        candidates = self._artifact_text(task, "idea/IDEA_CANDIDATES.md")
        recommended = self._markdown_section(candidates, "Recommended Candidate")
        if not recommended:
            return ""
        parts = [
            "Locked recommended candidate:\n" + recommended,
            "The critic and finalizer must work on this exact candidate. Do not substitute another candidate.",
        ]
        verification = self._artifact_text(task, "idea/IDEA_VERIFICATION.md")
        verdict = self._markdown_section(verification, "Verification Verdict")
        if verdict:
            parts.append("Locked critic verdict:\n" + verdict)
        return "\n\n".join(parts)

    @staticmethod
    def _markdown_section(content: str, title: str) -> str:
        match = re.search(
            rf"(?ms)^#{{1,6}}\s*{re.escape(title)}[^\n]*\n(.*?)(?=^#{{1,6}}\s|\Z)",
            content,
        )
        return match.group(1).strip() if match else ""

    async def _prepare_research_wiki_support(
        self,
        task: TaskRun,
        stage: StageDefinition,
    ) -> tuple[list, str, bool]:
        workspace_root = Path(task.artifact_root)
        wiki_store = ResearchWikiStore(workspace_root)
        artifacts: list = []
        source_refs = discover_pdf_sources(workspace_root)
        for source_ref in source_refs:
            paper = wiki_store.paper_for_source(source_ref)
            wiki_store.ensure_pdf_copy(paper)
            artifacts.append(
                self._record_existing(
                    task,
                    paper.wiki_pdf_relative_path,
                    kind="document",
                    description=f"Wiki PDF copy for {paper.paper_id}.",
                )
            )
            evidence_records = collect_paper_evidence(
                workspace_root,
                [source_ref],
                total_limit=18000,
                query=task.objective,
                prioritize_research_sections=True,
            )
            if not wiki_store.summary_exists(paper):
                generated_summary = await self._generate_wiki_paper_summary(
                    task,
                    paper,
                    evidence_records,
                )
                summary_content = wiki_store.paper_summary_markdown(paper, generated_summary)
                summary_artifact = self._write_text(
                    task,
                    paper.summary_relative_path,
                    summary_content,
                    kind="wiki",
                    description=f"Per-paper Wiki summary for {paper.paper_id}.",
                )
            else:
                summary_artifact = self._record_existing(
                    task,
                    paper.summary_relative_path,
                    kind="wiki",
                    description=f"Per-paper Wiki summary for {paper.paper_id}.",
                )
            artifacts.append(summary_artifact)

        index_artifact = self._write_text(
            task,
            "wiki/index.md",
            wiki_store.rebuild_index(),
            kind="wiki",
            description="Research Wiki paper index.",
        )
        artifacts.append(index_artifact)

        query = task.objective
        if task.command == "/idea" and stage.name == "idea_verification":
            query = f"{task.objective} 最近前例 反对证据 失败条件 实验范式"
        query_pack = wiki_store.query_pack(query)
        query_pack_artifact = self._write_text(
            task,
            "wiki/query_pack.md",
            query_pack,
            kind="wiki",
            description="Topic-focused Wiki retrieval pack.",
        )
        artifacts.append(query_pack_artifact)

        context_parts = ["Research Wiki retrieval context:\n" + query_pack[:9000]]
        has_wiki_papers = bool(wiki_store.list_papers())
        if not has_wiki_papers:
            local_context = resolve_local_research_context(workspace_root, task.objective)
            if local_context.context_text:
                context_parts.append("Local PDF evidence:\n" + local_context.context_text[:9000])
            if local_context.limitations:
                context_parts.append("Evidence limitations:\n- " + "\n- ".join(local_context.limitations))
        return artifacts, "\n\n".join(context_parts), has_wiki_papers

    async def _generate_wiki_paper_summary(
        self,
        task: TaskRun,
        paper: WikiPaper,
        evidence_records: list[dict],
    ) -> str:
        wiki_store = ResearchWikiStore(task.artifact_root)
        evidence_context = paper_evidence_prompt(evidence_records, limit=14000)
        if not evidence_context:
            return wiki_store.fallback_summary(paper, evidence_records)
        try:
            return await generate_text(
                system_prompt=(
                    "You create faithful Chinese research Wiki summaries from page-linked evidence. "
                    "Never invent facts, citations, datasets, results, limitations, or future work. "
                    "Clearly mark model inference and uncertainty. Return markdown without a level-one title."
                ),
                user_prompt=(
                    f"Paper ID: {paper.paper_id}\n"
                    f"Source PDF: {paper.source_relative_path}\n\n"
                    "Use exactly these sections: 研究问题, 核心方法, 数据集与实验设置, 主要结果, "
                    "作者讨论与局限, 作者提出的未来工作, 可复用证据, 与其他论文的关系, "
                    "对 Idea 生成的提示. Every factual point should cite an available Evidence ID and page. "
                    "If the evidence does not support a section, write 未确认.\n\n"
                    f"Evidence:\n{evidence_context}"
                ),
                model=config.upstream_model or None,
                temperature=0.15,
            )
        except Exception as exc:
            self._log_progress(
                task,
                f"论文 Wiki 总结调用失败，使用可追踪模板: {paper.paper_id} ({exc.__class__.__name__})",
                kind="warning",
            )
            return wiki_store.fallback_summary(paper, evidence_records)

    def _review_embed_fn(self):
        """Return an async embedder when a pluggable embedding provider is configured, else None."""

        if not config.embedding_enabled:
            return None

        async def _embed(texts: list[str]) -> list[list[float]]:
            return await embed_texts(texts, timeout=config.review_screen_timeout_seconds)

        return _embed

    async def _run_review_screening(
        self,
        task: TaskRun,
        papers: list[PaperRecord],
        *,
        topic: str,
        scope: str,
        round_label: str,
        max_candidates: int | None = None,
    ) -> dict:
        """Invoke the layered keep/drop screener (L1 hard-exclude -> L2 embedding rerank -> L3 LLM)."""

        model = config.review_screen_model or None
        summary = await screen_papers(
            papers,
            topic=topic,
            scope=scope,
            generate_text=generate_text,
            model=model,
            max_candidates=config.review_screen_max_candidates if max_candidates is None else max_candidates,
            batch_size=config.review_screen_batch_size,
            max_concurrency=config.review_screen_concurrency,
            hard_exclusion_terms=parse_exclusion_terms(config.review_hard_exclusion_terms),
            hard_relevance_floor=config.review_hard_relevance_floor,
            embed=self._review_embed_fn(),
            timeout=config.review_screen_timeout_seconds,
            on_event=lambda message: self._log_progress(task, f"[{round_label}筛选] {message}", kind="retrieval"),
        )
        if summary.get("screened"):
            self._log_progress(
                task,
                f"{round_label}分层精筛完成：core {summary.get('core', 0)} / transferable "
                f"{summary.get('transferable', 0)} / background {summary.get('background', 0)} / "
                f"drop {summary.get('dropped', 0)} / uncertain {summary.get('uncertain', 0)}"
                + (f"（rerank={summary.get('rerank_mode')}）" if summary.get("rerank_mode") else ""),
                kind="retrieval",
            )
        return summary

    async def _backfill_core_layer(
        self,
        task: TaskRun,
        bundle: LiteratureBundle,
        *,
        topic: str,
        scope: str,
        anchor_query: str,
        screen_summary: dict,
    ) -> dict:
        """Re-search the artifacts the scope names when screening admitted no core evidence.

        The scope reserves the core tier for public benchmarks, so ``core == 0`` means the
        retrieval never reached them. Directions clustered from the surviving transferable
        tier would then describe the *systems* being evaluated and silently swap the review
        question, so the named seeds are recovered before any direction card is built.
        """

        if int(screen_summary.get("core", 0) or 0) > 0:
            return {}
        seeds = extract_scope_artifact_names(scope)
        queries = core_backfill_queries(seeds, anchor_query)
        if not queries:
            self._log_progress(
                task,
                "[核心层告警] 分层筛选后 core=0，且范围卡中没有可定向补检的基准名称，"
                "方向卡将只能由 transferable 层聚出",
                kind="retrieval",
            )
            return {"seeds": [], "queries": [], "added_candidates": 0, "core_after": 0}
        self._log_progress(
            task,
            f"[核心层补检] core=0，按范围卡中的 {len(queries)} 个基准名定向补检（每个专名附领域锚定词）",
            kind="retrieval",
        )
        extra = await self.scholar.search_bundle(
            topic,
            queries=queries,
            per_source_limit=config.scholar_results_per_source,
            max_papers=max(24, config.review_discovery_max_papers),
            max_queries=len(queries),
            # Inherit *every* domain the first round resolved: the per-name queries must not be
            # re-classified as 'general', which would switch off the round's active indexes. Using
            # the full set (not the single coverage profile) keeps both cs and biomed sources on
            # for a multi-domain topic during the backfill.
            domain=inherited_domain_arg(bundle.quality.get("domains")) or None,
            search_round="discovery",
        )
        before = {id(paper) for paper in bundle.papers}
        merged = merge_paper_records([*bundle.papers, *extra.papers])
        for index, paper in enumerate(merged, start=1):
            paper.paper_id = f"P{index:03d}"
        fresh = [paper for paper in merged if not paper.screen_decision]
        added = sum(1 for paper in merged if id(paper) not in before)
        if fresh:
            await self._run_review_screening(
                task,
                fresh,
                topic=topic,
                scope=scope,
                round_label="核心层补检",
                max_candidates=config.review_screen_max_candidates,
            )
        bundle.papers[:] = merged
        for name, status in extra.provider_status.items():
            bundle.provider_status.setdefault(f"{name} (backfill)", status)
        core_after = sum(
            1 for paper in merged if paper.screen_decision == "keep" and paper.screen_tier == "core"
        )
        if core_after:
            self._log_progress(
                task, f"[核心层补检] 定向补检后核心层恢复 {core_after} 篇", kind="retrieval"
            )
        else:
            self._log_progress(
                task,
                "[核心层告警] 定向补检后核心层仍为空，方向卡可能把综述问题从『基准』换成『系统』，"
                "请在方向选择处人工干预",
                kind="retrieval",
            )
        return {
            "seeds": seeds,
            "queries": queries,
            "added_candidates": added,
            "core_after": core_after,
        }

    # --- Two-round /review helpers (clarify -> discovery -> direction selection -> deep dive) ---

    def _extract_markdown_section(self, path: Path, heading: str) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            rf"(?:^|\n)#{{1,6}}\s*{re.escape(heading)}\s*\n(.*?)(?=\n#{{1,6}}\s|\Z)",
            text,
            re.S,
        )
        return match.group(1).strip() if match else ""

    def _review_read_clarified_scope(self, workspace_root: Path) -> str:
        return self._extract_markdown_section(
            workspace_root / "bib" / "SCOPE_CLARIFICATION.md", "Clarified Scope"
        )

    def _review_read_focused_scope(self, workspace_root: Path) -> str:
        return self._extract_markdown_section(
            workspace_root / "bib" / "RESEARCH_DIRECTIONS.md", "Focused Retrieval Scope"
        )

    def _paper_from_dict(self, data: dict) -> PaperRecord:
        allowed = set(PaperRecord.__dataclass_fields__.keys())
        return PaperRecord(**{key: value for key, value in data.items() if key in allowed})

    def _review_load_discovery_bundle(self, workspace_root: Path) -> LiteratureBundle | None:
        path = workspace_root / "bib" / "LITERATURE_SEARCH.discovery.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        papers = [self._paper_from_dict(item) for item in payload.get("papers", []) if isinstance(item, dict)]
        return LiteratureBundle(
            query=str(payload.get("query", "")),
            queries=list(payload.get("queries", [])),
            papers=papers,
            provider_status=dict(payload.get("provider_status", {})),
            excluded_count=int(payload.get("excluded_count", 0) or 0),
            quality=dict(payload.get("quality", {}) or {}),
        )

    def _review_load_directions(self, workspace_root: Path) -> tuple[list, dict, bool]:
        default_meta = {"mode": "focused", "selected_ids": [], "custom_text": ""}
        path = workspace_root / "bib" / "RESEARCH_DIRECTIONS.json"
        if not path.exists():
            return [], default_meta, False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return [], default_meta, False
        directions = directions_from_json(payload.get("directions"))
        scope_meta = payload.get("scope_meta") or {
            "mode": "focused",
            "selected_ids": payload.get("recommended", []),
            "custom_text": "",
        }
        return directions, scope_meta, scope_meta.get("mode") == "panorama"

    def _track_review_side_artifacts(self, task: TaskRun, artifacts: list) -> None:
        tracked = {artifact.relative_path for artifact in artifacts}
        task.artifacts = [artifact for artifact in task.artifacts if artifact.relative_path not in tracked]
        task.artifacts.extend(artifacts)

    async def _run_review_clarify(self, task: TaskRun, stage: StageDefinition, feedback: str):
        workspace_root = Path(task.artifact_root)
        topic = clean_review_topic(task.objective)
        model = config.review_screen_model or None
        if not feedback.strip():
            self._archive_previous_review_outputs(workspace_root, task.task_id)
            if config.review_clarify_enabled:
                questions, scope_card = await build_clarify_draft(
                    topic, generate_text=generate_text, model=model
                )
            else:
                questions, scope_card = [], default_scope_card(topic)
            if config.review_clarify_enabled and config.enable_hitl:
                content = clarify_pass1_markdown(topic, questions, scope_card)
            else:
                content = (
                    f"# Scope Clarification\n\n- Topic: {topic}\n\n" + clarified_scope_markdown(scope_card)
                )
        else:
            prior_text = self._file_excerpt(str(workspace_root / "bib" / "SCOPE_CLARIFICATION.md"), 12000)
            content = await build_clarify_pass2(
                topic, feedback=feedback, prior_text=prior_text, generate_text=generate_text, model=model
            )
        return self._write_text(
            task,
            stage.artifact_path,
            content,
            kind=stage.artifact_kind,
            description="Review scope clarification.",
        )

    async def _run_direction_selection(self, task: TaskRun, stage: StageDefinition, feedback: str):
        topic = clean_review_topic(task.objective)
        if feedback.strip():
            return await self._direction_selection_pass2(task, stage, topic, feedback)
        workspace_root = Path(task.artifact_root)
        model = config.review_screen_model or None
        research_brief = self._file_excerpt(str(workspace_root / "bib" / "RESEARCH_BRIEF.md"), 12000)
        discovery_queries = parse_review_queries(task.objective, research_brief)
        # LLM-classified domains piggybacked on the brief (read from the full, often Chinese,
        # clarified scope). Passed as a hint; _resolve_domains unions it with the keyword scan,
        # so it can only add source families (e.g. keep Europe PMC/PMC on for a clinical topic
        # whose English queries otherwise read as pure cs), never remove one.
        brief_domains = parse_review_domains(research_brief or "")
        brief_domain_hint = " ".join(brief_domains)
        scope_text = self._review_read_clarified_scope(workspace_root) or (research_brief or topic).strip()
        # Always seed discovery with the benchmark names the clarified scope enumerates, so a
        # generically-worded brief still pulls the named artifacts into the pool. A keyword API
        # only returns a benchmark's own paper when the query names it; leaving this to the
        # brief's wording made recall of the seeds swing between runs (core 8 vs core 2). This
        # is the always-on counterpart to the core==0 emergency backfill.
        seed_names = extract_scope_artifact_names(scope_text, research_brief or "")
        name_queries = core_backfill_queries(
            seed_names, discovery_queries[0] if discovery_queries else topic
        )
        if name_queries:
            discovery_queries = list(dict.fromkeys([*discovery_queries, *name_queries]))
            self._log_progress(
                task,
                f"[发现轮] 已按范围卡中的 {len(name_queries)} 个基准名注入定向查询（每个专名附领域锚定词）",
                kind="retrieval",
            )
        # Re-entering direction_selection with empty feedback (e.g. a checkpoint rejected without
        # a real selection) used to re-run the whole discovery retrieval. Reuse the persisted
        # discovery bundle when the clarified scope and query plan are unchanged; a scope edit via
        # the clarify checkpoint changes the fingerprint and correctly forces a fresh round.
        scope_fingerprint = _discovery_fingerprint(scope_text, discovery_queries)
        cached_bundle = self._review_load_discovery_bundle(workspace_root)
        reuse_discovery = (
            cached_bundle is not None
            and bool(cached_bundle.papers)
            and str(cached_bundle.quality.get("scope_fingerprint", "")) == scope_fingerprint
        )
        if reuse_discovery:
            bundle = cached_bundle
            screened = bool(bundle.quality.get("screened", False))
            self._log_progress(
                task,
                "[发现轮] 范围与查询计划未变，复用上一轮检索与筛选结果，跳过重复检索",
                kind="retrieval",
            )
        else:
            self._log_progress(task, "正在执行第一轮探索性广泛检索", kind="retrieval")
            if brief_domains:
                self._log_progress(
                    task,
                    f"[发现轮] 研究简报判定领域 {', '.join(brief_domains)}，"
                    "已并入源激活（与关键词分类取并集，只增不减源）",
                    kind="retrieval",
                )
            bundle = await self.scholar.search_bundle(
                topic,
                queries=discovery_queries,
                per_source_limit=config.scholar_results_per_source,
                max_papers=max(24, config.review_discovery_max_papers),
                max_queries=max(config.review_query_limit, len(discovery_queries)),
                domain=brief_domain_hint or None,
                search_round="discovery",
            )
            screened = False
            screen_summary: dict = {}
            if config.review_screening_enabled:
                screen_summary = await self._run_review_screening(
                    task,
                    bundle.papers,
                    topic=topic,
                    scope=scope_text,
                    round_label="探索轮",
                    max_candidates=config.review_screen_max_candidates,
                )
                screened = bool(screen_summary.get("screened", False))
            core_backfill: dict = {}
            if screened:
                core_backfill = await self._backfill_core_layer(
                    task,
                    bundle,
                    topic=topic,
                    scope=scope_text,
                    anchor_query=(discovery_queries[0] if discovery_queries else topic),
                    screen_summary=screen_summary,
                )
                if core_backfill:
                    screen_summary = {
                        **screen_summary,
                        **_recount_screen_tiers(bundle.papers),
                    }
            relevant, traceable = screening_gate_counts(bundle.papers, screened=screened)
            bundle.quality.update(
                {
                    "relevant_count": relevant,
                    "traceable_count": traceable,
                    "screened": screened,
                    "search_round": "discovery",
                    "screen_error": screen_summary.get("screen_error", ""),
                    "screen_tiers": {
                        "core": screen_summary.get("core", 0),
                        "transferable": screen_summary.get("transferable", 0),
                        "background": screen_summary.get("background", 0),
                        "uncertain": screen_summary.get("uncertain", 0),
                        "unscreened": screen_summary.get("unscreened", 0),
                    },
                    "rerank_mode": screen_summary.get("rerank_mode", ""),
                }
            )
            if core_backfill:
                bundle.quality["core_backfill"] = core_backfill
            # The discovery round used to skip this entirely, so a provider that failed every
            # query was reported as 'Coverage state: unknown' and raised no degradation alarm.
            discovery_health = source_health(
                bundle.provider_status,
                domain=resolve_key_source_domain(bundle.quality.get("domains")),
            )
            bundle.quality["source_health"] = discovery_health
            bundle.quality["coverage_state"] = discovery_health["coverage_state"]
            bundle.quality["preprint_ratio"] = preprint_ratio(bundle)
            bundle.quality["scope_fingerprint"] = scope_fingerprint
            if discovery_health["coverage_state"] == "degraded":
                self._log_progress(
                    task,
                    "[覆盖告警] 第一轮关键索引未全部健康："
                    f"degraded={', '.join(discovery_health['degraded_sources']) or 'none'}; "
                    f"domain-gated-off={', '.join(discovery_health['domain_gated_sources']) or 'none'}",
                    kind="retrieval",
                )
        # Discover directions first so paper.direction_ids are annotated before the
        # discovery bundle is serialized (the deep-dive round reloads that provenance).
        directions: list = []
        recommended: list = []
        fallback_used = False
        if config.review_direction_selection_enabled:
            directions, recommended = await discover_review_directions(
                bundle.papers,
                topic=topic,
                scope=scope_text,
                generate_text=generate_text,
                model=model,
                max_candidates=config.review_direction_max_candidates,
                max_directions=config.review_direction_count,
                min_papers=config.review_direction_min_papers,
                timeout=config.review_screen_timeout_seconds,
                on_event=lambda message: self._log_progress(task, f"[方向发现] {message}", kind="retrieval"),
            )
        if not directions:
            directions, recommended = directions_from_facets(
                bundle.papers,
                max_directions=config.review_direction_count,
                min_papers=config.review_direction_min_papers,
            )
            fallback_used = True
        discovery_artifacts = [
            self._write_text(
                task,
                "bib/LITERATURE_SEARCH.discovery.md",
                bundle.to_markdown(),
                kind="note",
                description="First-round (discovery) scholarly search results.",
            ),
            self._write_text(
                task,
                "bib/LITERATURE_SEARCH.discovery.json",
                bundle.to_json(),
                kind="note",
                description="First-round (discovery) structured retrieval records.",
            ),
            self._write_text(
                task,
                "bib/RETRIEVAL_QUALITY.discovery.md",
                review_quality_markdown(bundle, local_sources=[], local_candidates=[]),
                kind="review",
                description="First-round discovery retrieval quality (soft discovery gate, non-blocking).",
            ),
        ]
        scope_meta = {"mode": "focused", "selected_ids": recommended, "custom_text": ""}
        persist = {
            "topic": topic,
            "directions": directions_to_json(directions),
            "recommended": recommended,
            "scope_meta": scope_meta,
            "fallback_used": fallback_used,
        }
        directions_json = self._write_text(
            task,
            "bib/RESEARCH_DIRECTIONS.json",
            json.dumps(persist, ensure_ascii=False, indent=2),
            kind="manifest",
            description="Discovered research directions and current selection state.",
        )
        if config.review_direction_selection_enabled and config.enable_hitl:
            content = build_directions_markdown(
                directions,
                recommended,
                topic=topic,
                fallback_used=fallback_used,
                core_count=int(bundle.quality.get("screen_tiers", {}).get("core", 0) or 0)
                if screened
                else None,
                coverage_state=str(bundle.quality.get("coverage_state", "")),
            )
        else:
            content = "# Research Directions\n\n- Topic: " + topic + "\n\n" + focused_scope_markdown(
                directions, topic=topic, selected_ids=recommended, mode="focused"
            )
        self._log_progress(
            task,
            f"方向发现完成：{len(directions)} 个方向（{'查询分面回退' if fallback_used else 'LLM 聚类'}）",
            kind="retrieval",
        )
        self._track_review_side_artifacts(task, [*discovery_artifacts, directions_json])
        return self._write_text(
            task,
            stage.artifact_path,
            content,
            kind=stage.artifact_kind,
            description="Research direction discovery and selection.",
        )

    async def _direction_selection_pass2(self, task: TaskRun, stage: StageDefinition, topic: str, feedback: str):
        workspace_root = Path(task.artifact_root)
        directions, _prev_meta, _panorama = self._review_load_directions(workspace_root)
        selection = interpret_selection(feedback, directions)
        scope_md = focused_scope_markdown(
            directions,
            topic=topic,
            selected_ids=selection["selected_ids"],
            mode=selection["mode"],
            custom_text=selection["custom_text"],
        )
        summary_line = ", ".join(selection["selected_ids"]) or selection["custom_text"] or "(recommended)"
        content = (
            f"# Research Directions\n\n- Topic: {topic}\n- Selection: {selection['mode']} {summary_line}\n\n"
            + scope_md
        )
        path = workspace_root / "bib" / "RESEARCH_DIRECTIONS.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (ValueError, OSError):
            payload = {}
        payload["scope_meta"] = selection
        directions_json = self._write_text(
            task,
            "bib/RESEARCH_DIRECTIONS.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            kind="manifest",
            description="Discovered research directions and current selection state.",
        )
        self._track_review_side_artifacts(task, [directions_json])
        return self._write_text(
            task,
            stage.artifact_path,
            content,
            kind=stage.artifact_kind,
            description="Research direction selection.",
        )

    async def _build_focused_bundle(self, task: TaskRun, topic: str, discovery_queries: list[str]) -> LiteratureBundle:
        workspace_root = Path(task.artifact_root)
        directions, scope_meta, panorama = self._review_load_directions(workspace_root)
        discovery_bundle = self._review_load_discovery_bundle(workspace_root)
        discovery_papers = discovery_bundle.papers if discovery_bundle else []
        scope_text = self._review_read_focused_scope(workspace_root) or (
            self._file_excerpt(str(workspace_root / "bib" / "RESEARCH_BRIEF.md"), 12000) or topic
        ).strip()
        model = config.review_screen_model or None
        mode = scope_meta.get("mode", "focused")
        if mode == "panorama" or panorama:
            self._log_progress(task, "全景模式：沿用第一轮保留文献，跳过第二轮定向检索", kind="retrieval")
            working = [p for p in discovery_papers if p.screen_decision != "drop"] or list(discovery_papers)
            provider_status = discovery_bundle.provider_status if discovery_bundle else {}
            queries_used = discovery_bundle.queries if discovery_bundle else discovery_queries
        else:
            selected_ids = set(scope_meta.get("selected_ids", []))
            selected_discovery = [
                p
                for p in discovery_papers
                if p.screen_decision != "drop" and (not selected_ids or selected_ids.intersection(p.direction_ids))
            ]
            if not selected_discovery:
                selected_discovery = [p for p in discovery_papers if p.screen_decision != "drop"]
            focused_queries = focused_query_plan(
                scope_meta,
                directions,
                topic=topic,
                limit=config.review_focused_query_limit,
                papers=discovery_papers,
                discovery_queries=discovery_queries,
            ) or discovery_queries
            self._log_progress(task, "正在执行第二轮定向深度检索", kind="retrieval")
            # Inherit the domains the discovery round resolved so the focused round keeps the same
            # source families active. Without a hint the focused round classified only its
            # (method-heavy, English) focused_queries and could silently drop the clinical indexes
            # a biomedical topic needs, or the arXiv/DBLP a cs topic needs.
            focused_domain_hint = inherited_domain_arg(
                discovery_bundle.quality.get("domains") if discovery_bundle else None
            )
            round2 = await self.scholar.search_bundle(
                topic,
                queries=focused_queries,
                per_source_limit=config.scholar_results_per_source,
                max_papers=config.review_focused_max_papers,
                max_queries=config.review_focused_query_limit,
                domain=focused_domain_hint or None,
                search_round="focused",
            )
            working = merge_paper_records([*selected_discovery, *round2.papers])
            working.sort(key=lambda item: item.relevance_score, reverse=True)
            working = working[: config.review_focused_max_papers]
            provider_status = round2.provider_status
            queries_used = focused_queries
        for index, paper in enumerate(working, start=1):
            paper.paper_id = f"P{index:03d}"
        screened = False
        screen_summary: dict = {}
        if config.review_screening_enabled and working:
            screen_summary = await self._run_review_screening(
                task,
                working,
                topic=topic,
                scope=scope_text,
                round_label="深挖轮",
                max_candidates=config.review_focused_screen_max_candidates,
            )
            screened = bool(screen_summary.get("screened", False))
        # Order kept papers by screening tier so the capped PDF download prioritizes core work.
        tier_rank = {"core": 0, "transferable": 1, "background": 2, "unscreened": 4, "": 5, "exclude": 9}
        working.sort(
            key=lambda paper: (
                0 if paper.screen_decision != "drop" else 1,
                tier_rank.get(paper.screen_tier, 5),
                -paper.relevance_score,
            )
        )
        for index, paper in enumerate(working, start=1):
            paper.paper_id = f"P{index:03d}"
        relevant, traceable = screening_gate_counts(working, screened=screened)
        dropped = sum(1 for paper in working if paper.screen_decision == "drop")
        quality = {
            "status": "adequate"
            if relevant >= config.review_recommended_sources
            else "marginal"
            if relevant >= config.review_minimum_sources
            else "insufficient",
            "mode": "panorama" if (mode == "panorama" or panorama) else "focused",
            "selected_directions": sorted(scope_meta.get("selected_ids", [])),
            "candidate_count": len(working),
            "eligible_count": relevant,
            "relevant_count": relevant,
            "traceable_count": traceable,
            "truncated_count": 0,
            "screened": screened,
            "screen_error": screen_summary.get("screen_error", ""),
            "screen_tiers": {
                "core": screen_summary.get("core", 0),
                "transferable": screen_summary.get("transferable", 0),
                "background": screen_summary.get("background", 0),
                "uncertain": screen_summary.get("uncertain", 0),
                "unscreened": screen_summary.get("unscreened", 0),
            },
            "rerank_mode": screen_summary.get("rerank_mode", ""),
            "minimum_for_synthesis": config.review_minimum_sources,
            "recommended_for_review": config.review_recommended_sources,
            "provider_success_count": sum(
                str(status).startswith("ok") for status in provider_status.values()
            ),
        }
        # Judge the deep-dive round against the domain the *discovery* round resolved.
        # Without it the focused round is scored against the backbone list only, so arXiv
        # and DBLP being switched off by proper-noun queries never showed up as degraded.
        health = source_health(
            provider_status,
            domain=resolve_key_source_domain(
                (discovery_bundle.quality.get("domains") if discovery_bundle else None)
            ),
        )
        quality["source_health"] = health
        quality["coverage_state"] = health["coverage_state"]
        bundle = LiteratureBundle(
            query=topic,
            queries=queries_used,
            papers=working,
            provider_status=provider_status,
            excluded_count=dropped,
            quality=quality,
        )
        quality["preprint_ratio"] = preprint_ratio(bundle)
        return bundle

    async def _prepare_review_support(self, task: TaskRun, stage: StageDefinition) -> tuple[list, str]:
        # clarify and direction_selection are produced deterministically in _execute_stage
        # (scope Q&A / first-round retrieval + direction discovery); no LLM prompt support needed.
        if stage.name in {"clarify", "direction_selection"}:
            return [], ""
        workspace_root = Path(task.artifact_root)
        local_refs = self._review_local_source_refs(workspace_root)
        local_records = collect_paper_evidence(workspace_root, local_refs, total_limit=18000)
        topic = clean_review_topic(task.objective)
        local_context = self._review_local_context(
            self._filter_review_local_records(local_records, [topic])
        )
        if stage.name == "research_brief":
            clarified_scope = self._review_read_clarified_scope(workspace_root)
            context = (
                "Review retrieval protocol: define 4-6 query variants as lines formatted exactly `Q1: ...`, `Q2: ...`. "
                "Use space-separated keyword strings (not boolean syntax). Include the cleaned core topic, canonical "
                "English terms, Chinese variants, domain aliases, one recent-review query, and one foundational query. "
                "Do not claim that external retrieval has already succeeded. Do not claim citation snowballing.\n\n"
                "Source Coverage must list ONLY these actually wired providers (omit any not named here): "
                "OpenAlex, Crossref, Semantic Scholar; domain-gated extras arXiv and DBLP (cs) or Europe PMC and PMC "
                "(biomed). Optional sources CORE/OpenAIRE/BASE/Zenodo/HAL/WoS/CNKI are disabled unless configured. "
                "Do not mention Papers with Code, Hugging Face Datasets, Wanfang, ChinaXiv, Google Patents, or DeepXiv.\n\n"
                "Also add exactly one line formatted `Domains: cs, biomed` that lists every research domain whose "
                "literature must be searched, chosen ONLY from this closed set: cs, biomed, physics_math. Judge from "
                "the scope's intent, not just method words: a topic that applies machine learning / deep learning to a "
                "clinical, medical, physiological-signal, patient, or health subject is BOTH cs and biomed, so it must "
                "list both. Use physics_math only for physics or mathematics subjects. Emit this line verbatim once.\n\n"
            )
            if clarified_scope:
                context += "Confirmed scope from clarification:\n" + clarified_scope + "\n\n"
            if local_context:
                context += local_context
            return [], context

        search_markdown = workspace_root / "bib" / "LITERATURE_SEARCH.md"
        search_json = workspace_root / "bib" / "LITERATURE_SEARCH.json"
        quality_path = workspace_root / "bib" / "RETRIEVAL_QUALITY.md"
        research_brief = self._file_excerpt(
            str(workspace_root / "bib" / "RESEARCH_BRIEF.md"),
            12000,
        )
        queries = parse_review_queries(task.objective, research_brief)
        local_records = self._filter_review_local_records(local_records, queries)
        local_evidence_refs = sorted({str(record["source_path"]) for record in local_records})
        local_context = self._review_local_context(local_records)
        if stage.name == "literature_synthesis":
            self._log_progress(task, "正在按已确认方向执行第二轮定向深挖检索", kind="retrieval")
            bundle = await self._build_focused_bundle(task, topic, queries)
            self._log_progress(
                task,
                f"文献检索完成：保留 {len(bundle.papers)} 篇，排除 {bundle.excluded_count} 个弱相关候选",
                kind="retrieval",
            )
            bundle.quality["local_evidence_sources"] = local_evidence_refs
            download_artifacts = []

            def write_download(relative_path: str, content: bytes):
                artifact = self._write_bytes(
                    task,
                    relative_path,
                    content,
                    kind="document",
                    description="Publicly available literature PDF downloaded from the source URL.",
                )
                download_artifacts.append(artifact)
                return artifact

            self._log_progress(task, "正在检查检索结果中的公开 PDF 地址", kind="retrieval")
            download_manifest = await download_public_pdfs(
                bundle,
                workspace_root,
                enabled=config.review_download_enabled,
                limit=config.review_download_limit,
                max_mb=config.review_download_max_mb,
                timeout_seconds=config.review_download_timeout_seconds,
                write_file=write_download,
            )
            download_counts = download_manifest.get("counts", {})
            self._log_progress(
                task,
                "公开全文检查完成：下载 "
                f"{download_counts.get('downloaded', 0)} 篇，无公开 PDF "
                f"{download_counts.get('no_public_pdf', 0)} 篇，失败 "
                f"{download_counts.get('failed', 0) + download_counts.get('invalid_pdf', 0)} 篇",
                kind="retrieval",
            )
            institutional_handoff = build_institutional_handoff(
                bundle,
                workspace_root,
                enabled=config.institution_access_enabled,
                institution_name=config.institution_name,
                gateway_base=config.institution_gateway_base,
                eproxy_base=config.institution_eproxy_base,
                openurl_base=config.institution_openurl_base,
                proxy_prefix=config.institution_proxy_prefix,
                mode=config.institution_download_mode,
            )
            institutional_count = len(institutional_handoff.get("items", []))
            if institutional_count:
                self._log_progress(
                    task,
                    f"机构授权接力清单已生成：{institutional_count} 篇需要用户使用自己的机构账号访问后上传",
                    kind="retrieval",
                )
            generated = [
                self._write_text(
                    task,
                    "bib/LITERATURE_SEARCH.md",
                    bundle.to_markdown(),
                    kind="note",
                    description="Expanded, relevance-filtered scholarly search results.",
                ),
                self._write_text(
                    task,
                    "bib/LITERATURE_SEARCH.json",
                    bundle.to_json(),
                    kind="note",
                    description="Structured review retrieval records with stable paper IDs.",
                ),
                self._write_text(
                    task,
                    "bib/LITERATURE_DOWNLOADS.json",
                    json.dumps(download_manifest, ensure_ascii=False, indent=2),
                    kind="manifest",
                    description="Public literature PDF download manifest with per-paper status.",
                ),
                self._write_text(
                    task,
                    "bib/INSTITUTIONAL_ACCESS.json",
                    json.dumps(institutional_handoff, ensure_ascii=False, indent=2),
                    kind="manifest",
                    description="Institutional-login handoff links for papers without an automatically downloaded public PDF.",
                ),
                self._write_text(
                    task,
                    "bib/INSTITUTIONAL_ACCESS.md",
                    institutional_handoff_markdown(institutional_handoff),
                    kind="note",
                    description="User instructions for authorized institutional PDF retrieval and upload.",
                ),
                self._write_text(
                    task,
                    "bib/RETRIEVAL_QUALITY.md",
                    review_quality_markdown(
                        bundle,
                        local_sources=local_evidence_refs,
                        local_candidates=local_refs,
                    ),
                    kind="review",
                    description="Deterministic retrieval coverage and evidence gate.",
                ),
            ]
            generated.extend(download_artifacts)
            generated_paths = {artifact.relative_path for artifact in generated}
            task.artifacts = [
                artifact for artifact in task.artifacts if artifact.relative_path not in generated_paths
            ]
            if not review_evidence_is_sufficient(bundle, local_source_count=len(local_evidence_refs)):
                task.artifacts.extend(generated)
                self._log_progress(task, "文献证据门控失败：相关且可追溯的来源不足，已停止正式综述。")
                self.store.save_task(task)
                await self.sync_task_workspace(task)
                raise ReviewEvidenceError(
                    "检索完成，但相关且可追溯的文献不足，已停止生成正式综述、证据图和研究空白，"
                    "避免用无关结果拼接内容。请查看 `bib/RETRIEVAL_QUALITY.md`，上传相关论文或恢复更多文献源后重试。"
                )
            context = (
                "Use only the admitted evidence below. Cite every paper-level factual claim with stable IDs such as [P001]. "
                "Do not cite excluded candidates or invent bibliographic fields. Distinguish metadata/abstract evidence from "
                "full-text evidence.\n\n"
                + bundle.prompt_excerpt(limit=18000)
            )
            if local_context:
                context += "\n\n" + local_context
            return generated, context

        context_parts = []
        if search_markdown.exists():
            context_parts.append(
                "Stable paper IDs from the admitted retrieval set:\n"
                + self._file_excerpt(str(search_markdown), 18000)
            )
        if quality_path.exists():
            context_parts.append(self._file_excerpt(str(quality_path), 6000))
        if local_context:
            context_parts.append(local_context)
        return [], "\n\n".join(context_parts)

    def _review_local_source_refs(self, workspace_root: Path) -> list[str]:
        allowed = {".bib", ".docx", ".enw", ".md", ".nbib", ".pdf", ".ris", ".tex", ".txt"}
        candidates = [
            path
            for path in workspace_root.glob("*/uploads/*")
            if path.is_file() and path.suffix.lower() in allowed
        ]
        candidates.extend(
            path
            for path in (workspace_root / "bib" / "papers").glob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return [path.relative_to(workspace_root).as_posix() for path in candidates[:20]]

    def _review_local_context(self, records: list[dict]) -> str:
        if not records:
            return ""
        return (
            "Local user-provided literature has priority over external metadata. Cite it by relative source path and page "
            "when available.\n\n"
            + paper_evidence_prompt(records, limit=18000)
        )

    def _filter_review_local_records(self, records: list[dict], queries: list[str]) -> list[dict]:
        stopwords = {
            "article", "current", "foundational", "literature", "recent", "review", "systematic", "survey",
        }
        terms: list[str] = []
        for query in queries:
            terms.extend(
                term
                for term in re.findall(r"[a-z][a-z0-9-]{2,}", query.casefold())
                if term not in stopwords
            )
            terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", query))
        terms = list(dict.fromkeys(terms))
        if not terms:
            return records
        admitted_sources = {
            str(record["source_path"])
            for record in records
            if any(
                term in f"{record['source_path']} {record.get('excerpt', '')}".casefold()
                for term in terms
            )
        }
        return [record for record in records if str(record["source_path"]) in admitted_sources]

    def _archive_previous_review_outputs(self, workspace_root: Path, task_id: str) -> None:
        review_outputs = (
            "SCOPE_CLARIFICATION.md",
            "RESEARCH_BRIEF.md",
            "RESEARCH_DIRECTIONS.md",
            "RESEARCH_DIRECTIONS.json",
            "LITERATURE_SEARCH.discovery.md",
            "LITERATURE_SEARCH.discovery.json",
            "RETRIEVAL_QUALITY.discovery.md",
            "LITERATURE_SEARCH.md",
            "LITERATURE_SEARCH.json",
            "LITERATURE_DOWNLOADS.json",
            "INSTITUTIONAL_ACCESS.json",
            "INSTITUTIONAL_ACCESS.md",
            "RETRIEVAL_QUALITY.md",
            "LITERATURE_REVIEW.md",
            "EVIDENCE_MAP.md",
            "RESEARCH_GAPS.md",
            "CITATION_AUDIT.json",
            "REVIEW_COVERAGE.json",
        )
        existing = [workspace_root / "bib" / name for name in review_outputs]
        existing = [path for path in existing if path.exists()]
        if not existing:
            return
        archive_root = workspace_root / "bib" / "archive" / f"prior-to-{task_id}"
        archive_root.mkdir(parents=True, exist_ok=True)
        for path in existing:
            path.replace(archive_root / path.name)

    def _prepare_presentation_support(self, task: TaskRun) -> tuple[list, str]:
        workspace_root = Path(task.artifact_root)
        source_config = task.presentation_source or resolve_presentation_source_config(
            task.objective,
            workspace_root,
            source_limit=config.presentation_source_limit,
        )
        if task.presentation_source is None:
            task.presentation_source = source_config
            self.store.save_task(task)
        mode = source_config.presentation_type
        template = select_presentation_template(task.objective, mode, config.presentation_template)
        evidence = collect_workspace_evidence(
            workspace_root,
            mode,
            source_refs=source_config.source_refs,
        )
        selection_path = workspace_root / "Content" / "PRESENTATION_SOURCE_SELECTION.json"
        previous_selection: dict = {}
        if selection_path.exists():
            try:
                previous_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous_selection = {}
        assets_path = workspace_root / "Content" / "PRESENTATION_ASSETS.json"
        if assets_path.exists() and previous_selection.get("task_id") == task.task_id:
            assets = presentation_assets_from_json(assets_path.read_text(encoding="utf-8"))
        else:
            assets = extract_presentation_assets(
                workspace_root,
                source_config.source_refs,
                f"figures/extracted/presentation/{task.task_id}",
            )
        source_index_path = workspace_root / "Content" / "PRESENTATION_SOURCE_INDEX.md"
        source_index = (
            "# Presentation Source Index\n\n"
            f"- Mode: `{mode}`\n"
            f"- Template: `{template.name}`\n"
            f"- Requested scope: `{source_config.requested_scope}`\n"
            f"- Resolved scope: `{source_config.resolved_scope}`\n"
            f"- Upload batch: `{source_config.upload_batch_id or 'none'}`\n"
            f"- Selection reason: {source_config.selection_reason}\n"
            f"- Files selected: {len(evidence.files)}\n"
            f"- Original assets extracted: {len(assets)}\n\n"
            "## Selected Source Files\n"
            + ("\n".join(f"- `{path}`" for path in evidence.files) or "- None")
            + "\n"
        )
        source_selection = {
            "task_id": task.task_id,
            **source_config.model_dump(),
            "selected_files": list(evidence.files),
            "asset_count": len(assets),
        }
        artifacts: list = []
        if not source_index_path.exists() or source_index_path.read_text(encoding="utf-8", errors="ignore") != source_index:
            artifacts.append(
                self._write_text(
                    task,
                    "Content/PRESENTATION_SOURCE_INDEX.md",
                    source_index,
                    kind="note",
                    description="Workspace evidence index used by the presentation workflow.",
                )
            )
        selection_json = json.dumps(source_selection, ensure_ascii=False, indent=2)
        if not selection_path.exists() or selection_path.read_text(encoding="utf-8", errors="ignore") != selection_json:
            artifacts.append(
                self._write_text(
                    task,
                    "Content/PRESENTATION_SOURCE_SELECTION.json",
                    selection_json,
                    kind="note",
                    description="Frozen source boundary for the presentation workflow.",
                )
            )
        assets_json = presentation_assets_to_json(assets)
        if not assets_path.exists() or assets_path.read_text(encoding="utf-8", errors="ignore") != assets_json:
            artifacts.append(
                self._write_text(
                    task,
                    "Content/PRESENTATION_ASSETS.json",
                    assets_json,
                    kind="note",
                    description="Original image and table assets extracted from the frozen SourceSet.",
                )
            )
        asset_context = "\n".join(
            f"- {asset.asset_id} | {asset.kind} | {asset.source_path}"
            + (f" | page {asset.page}" if asset.page else "")
            + (f" | {asset.caption}" if asset.caption else "")
            for asset in assets[:80]
        )
        context = (
            "Presentation requirements:\n"
            f"- Detected mode: {mode}\n"
            f"- Selected template: {template.name}\n"
            f"- Frozen source scope: {source_config.resolved_scope}\n"
            f"- Selection reason: {source_config.selection_reason}\n"
            "- Use only the evidence below for claims and numbers.\n"
            "- Prefer matching original image/table asset_ids over regenerated data visuals.\n"
            "- Stage mode prioritizes plan, progress, results, risks, decisions, and next steps.\n"
            "- Paper mode treats final artifacts in paper/ as authoritative and follows a complete paper-talk arc.\n\n"
            f"Available original assets:\n{asset_context or '- None'}\n\n"
            f"Selected evidence:\n{evidence.text or 'No readable selected evidence found; rely only on the user objective.'}"
        )
        return artifacts, context

    def _prepare_rebuttal_support(self, task: TaskRun) -> tuple[list, str]:
        workspace_root = Path(task.artifact_root)
        source_config = task.rebuttal_source
        if source_config is None:
            session = self.store.load_session(task.session_id)
            source_config = resolve_rebuttal_source_config(
                task.objective,
                workspace_root,
                session.upload_batches if session else (),
            )
            task.rebuttal_source = source_config
            self.store.save_task(task)

        paper_evidence = collect_workspace_evidence(
            workspace_root,
            "paper",
            limit=18000,
            source_refs=source_config.paper_refs,
        )
        review_evidence = collect_workspace_evidence(
            workspace_root,
            "stage",
            limit=18000,
            source_refs=source_config.review_refs,
        )
        selection = {
            "task_id": task.task_id,
            **source_config.model_dump(),
            "paper_files_read": list(paper_evidence.files),
            "review_files_read": list(review_evidence.files),
        }
        selection_json = json.dumps(selection, ensure_ascii=False, indent=2)
        selection_path = workspace_root / "Content" / "REBUTTAL_SOURCE_SELECTION.json"
        artifacts: list = []
        if (
            not selection_path.exists()
            or selection_path.read_text(encoding="utf-8", errors="ignore") != selection_json
        ):
            artifacts.append(
                self._write_text(
                    task,
                    "Content/REBUTTAL_SOURCE_SELECTION.json",
                    selection_json,
                    kind="note",
                    description="Frozen completed-paper and reviewer-comment sources for /rebuttal.",
                )
            )

        context = (
            "Rebuttal SourceSet contract:\n"
            "- The completed paper and reviewer comments below are both required and have been frozen for this task.\n"
            "- Analyze reviewer comments against the actual paper. Do not answer from the task objective alone.\n"
            "- Preserve reviewer-by-reviewer and comment-by-comment traceability.\n"
            "- Every proposed response must point to a paper section, claim, figure, table, evidence item, or an explicit gap.\n"
            "- Distinguish current paper text, proposed response language, promised revision, and new experiment needs.\n\n"
            "Completed paper sources:\n"
            + "\n".join(f"- `{path}`" for path in source_config.paper_refs)
            + "\n\nCompleted paper content:\n"
            + (paper_evidence.text or "No readable paper text could be extracted from the frozen files.")
            + "\n\nReviewer-comment sources:\n"
            + "\n".join(f"- `{path}`" for path in source_config.review_refs)
            + "\n\nReviewer comments:\n"
            + (review_evidence.text or "No readable reviewer text could be extracted from the frozen files.")
        )
        workflow_context = []
        for relative in (
            "rebuttal/REVIEW_TO_PAPER_MAP.md",
            "rebuttal/RESPONSE_STRATEGY.md",
            "rebuttal/REBUTTAL_DRAFT.md",
            "rebuttal/REVISION_PLAN.md",
            "paper/PAPER_REVISED_AFTER_REVIEW.md",
        ):
            excerpt = self._artifact_excerpt(task, relative)
            if excerpt:
                workflow_context.append(f"### {relative}\n{excerpt}")
        if workflow_context:
            context += "\n\nExisting rebuttal workflow artifacts:\n" + "\n\n".join(workflow_context)
        return artifacts, context

    async def _prepare_download_support(self, task: TaskRun, stage: StageDefinition) -> tuple[list, str]:
        workspace_root = Path(task.artifact_root)
        source_config = task.download_source
        if source_config is None:
            session = self.store.load_session(task.session_id)
            source_config = resolve_download_source_config(
                task.objective,
                workspace_root,
                session.upload_batches if session else (),
                source_limit=config.review_download_limit,
            )
            task.download_source = source_config
            self.store.save_task(task)

        source_records = discover_download_sources(workspace_root, source_config.source_refs)
        selection = {
            "task_id": task.task_id,
            **source_config.model_dump(),
            "source_files_read": [record.source_path for record in source_records],
        }
        selection_json = json.dumps(selection, ensure_ascii=False, indent=2)
        selection_path = workspace_root / "Content" / "DOWNLOAD_SOURCE_SELECTION.json"
        artifacts: list = []
        if (
            not selection_path.exists()
            or selection_path.read_text(encoding="utf-8", errors="ignore") != selection_json
        ):
            artifacts.append(
                self._write_text(
                    task,
                    "Content/DOWNLOAD_SOURCE_SELECTION.json",
                    selection_json,
                    kind="note",
                    description="Frozen source boundary for the literature download workflow.",
                )
            )

        source_context = download_targets_markdown(source_records)
        query_terms = source_config.query_terms
        if stage.name == "download_plan":
            context = (
                "Download retrieval protocol: freeze the source set, identify paper titles, DOI targets, and query terms. "
                "Do not claim any file has been downloaded yet.\n\n"
                + source_context
            )
            return artifacts, context

        if stage.name == "download_search":
            query_plan = "\n".join(f"- {term}" for term in query_terms) if query_terms else "- None"
            context = (
                "Download search stage: locate public PDFs and resolve candidate URLs for the source set below.\n\n"
                f"## Query Plan\n{query_plan}\n\n"
                f"{source_context}"
            )
            return artifacts, context

        query = query_terms[0] if query_terms else task.objective
        bundle = await self.scholar.search_bundle(
            query,
            queries=query_terms or None,
            per_source_limit=config.scholar_results_per_source,
            max_papers=24,
        )
        download_artifacts: list = []

        def write_download(relative_path: str, content: bytes):
            artifact = self._write_bytes(
                task,
                relative_path,
                content,
                kind="document",
                description="Publicly available literature PDF downloaded from the resolved source URLs.",
            )
            download_artifacts.append(artifact)
            return artifact

        self._log_progress(task, "正在根据下载源搜索可公开获取的 PDF", kind="retrieval")
        download_manifest = await download_public_pdfs_from_sources(
            bundle,
            workspace_root,
            source_config.source_refs,
            enabled=config.review_download_enabled,
            limit=config.review_download_limit,
            max_mb=config.review_download_max_mb,
            timeout_seconds=config.review_download_timeout_seconds,
            write_file=write_download,
        )
        counts = download_manifest.get("counts", {})
        context = (
            "Download retrieval completed.\n\n"
            f"## Download Sources\n{source_context}\n\n"
            f"## Query Terms\n" + ("\n".join(f"- {term}" for term in query_terms) if query_terms else "- None")
        )
        generated = [
            self._write_text(
                task,
                "bib/DOWNLOAD_MANIFEST.json",
                json.dumps(download_manifest, ensure_ascii=False, indent=2),
                kind="manifest",
                description="Public literature PDF download manifest with per-paper status.",
            ),
            self._write_text(
                task,
                "bib/DOWNLOAD_MANIFEST.md",
                f"# Download Manifest\n\n- Downloaded: {counts.get('downloaded', 0)}\n- Failed: {counts.get('failed', 0)}\n- Invalid PDF: {counts.get('invalid_pdf', 0)}\n- No public PDF: {counts.get('no_public_pdf', 0)}\n",
                kind="note",
                description="Human-readable download manifest summary.",
            ),
        ]
        return [*artifacts, *download_artifacts, *generated], context
    def _prepare_write_support(self, task: TaskRun) -> tuple[list, str]:
        workspace_root = Path(task.artifact_root)
        source_config = task.write_source
        if source_config is None:
            session = self.store.load_session(task.session_id)
            source_config = resolve_write_source_config(
                task.objective,
                workspace_root,
                session.upload_batches if session else (),
                source_limit=config.write_source_limit,
            )
            task.write_source = source_config
            self.store.save_task(task)

        selection = {
            "task_id": task.task_id,
            **source_config.model_dump(),
            "source_boundary": "frozen_at_task_start",
        }
        selection_json = json.dumps(selection, ensure_ascii=False, indent=2)
        selection_path = workspace_root / "Content" / "PAPER_SOURCE_SELECTION.json"
        artifacts: list = []
        if (
            not selection_path.exists()
            or selection_path.read_text(encoding="utf-8", errors="ignore") != selection_json
        ):
            artifacts.append(
                self._write_text(
                    task,
                    "Content/PAPER_SOURCE_SELECTION.json",
                    selection_json,
                    kind="note",
                    description="Frozen source boundary for the paper-writing workflow.",
                )
            )

        records = self._paper_evidence_records(task)
        venue_profile = select_venue_profile(task.objective)
        profile_json = json.dumps(venue_profile, ensure_ascii=False, indent=2)
        profile_path = workspace_root / "Content" / "PAPER_VENUE_PROFILE.json"
        if (
            not profile_path.exists()
            or profile_path.read_text(encoding="utf-8", errors="ignore") != profile_json
        ):
            artifacts.append(
                self._write_text(
                    task,
                    "Content/PAPER_VENUE_PROFILE.json",
                    profile_json,
                    kind="note",
                    description="Paper type, target venue, and content delivery requirements.",
                )
            )
        context = (
            "Paper writing SourceSet contract:\n"
            f"- Frozen scope: {source_config.resolved_scope}\n"
            f"- Selection reason: {source_config.selection_reason}\n"
            f"- Writing profile: {venue_profile['profile']}\n"
            f"- Target venue: {venue_profile['venue']}\n"
            "- Use only the frozen evidence records below for factual claims, numbers, figures, tables, and citations.\n"
            "- Cite evidence IDs such as PE-XXXXXXXXXX in planning, self-review, and revision traceability.\n"
            "- Preserve uncertainty and write [AUTHOR INPUT NEEDED] when evidence is missing.\n"
            "- Do not claim that a venue template is validated unless a real publisher template was supplied and compiled.\n\n"
            "Frozen source files:\n"
            + ("\n".join(f"- `{path}`" for path in source_config.source_refs) or "- None")
            + "\n\nEvidence records:\n"
            + (paper_evidence_prompt(records) or "No readable evidence was extracted; keep all unsupported sections explicit.")
        )
        return artifacts, context

    def _paper_evidence_records(self, task: TaskRun) -> list[dict]:
        workspace_root = Path(task.artifact_root)
        source_config = task.write_source
        if source_config is None:
            return []
        evidence_path = workspace_root / "paper" / "PAPER_EVIDENCE_MAP.json"
        metadata_path = workspace_root / "Content" / "PAPER_EVIDENCE_METADATA.json"
        metadata: dict = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        cache_matches = (
            metadata.get("task_id") == task.task_id
            and metadata.get("source_refs") == list(source_config.source_refs)
        )
        if cache_matches and evidence_path.exists():
            try:
                records = json.loads(evidence_path.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    return records
            except json.JSONDecodeError:
                pass
        return collect_paper_evidence(workspace_root, source_config.source_refs)

    async def _write_research_contract(self, task: TaskRun) -> None:
        template_path = Path(config.aris_repo_root) / "templates" / "RESEARCH_CONTRACT_TEMPLATE.md"
        template = template_path.read_text(encoding="utf-8") if template_path.exists() else "# Research Contract"
        candidates = self._artifact_excerpt(task, "idea/IDEA_CANDIDATES.md")
        verification = self._artifact_excerpt(task, "idea/IDEA_VERIFICATION.md")
        final_idea = self._artifact_excerpt(task, "idea/FINAL_IDEA.md")
        session_context = self._session_context_for_task(task)
        user_prompt = "\n".join(
            part
            for part in [
                "Using the template and the idea artifacts below, fill a focused research contract for handoff to /plan.",
                "Do not invent an experiment plan; preserve unresolved evidence and planning needs.",
                "Keep the contract within about 3,000 Chinese characters. Return one markdown document only, without code fences or repeated drafts.",
                f"Session context:\n{session_context}" if session_context else "",
                f"Template:\n{template}",
                f"IDEA_CANDIDATES excerpt:\n{candidates}",
                f"IDEA_VERIFICATION excerpt:\n{verification}",
                f"FINAL_IDEA excerpt:\n{final_idea}",
            ]
            if part
        )
        content = await generate_text(
            system_prompt="You create focused research contracts for continuation and session recovery.",
            user_prompt=user_prompt,
            temperature=0.2,
        )
        content = self._normalize_generated_markdown(content, [])
        artifact = self._write_text(
            task,
            "idea/docs/research_contract.md",
            content,
            kind="contract",
            description="Focused research contract for the selected idea.",
        )
        self._upsert_task_artifact(task, artifact)
        for wiki_artifact in self._write_idea_to_wiki(task):
            self._upsert_task_artifact(task, wiki_artifact)
        self.store.save_task(task)

    def _write_idea_to_wiki(self, task: TaskRun) -> list:
        final_idea = self._artifact_text(task, "idea/FINAL_IDEA.md")
        if not final_idea.strip():
            return []
        verification = self._artifact_text(task, "idea/IDEA_VERIFICATION.md")
        query_pack = self._artifact_text(task, "wiki/query_pack.md")
        wiki_store = ResearchWikiStore(task.artifact_root)
        paper_ids = wiki_store.paper_ids_from_query_pack(query_pack)
        idea_relative_path = wiki_store.write_idea_page(
            task.task_id,
            final_idea=final_idea,
            verification=verification,
            source_paper_ids=paper_ids,
        )
        relations_relative_path = wiki_store.append_idea_relations(task.task_id, paper_ids)
        return [
            self._record_existing(
                task,
                idea_relative_path,
                kind="wiki",
                description="Final Idea writeback into the Research Wiki.",
            ),
            self._record_existing(
                task,
                relations_relative_path,
                kind="wiki",
                description="Minimal Research Wiki relationship edges.",
            ),
        ]

    async def _write_figure_delivery_artifacts(self, task: TaskRun) -> list:
        briefs = self._artifact_text(task, "figures/FIGURE_BRIEFS.md")
        inventory = self._artifact_excerpt(task, "figures/FIGURE_INVENTORY.md")
        if not briefs and not inventory:
            return []

        session = self.store.load_session(task.session_id)
        data_sources = select_data_sources(
            Path(task.artifact_root),
            session.upload_batches if session else (),
        )
        try:
            rendered = render_code_figure(
                Path(task.artifact_root),
                session.upload_batches if session else (),
                objective=task.objective,
            )
        except NoRenderableDataError as exc:
            rendered = None
            data_warnings = exc.warnings
        else:
            image_artifact = self._write_bytes(
                task,
                "figures/generated/FIGURE_01.png",
                rendered.png_bytes,
                kind="image",
                description=f"Precise {rendered.chart_type} chart rendered from {rendered.source_ref}.",
            )
            svg_artifact = self._write_bytes(
                task,
                "figures/generated/FIGURE_01.svg",
                rendered.svg_bytes,
                kind="image",
                description="Editable vector export of the code-rendered research chart.",
            )
            manifest = {
                "mode": "code",
                "input_files": [rendered.source_ref],
                "sheet": rendered.sheet_name,
                "chart_type": rendered.chart_type,
                "x_column": rendered.x_column,
                "y_columns": rendered.y_columns,
                "row_count": rendered.row_count,
                "outputs": [image_artifact.relative_path, svg_artifact.relative_path],
                "renderer": "matplotlib",
                "warnings": rendered.warnings,
            }
            manifest_artifact = self._write_text(
                task,
                "figures/generated/FIGURE_DELIVERY.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
                kind="manifest",
                description="Traceable delivery manifest for the generated figure.",
            )
            self._log_progress(
                task,
                f"检测到明确数据，已通过代码生成 {rendered.chart_type} 图: {image_artifact.relative_path}",
                kind="figure",
            )
            self.store.save_task(task)
            return [image_artifact, svg_artifact, manifest_artifact]

        prompt = await self._build_figure_render_prompt(task, inventory, briefs)
        prompt_artifact = self._write_text(
            task,
            "figures/generated/FIGURE_RENDER_PROMPT.md",
            prompt,
            kind="note",
            description="Render prompt used for gpt-image-2 figure generation.",
        )
        self._log_progress(task, f"开始调用 {config.image_model} 生成图像")
        size = self._select_figure_image_size(task.objective, briefs)
        image = await generate_image(
            prompt=prompt,
            model=config.image_model,
            size=size,
            quality="low",
            output_format="png",
        )
        image_artifact = self._write_bytes(
            task,
            "figures/generated/FIGURE_01.png",
            image.image_bytes,
            kind="image",
            description=f"Primary generated research figure via {image.model}.",
        )
        metadata = {
            "mode": "image2",
            "input_files": data_sources,
            "model": image.model,
            "size": image.size,
            "quality": image.quality,
            "output_format": image.output_format,
            "mime_type": image.mime_type,
            "revised_prompt": image.revised_prompt,
            "outputs": [image_artifact.relative_path],
            "warnings": data_warnings,
        }
        metadata_artifact = self._write_text(
            task,
            "figures/generated/FIGURE_DELIVERY.json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            kind="manifest",
            description="Traceable delivery manifest for the generated figure.",
        )
        self._log_progress(
            task,
            f"未检测到可精确绘制的数据，已通过 {image.model} 生成科研示意图: {image_artifact.relative_path}",
            kind="figure",
        )
        self.store.save_task(task)
        return [prompt_artifact, image_artifact, metadata_artifact]

    async def _write_presentation_delivery_artifacts(self, task: TaskRun) -> list:
        content = self._artifact_text(task, "presentation/SLIDE_CONTENT.md")
        content_slides = parse_slide_content(content, max_slides=config.presentation_max_slides)
        outline = self._artifact_text(task, "presentation/SLIDES_OUTLINE.md")
        outline_slides = parse_slide_content(outline, max_slides=config.presentation_max_slides)
        slides = reconcile_slide_specs(
            content_slides,
            outline_slides,
            max_slides=config.presentation_max_slides,
        )
        slides = enforce_presentation_page_mix(slides)
        if not slides:
            raise RuntimeError("SLIDE_CONTENT.md does not contain any parseable 'Slide N: Title' sections.")

        workspace_root = Path(task.artifact_root)
        source_config = task.presentation_source or resolve_presentation_source_config(
            task.objective,
            workspace_root,
            source_limit=config.presentation_source_limit,
        )
        mode = source_config.presentation_type
        template = select_presentation_template(task.objective, mode, config.presentation_template)
        session_context = self._session_context_for_task(task)
        assets_path = workspace_root / "Content" / "PRESENTATION_ASSETS.json"
        assets = (
            presentation_assets_from_json(assets_path.read_text(encoding="utf-8"))
            if assets_path.exists()
            else extract_presentation_assets(
                workspace_root,
                source_config.source_refs,
                f"figures/extracted/presentation/{task.task_id}",
            )
        )
        artifacts: list = []
        task.current_stage_name = "presentation_delivery"
        self._log_progress(task, "阶段开始: PPT 页面生成与组装")
        template_artifact = self._write_text(
            task,
            "Content/PRESENTATION_TEMPLATE.json",
            json.dumps(template_manifest(template, mode), ensure_ascii=False, indent=2),
            kind="note",
            description="Selected presentation mode and Image-2 visual template.",
        )
        artifacts.append(template_artifact)
        task.artifacts.append(template_artifact)
        design_spec_artifact = self._write_text(
            task,
            "Content/PRESENTATION_DESIGN_SPEC.json",
            presentation_design_spec_to_json(slides),
            kind="note",
            description="Final page-level presentation render contract.",
        )
        artifacts.append(design_spec_artifact)
        task.artifacts.append(design_spec_artifact)
        self.store.save_task(task)

        renders: list[SlideRender] = []
        used_asset_ids: set[str] = set()
        for slide in slides:
            asset = select_slide_asset(slide, assets, used_asset_ids)
            if not slide.render_mode_explicit:
                if asset is not None:
                    slide = replace(
                        slide,
                        page_type="evidence",
                        render_mode="evidence",
                    )
                elif slide.render_mode == "evidence":
                    slide = replace(
                        slide,
                        page_type="narrative",
                        render_mode="image2_full",
                    )
            if slide.render_mode == "evidence" and asset is None:
                raise RuntimeError(
                    f"Slide {slide.number} is planned as an evidence page but has no valid Asset ID. "
                    "Fix SLIDE_CONTENT.md instead of asking Image-2 to recreate the evidence."
                )
            if asset:
                used_asset_ids.add(asset.asset_id)
            preview_relative_path = f"presentation/slides/SLIDE_{slide.number:02d}.png"
            preview_path = workspace_root / preview_relative_path
            generated_path: Path | None = None
            image_metadata = {
                "model": "",
                "size": "",
                "quality": "",
                "revised_prompt": "",
            }
            if slide.render_mode == "image2_full":
                generated_relative_path = f"presentation/generated/SLIDE_{slide.number:02d}.png"
                generated_path = workspace_root / generated_relative_path
                prompt = build_slide_prompt(slide, template, mode)
                if session_context:
                    prompt = prompt + "\n\nSession context:\n" + session_context
                prompt_artifact = self._write_text(
                    task,
                    f"Content/presentation-prompts/SLIDE_{slide.number:02d}_PROMPT.md",
                    prompt,
                    kind="note",
                    description=f"Image-2 full-page render prompt for slide {slide.number}.",
                )
                artifacts.append(prompt_artifact)
                task.artifacts.append(prompt_artifact)
                same_task_page = any(
                    artifact.relative_path == generated_relative_path for artifact in task.artifacts
                )
                if same_task_page and generated_path.exists() and generated_path.stat().st_size > 0:
                    self._log_progress(task, f"复用已生成的完整页面: {generated_relative_path}")
                else:
                    self._log_progress(
                        task,
                        f"开始调用 {config.image_model} 生成完整PPT页面 {slide.number}/{len(slides)}",
                    )
                    self.store.save_task(task)
                    image = await generate_image(
                        prompt=prompt,
                        model="gpt-image-2",
                        size="1536x1024",
                        quality="medium",
                        output_format="png",
                    )
                    generated_artifact = self._write_bytes(
                        task,
                        generated_relative_path,
                        crop_slide_image(image.image_bytes),
                        kind="image",
                        description=f"Complete standalone Image-2 slide {slide.number}.",
                    )
                    artifacts.append(generated_artifact)
                    task.artifacts.append(generated_artifact)
                    generated_path = Path(generated_artifact.absolute_path)
                    image_metadata = {
                        "model": image.model,
                        "size": image.size,
                        "quality": image.quality,
                        "revised_prompt": image.revised_prompt,
                    }
            else:
                self._log_progress(
                    task,
                    f"使用原始资产排版独立证据页 {slide.number}/{len(slides)}: {asset.asset_id}",
                )
            preview_bytes = compose_slide_preview(
                generated_path,
                slide,
                preview_path,
                workspace_root,
                asset,
            )
            preview_artifact = self._write_bytes(
                task,
                preview_relative_path,
                preview_bytes,
                kind="image",
                description=f"Final preview for standalone slide {slide.number}.",
            )
            artifacts.append(preview_artifact)
            task.artifacts.append(preview_artifact)
            preview_path = Path(preview_artifact.absolute_path)
            metadata = {
                "slide": slide.number,
                "title": slide.title,
                "page_type": slide.page_type,
                "render_mode": slide.render_mode,
                "layout_hint": slide.layout_hint,
                **image_metadata,
                "source_files": list(slide.source_files),
                "asset_id": asset.asset_id if asset else "",
                "asset_kind": asset.kind if asset else "",
                "asset_source": asset.source_path if asset else "",
                "asset_page": asset.page if asset else None,
                "render_policy": (
                    "standalone-original-evidence-page"
                    if slide.render_mode == "evidence"
                    else "standalone-image2-page"
                ),
            }
            metadata_artifact = self._write_text(
                task,
                f"Content/presentation-prompts/SLIDE_{slide.number:02d}_METADATA.json",
                json.dumps(metadata, ensure_ascii=False, indent=2),
                kind="note",
                description=f"Traceability metadata for slide {slide.number}.",
            )
            artifacts.append(metadata_artifact)
            task.artifacts.append(metadata_artifact)
            renders.append(
                SlideRender(
                    slide=slide,
                    base_image=generated_path,
                    preview_image=preview_path,
                    asset=asset,
                )
            )
            self._log_progress(task, f"幻灯片已完成: {preview_relative_path}")
            self.store.save_task(task)

        specs_artifact = self._write_text(
            task,
            "Content/SLIDE_SPECS.json",
            slide_specs_to_json(renders),
            kind="note",
            description="Structured slide specifications and source-asset traceability.",
        )
        artifacts.append(specs_artifact)
        task.artifacts.append(specs_artifact)
        speaker_notes = parse_speaker_notes(
            self._artifact_text(task, "presentation/SPEAKER_NOTES.md"),
            (render.slide for render in renders),
        )
        notes_artifact = self._write_text(
            task,
            "Content/SPEAKER_NOTES.json",
            speaker_notes_to_json(speaker_notes),
            kind="note",
            description="Structured per-slide speaker notes embedded in the final PowerPoint.",
        )
        artifacts.append(notes_artifact)
        task.artifacts.append(notes_artifact)
        filename = "PAPER_TALK.pptx" if mode == "paper" else "STAGE_REPORT.pptx"
        deck = assemble_mixed_deck(
            renders,
            title=self._detect_title(content) or task.objective,
            speaker_notes=speaker_notes,
        )
        deck_artifact = self._write_bytes(
            task,
            f"presentation/{filename}",
            deck,
            kind="presentation",
            description=(
                f"Final {mode} presentation with standalone Image-2 pages, separate original-evidence pages, "
                "and native per-slide speaker notes."
            ),
        )
        artifacts.append(deck_artifact)
        task.artifacts.append(deck_artifact)
        self._log_progress(task, f"PPT 已组装: presentation/{filename}")
        self.store.save_task(task)
        return artifacts

    def _image_mime_type(self, path: Path) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")

    def _make_checkpoint(self, task: TaskRun, stage: StageDefinition, feedback: str) -> ApprovalCheckpoint:
        artifact = task.artifacts[-1] if task.artifacts else None
        decision = self._blocking_decision_section(task, stage)
        prompt_lines = [
            f"Checkpoint: {stage.checkpoint_title or stage.title}",
            f"Task: {task.task_id}",
            f"Objective: {task.objective}",
        ]
        if artifact:
            prompt_lines.append(f"Review artifact: {artifact.absolute_path}")
        if feedback:
            prompt_lines.append(f"Latest feedback applied: {feedback}")
        if decision:
            prompt_lines.extend(["", "Blocking decision:", decision])
            prompt_lines.append("Approve to use Recommended Default, or send your selected option/revision feedback.")
        else:
            prompt_lines.append("Approve to continue, or send revision feedback.")
        checkpoint = ApprovalCheckpoint(
            stage_name=stage.name,
            stage_index=task.current_stage_index,
            title=stage.checkpoint_title or stage.title,
            prompt="\n".join(prompt_lines),
        )
        checkpoint_artifact = self._write_text(
            task,
            f"Content/{stage.name}_checkpoint.md",
            checkpoint.prompt,
            kind="checkpoint",
            description=f"Human checkpoint for {stage.title}.",
        )
        task.artifacts.append(checkpoint_artifact)
        return checkpoint

    def _build_reply(
        self, task: TaskRun, *, text: str, checkpoint: ApprovalCheckpoint | None = None
    ) -> dict:
        latest_artifacts = [artifact.model_dump() for artifact in task.artifacts[-6:]]
        session = self.store.load_session(task.session_id)
        return {
            "text": self._append_cloud_delivery_link(text, session, has_artifacts=bool(task.artifacts)),
            "task_id": task.task_id,
            "status": task.status,
            "command": task.command,
            "workflow_title": task.workflow_title,
            "artifact_root": task.artifact_root,
            "artifacts": latest_artifacts,
            "progress": task.progress_log[-10:],
            "checkpoint": checkpoint.model_dump() if checkpoint else None,
            "cloud_workspace": session.cloud_workspace.model_dump() if session else {},
        }

    def _append_cloud_delivery_link(
        self,
        text: str,
        session: ChatSession | None,
        *,
        has_artifacts: bool = False,
    ) -> str:
        if not has_artifacts or session is None or session.cloud_workspace.status != "synced":
            return text
        cloud_url = self._cloud_workspace_url(session.cloud_workspace)
        if not cloud_url or cloud_url in text:
            return text
        return f"{text}\n\n清华网盘预览/下载链接：{cloud_url}"

    def _cloud_workspace_url(self, cloud_workspace: CloudWorkspaceState) -> str:
        return (
            cloud_workspace.preview_url
            or cloud_workspace.download_url
            or cloud_workspace.share_url
        )

    def enforce_cloud_delivery(
        self,
        task: TaskRun,
        cloud_workspace: CloudWorkspaceState,
    ) -> None:
        if not config.cloud_delivery_required:
            return
        cloud_url = self._cloud_workspace_url(cloud_workspace)
        if cloud_workspace.status == "synced" and cloud_url:
            return
        detail = cloud_workspace.error or cloud_workspace.configuration_hint or cloud_workspace.status
        task.status = "failed"
        task.error = f"产物已在本地生成，但清华网盘交付失败：{detail}"
        task.summary = task.error
        self._log_progress(task, task.error, kind="cloud")
        self.store.save_task(task)

    async def sync_session_workspace(
        self,
        session: ChatSession,
        *,
        task: TaskRun | None = None,
    ) -> CloudWorkspaceState:
        workspace_value = session.workspace_root or (task.artifact_root if task else "")
        if workspace_value:
            workspace_root = Path(workspace_value)
        else:
            workspace_root = self.artifacts.session_root(
                user_id=session.user_id,
                session_id=session.session_id,
            )
            session.workspace_root = str(workspace_root.resolve())
        result = await self.cloud.sync_workspace(
            workspace_root,
            user_id=session.user_id,
            session_id=session.session_id,
        )
        cloud_config = self.cloud.configuration_status()
        session.cloud_workspace = CloudWorkspaceState(
            provider="seafile",
            status=result.status,
            configured=bool(cloud_config["configured"]),
            configuration_hint=(
                result.error
                if result.status == "error"
                else str(cloud_config["hint"])
            ),
            auth_mode=str(cloud_config["auth_mode"]),
            remote_path=result.remote_path,
            share_url=result.share_url,
            preview_url=result.preview_url,
            download_url=result.download_url,
            repo_id=result.repo_id,
            synced_files=result.synced_files,
            uploaded_files=result.uploaded_files,
            last_synced_at=utc_now() if result.status == "synced" else "",
            error=result.error,
        )
        self.store.save_session(session)
        if task:
            if result.status == "synced":
                cloud_url = self._cloud_workspace_url(session.cloud_workspace)
                link_suffix = f"；清华网盘工作区：{cloud_url}" if cloud_url else ""
                self._log_progress(
                    task,
                    f"已同步 {result.uploaded_files} 个更新文件到清华网盘 {result.remote_path}{link_suffix}",
                    kind="cloud",
                )
            elif result.status == "error":
                self._log_progress(task, f"Cloud sync warning: {result.error}", kind="cloud")
            self.store.save_task(task)
        return session.cloud_workspace

    async def sync_task_workspace(self, task: TaskRun) -> CloudWorkspaceState:
        session = self.store.load_session(task.session_id)
        if session is None:
            return CloudWorkspaceState(
                status="error",
                error=f"Unknown session: {task.session_id}",
            )
        return await self.sync_session_workspace(session, task=task)

    def _log_progress(self, task: TaskRun, message: str, *, kind: str = "progress") -> None:
        task.progress_log.append(message)
        previous_sequence = task.progress_events[-1].sequence if task.progress_events else 0
        task.progress_events.append(
            ProgressEvent(
                sequence=previous_sequence + 1,
                kind=kind,
                message=message,
                stage=task.current_stage_name,
            )
        )
        task.progress_events = task.progress_events[-500:]
        self.store.save_task(task)

    def _finalize_task_record(self, task: TaskRun, workflow: WorkflowDefinition):
        relative_path = f"wiki/agent-notes/{task.task_id}.md"
        existing_artifacts = [
            artifact for artifact in task.artifacts if artifact.relative_path != relative_path
        ]
        task.artifacts = existing_artifacts
        if workflow.stage_definitions:
            task.current_stage_name = workflow.stage_definitions[-1].name
        task.status = "running"
        task.summary = "研究产物已生成，正在完成云端交付。"
        wiki_note = self._write_text(
            task,
            relative_path,
            self.wiki.render_task_note(
                task,
                status="completed",
                summary=f"{workflow.title} completed with {len(existing_artifacts) + 1} artifacts.",
            ),
            kind="wiki",
            description="Session-local task record and artifact index.",
        )
        task.artifacts.append(wiki_note)
        task.notes = [note for note in task.notes if not note.startswith("Wiki note: ")]
        task.notes.append(f"Wiki note: {wiki_note.absolute_path}")
        return wiki_note

    def _mark_task_completed(self, task: TaskRun, workflow: WorkflowDefinition) -> None:
        task.status = "completed"
        task.summary = f"{workflow.title} completed with {len(task.artifacts)} artifacts."
        self._log_progress(task, f"工作流完成: {workflow.title}")
        self.store.save_task(task)

    def _write_review_delivery_artifacts(self, task: TaskRun) -> list:
        citation_audit, coverage_report = build_review_quality_reports(Path(task.artifact_root))
        return [
            self._write_text(
                task,
                "bib/CITATION_AUDIT.json",
                json.dumps(citation_audit, ensure_ascii=False, indent=2),
                kind="review",
                description="Deterministic stable-paper-ID citation audit.",
            ),
            self._write_text(
                task,
                "bib/REVIEW_COVERAGE.json",
                json.dumps(coverage_report, ensure_ascii=False, indent=2),
                kind="review",
                description="Review retrieval, traceability, and citation coverage report.",
            ),
        ]

    def _write_text(self, task: TaskRun, relative_path: str, content: str, *, kind, description: str):
        artifact = self.artifacts.write_text(
            task.task_id,
            self._canonical_artifact_path(relative_path),
            content,
            kind=kind,
            description=description,
            task_root=task.artifact_root,
        )
        self._schedule_cloud_sync(task.session_id)
        return artifact

    def _write_bytes(self, task: TaskRun, relative_path: str, content: bytes, *, kind, description: str):
        artifact = self.artifacts.write_bytes(
            task.task_id,
            self._canonical_artifact_path(relative_path),
            content,
            kind=kind,
            description=description,
            task_root=task.artifact_root,
        )
        self._schedule_cloud_sync(task.session_id)
        return artifact

    def _record_existing(self, task: TaskRun, relative_path: str, *, kind, description: str):
        artifact = self.artifacts.record_existing(
            task.task_id,
            self._canonical_artifact_path(relative_path),
            kind=kind,
            description=description,
            task_root=task.artifact_root,
        )
        self._schedule_cloud_sync(task.session_id)
        return artifact

    def _upsert_task_artifact(self, task: TaskRun, artifact) -> None:
        task.artifacts = [
            existing
            for existing in task.artifacts
            if existing.relative_path != artifact.relative_path
        ]
        task.artifacts.append(artifact)

    def _record_idea_trace(
        self,
        task: TaskRun,
        stage: StageDefinition,
        *,
        selected_model: str | None,
        elapsed_ms: float,
        fallback_used: bool,
        evidence_validation: dict | None,
    ) -> None:
        trace_path = Path(task.artifact_root) / "Content" / "IDEA_TRACE.json"
        trace: dict = {
            "task_id": task.task_id,
            "orchestration": "single-herness-stage-routing",
            "stages": [],
        }
        if trace_path.exists():
            try:
                loaded = json.loads(trace_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    trace.update(loaded)
            except json.JSONDecodeError:
                pass
        stages = trace.setdefault("stages", [])
        stages.append(
            {
                "stage": stage.name,
                "model_role": stage.model_role,
                "model": selected_model or "auto",
                "fallback_used": fallback_used,
                "elapsed_ms": elapsed_ms,
                "recorded_at": utc_now(),
            }
        )
        if evidence_validation is not None:
            trace["evidence_validation"] = evidence_validation
        trace_artifact = self._write_text(
            task,
            "Content/IDEA_TRACE.json",
            json.dumps(trace, ensure_ascii=False, indent=2),
            kind="note",
            description="Idea stage model routing and evidence validation trace.",
        )
        self._upsert_task_artifact(task, trace_artifact)

    def _validate_idea_evidence(self, task: TaskRun, content: str) -> tuple[str, dict]:
        identifier_pattern = re.compile(r"\b(?:PE-[A-F0-9]{10}|P\d{3}|P-[A-F0-9]{12})\b", re.I)
        workspace_root = Path(task.artifact_root)
        evidence_paths = [
            workspace_root / "bib" / "EVIDENCE_MAP.md",
            workspace_root / "bib" / "LITERATURE_REVIEW.md",
            workspace_root / "bib" / "LITERATURE_SEARCH.md",
            workspace_root / "wiki" / "query_pack.md",
        ]
        evidence_paths.extend((workspace_root / "wiki" / "papers").glob("*/summary.md"))
        allowed_ids: set[str] = set()
        evidence_pages: dict[str, int] = {}
        for path in evidence_paths:
            if not path.exists():
                continue
            evidence_text = path.read_text(encoding="utf-8", errors="ignore")
            allowed_ids.update(identifier.upper() for identifier in identifier_pattern.findall(evidence_text))
            for match in re.finditer(
                r"(?P<id>PE-[A-F0-9]{10})(?:[^\n\]]{0,40}?)(?:p\.?\s*|第\s*)(?P<page>\d+)(?:\s*页)?",
                evidence_text,
                re.I,
            ):
                evidence_pages.setdefault(match.group("id").upper(), int(match.group("page")))
        referenced_ids = {
            identifier.upper() for identifier in identifier_pattern.findall(content)
        }
        invalid_ids = sorted(referenced_ids - allowed_ids)
        validated_content = content
        for invalid_id in invalid_ids:
            validated_content = re.sub(
                rf"\b{re.escape(invalid_id)}\b",
                f"UNVERIFIED({invalid_id})",
                validated_content,
                flags=re.I,
            )
        for evidence_id, page in evidence_pages.items():
            validated_content = re.sub(
                rf"\[\s*{re.escape(evidence_id)}\s*\]",
                f"[{evidence_id}, p.{page}]",
                validated_content,
                flags=re.I,
            )
        page_linked_ids = sorted(
            evidence_id
            for evidence_id in referenced_ids
            if evidence_id.startswith("PE-")
            and re.search(
                rf"{re.escape(evidence_id)}(?:[^\n\]]{{0,30}}?)(?:p\.?\s*\d+|第\s*\d+\s*页)",
                validated_content,
                re.I,
            )
        )
        unpaged_ids = sorted(
            evidence_id
            for evidence_id in referenced_ids
            if evidence_id.startswith("PE-") and evidence_id not in page_linked_ids
        )
        status = "pass"
        if invalid_ids:
            status = "needs_attention"
            validated_content = (
                validated_content.rstrip()
                + "\n\n## Evidence Validation\n\n"
                + "- Status: needs attention\n"
                + "- 以下编号未出现在当前证据包中，已标记为未验证："
                + ", ".join(f"`{identifier}`" for identifier in invalid_ids)
                + "\n"
            )
        elif not referenced_ids:
            status = "no_explicit_evidence_ids"
        elif unpaged_ids:
            status = "needs_attention"
        return validated_content, {
            "status": status,
            "allowed_ids": sorted(allowed_ids),
            "referenced_ids": sorted(referenced_ids),
            "invalid_ids": invalid_ids,
            "page_linked_ids": page_linked_ids,
            "unpaged_ids": unpaged_ids,
        }

    @staticmethod
    def _normalize_generated_markdown(content: str, required_sections: list[str]) -> str:
        normalized = content.strip()
        outer_fence = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*?)\n```", normalized, re.I | re.S)
        if outer_fence:
            normalized = outer_fence.group(1).strip()
        variants = re.split(r"\n```(?:markdown|md)\s*\n(?=#)", normalized, flags=re.I)
        candidates: list[tuple[int, int, str]] = []
        for variant in variants:
            cleaned = re.sub(r"\n```\s*$", "", variant.strip()).strip()
            section_count = sum(
                bool(re.search(rf"(?mi)^##+\s*{re.escape(section)}(?:\s|$)", cleaned))
                for section in required_sections
            )
            candidates.append((section_count, -len(cleaned), cleaned))
        if candidates:
            normalized = max(candidates)[2]
        return normalized.rstrip() + "\n"

    def _schedule_cloud_sync(self, session_id: str) -> None:
        if not config.cloud_sync_enabled:
            return
        active = self._cloud_sync_jobs.get(session_id)
        if active and not active.done():
            self._cloud_sync_pending.add(session_id)
            return

        async def run_sync() -> None:
            try:
                while True:
                    self._cloud_sync_pending.discard(session_id)
                    session = self.store.load_session(session_id)
                    if session:
                        await self.sync_session_workspace(session)
                    if session_id not in self._cloud_sync_pending:
                        break
            finally:
                self._cloud_sync_jobs.pop(session_id, None)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cloud_sync_jobs[session_id] = loop.create_task(run_sync())

    def _canonical_artifact_path(self, relative_path: str) -> str:
        normalized = relative_path.replace("\\", "/")
        replacements = {
            "plan-stage/": "plan/",
            "idea-stage/": "idea/",
            "refine-logs/": "plan/",
            "code-stage/": "code/",
            "fig-stage/": "figures/",
            "review-stage/": "bib/",
            "present/": "presentation/",
            "wiki-stage/": "wiki/",
            "hitl/": "Content/",
            "utils/": "Content/",
        }
        for old, new in replacements.items():
            if normalized.startswith(old):
                return new + normalized[len(old) :]
        return normalized

    def _build_stage_prompt(
        self, task: TaskRun, workflow: WorkflowDefinition, stage: StageDefinition, revision_feedback: str
    ) -> dict[str, str]:
        skill_context = self._skill_context(stage.skill_paths)
        lightweight_workflow = task.command in {"/wiki", "/idea"}
        prd_context = "" if lightweight_workflow else self._file_excerpt(config.prd_path, 5000)
        tech_context = "" if lightweight_workflow else self._file_excerpt(config.tech_spec_path, 5000)
        session_context = self._session_context_for_task(task)
        prior_artifact_sections: list[str] = []
        for artifact in reversed(task.artifacts):
            if not self._is_prompt_text_artifact(artifact.relative_path):
                continue
            excerpt = self._artifact_excerpt(task, artifact.relative_path)
            if not excerpt:
                continue
            prior_artifact_sections.append(f"### {artifact.relative_path}\n{excerpt}")
            if len(prior_artifact_sections) == 4:
                break
        prior_artifacts = "\n\n".join(reversed(prior_artifact_sections))
        write_output_hint = ""
        if task.command == "/write":
            write_output_hint = (
                "Requested delivery formats: "
                + ", ".join(detect_write_formats(task.objective))
                + ". Structure the draft so it can be exported cleanly.\n\n"
            )
        checkpoint_hint = ""
        if stage.hitl:
            checkpoint_hint = (
                "Checkpoint policy: continue automatically with a conservative recommended choice whenever possible. "
                "Do not pause for routine review, quality approval, wording, layout preferences, or reversible choices. "
                "A checkpoint is allowed only when the decision materially changes scope, cost, risk, claims, or an external "
                "commitment and available evidence cannot resolve it for the user. Do not add a 'Decision Required' section "
                "unless such a blocking choice truly exists. For a blocking choice, use exactly this structure: "
                "'Blocking: Yes', 'Question: ...', 'Why user input is necessary: ...', at least two named lines such as "
                "'Option A: ...' and 'Option B: ...', and 'Recommended Default: ...'. Missing fields cause the workflow to "
                "choose its recommendation and continue automatically.\n\n"
            )
        language_hint = ""
        if re.search(r"[\u4e00-\u9fff]", task.objective):
            language_hint = (
                "Output language policy: write all explanatory prose in clear Simplified Chinese. "
                "Keep the required section headings exactly as provided so validation remains stable, "
                "but do not write English paragraphs. English is allowed only for proper nouns, model or dataset names, "
                "formulas, code, paths, and evidence identifiers.\n\n"
            )
        artifact_budget_hint = ""
        if task.command == "/idea":
            stage_budgets = {
                "idea_candidates": "3,500",
                "idea_verification": "2,500",
                "final_idea": "4,800",
            }
            budget = stage_budgets.get(stage.name, "2,600")
            artifact_budget_hint = (
                f"Artifact budget: keep the complete artifact within about {budget} Chinese characters. "
                "Return exactly one Markdown document and never wrap the whole artifact in a code fence. "
                "Do not repeat prior artifacts. Do not claim 'first', 'state of the art', complete novelty, "
                "or preservation of the original theory unless the supplied evidence directly proves that claim. "
                "Every factual or numeric claim must cite an Evidence ID with page, or be labeled 未确认. "
                "Do not assign symbols, formulas, datasets, or parameter meanings that are absent from the evidence.\n\n"
            )
        final_idea_writing_hint = ""
        if task.command == "/idea" and stage.name == "final_idea":
            final_idea_writing_hint = (
                "Final Idea reader profile: the document will be read by strong graduate and doctoral researchers who need "
                "to understand the idea quickly and judge whether it is worth developing. Write professionally but do not use "
                "dense slogan-like phrases. Use the exact Chinese headings below. Start each section with a direct conclusion, "
                "then explain the reasoning in plain academic Chinese. The first section must contain a one-sentence Idea and "
                "a short plain-language explanation. The gap section must say whether the gap is author-explicit or an "
                "evidence-backed inference. The method section must explain the mechanism, why it may work, and which parts are "
                "still assumptions. The innovation section must keep one dominant contribution and avoid a shopping list. The "
                "evidence section should map key claims to Evidence IDs and page numbers. The risk section must include the "
                "strongest plausible rejection argument and the condition under which the idea should be abandoned. Use short "
                "paragraphs, bullets, or a small table only when they improve readability. Do not repeat the Wiki summary or "
                "write a full experiment plan. Use only the nine required Chinese sections after the title. Do not mention "
                "skills, prompts, agents, execution metadata, or add a separate plain-language summary after the last section.\n\n"
            )
        user_prompt = (
            f"Workflow: {workflow.title}\n"
            f"Command: {task.command}\n"
            f"Objective: {task.objective}\n"
            f"Current stage: {stage.title}\n"
            f"Route source: {task.route_source}\n\n"
            + (
                "Applied ARIS skills:\n"
                + "\n".join(f"- {name}" for name in self._skill_names(stage.skill_paths))
                + "\n\n"
                if stage.skill_paths
                else ""
            )
            + write_output_hint
            + checkpoint_hint
            + language_hint
            + artifact_budget_hint
            + final_idea_writing_hint
            + f"Stage instruction:\n{stage.instruction}\n\n"
            + "Required sections:\n"
            + "\n".join(f"- {section}" for section in stage.required_sections)
            + "\n\n"
            + (f"Relevant prior session context:\n{session_context}\n\n" if session_context else "")
            + (f"Revision feedback to incorporate:\n{revision_feedback}\n\n" if revision_feedback else "")
            + (f"Recent artifacts:\n{prior_artifacts}\n\n" if prior_artifacts else "")
            + "Return only the markdown for the artifact file. Keep it concrete, honest, and execution-oriented."
        )
        specification_context = ""
        if prd_context or tech_context:
            specification_context = (
                f"PRD excerpt:\n{prd_context}\n\n"
                f"Tech spec excerpt:\n{tech_context}\n\n"
            )
        system_prompt = (
            "You are the orchestration core of a research agent platform. "
            "You must follow the PRD and tech-spec constraints, use the ARIS skill patterns as execution guidance, "
            "and produce file-ready markdown artifacts. Separate assumptions from grounded facts, use human checkpoints only for genuine user decisions, "
            "and optimize for local collaboration with a human researcher. "
            f"The sole writable research workspace for this task is: {Path(task.artifact_root).resolve()}. "
            "This path is exactly agent-workspace/local/<session_id>/. Every model-generated research artifact, task note, "
            "context file, log, intermediate file, and final deliverable must remain inside this one session directory. "
            "Never write or propose writing research files to the project root, research-wiki/, .agent-state/, a task-specific "
            "folder, another user, or another session. The only allowed top-level directories inside the session are exactly "
            "bib/, plan/, idea/, code/, figures/, paper/, presentation/, rebuttal/, wiki/, Content/, and logs/. Do not create "
            "additional top-level directories. Use wiki/ for session memory and task notes. Use Content/ only for context, "
            "progress records, checkpoints, prompts, traceability metadata, and manifests. When mentioning output paths in an "
            "artifact, use paths relative to the session root. "
            + language_hint
            + f"{CLOUD_DELIVERY_SYSTEM_POLICY}\n\n"
            + specification_context
            + f"Relevant ARIS guidance:\n{skill_context}\n"
        )
        return {"system": system_prompt, "user": user_prompt}

    async def _build_figure_render_prompt(self, task: TaskRun, inventory: str, briefs: str) -> str:
        session_context = self._session_context_for_task(task)
        manuscript = self._artifact_text(task, "paper/PAPER_REVISED.md") or self._artifact_text(task, "paper/PAPER_DRAFT.md")
        manuscript_block = f"Current paper excerpt:\n{manuscript[:4000]}\n\n" if manuscript else ""
        system_prompt = (
            "You convert research figure plans into a single production-ready gpt-image-2 prompt. "
            "Return plain prompt text only, no markdown, no bullets. "
            "Prefer clean academic diagrams, short English labels, high readability, white background, and publication-friendly layout. "
            "If this task belongs to an ongoing session with prior paper, review, plan, or code work, keep the same topic, "
            "claims, terminology, and visual framing instead of inventing a new subject from the figure command alone."
        )
        context_block = (
            f"Relevant prior session context:\n{session_context}\n\n" if session_context else ""
        )
        user_prompt = (
            f"Objective:\n{task.objective}\n\n"
            f"{context_block}"
            f"{manuscript_block}"
            f"Figure inventory:\n{inventory}\n\n"
            f"Figure briefs:\n{briefs[:6000]}\n\n"
            "Choose the single highest-value figure to render first. "
            "Describe composition, visual hierarchy, color palette, labels, arrows, panel structure, and style constraints clearly enough for image generation. "
            "When prior session context exists, inherit its topic and evidence rather than defaulting to a generic workflow diagram."
        )
        content = await generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )
        prompt = content.strip().strip("`")
        if prompt:
            return prompt
        fallback = (
            f"Create a clean academic research figure for: {task.objective}. "
            "Use a white background, blue and slate accents, clear panel layout, thin arrows, short English labels, and publication-ready typography. "
            "Make it look like a polished systems or workflow diagram rather than marketing art."
        )
        return fallback

    def _session_context_for_task(self, task: TaskRun, *, task_limit: int = 3, artifact_limit: int = 3) -> str:
        session = self.store.load_session(task.session_id)
        if not session:
            return ""
        return self._session_context(session, task_limit=task_limit, artifact_limit=artifact_limit)

    def _session_task_context(self, task: TaskRun) -> str:
        prior_tasks = [item for item in self.store.list_tasks(task.session_id) if item.task_id != task.task_id]
        blocks: list[str] = []
        total_chars = 0
        for prior in prior_tasks[:3]:
            lines = [
                f"## Prior Task {prior.task_id}",
                f"Command: {prior.command}",
                f"Objective: {prior.objective}",
                f"Status: {prior.status}",
            ]
            if prior.summary:
                lines.append(f"Summary: {prior.summary}")
            artifacts = sorted(
                prior.artifacts,
                key=lambda artifact: self._handoff_artifact_priority(task.command, artifact.relative_path),
            )
            included = 0
            for artifact in artifacts:
                if artifact.kind in {"checkpoint", "manifest"}:
                    continue
                excerpt = self._artifact_excerpt(prior, artifact.relative_path)
                if not excerpt:
                    continue
                lines.append(f"### {artifact.relative_path}")
                lines.append(excerpt[:1600])
                included += 1
                if included >= 3:
                    break
            block = "\n".join(lines)
            total_chars += len(block)
            if total_chars > 7000:
                break
            blocks.append(block)
        return "\n\n".join(blocks)

    def _handoff_artifact_priority(self, command: str, relative_path: str) -> tuple[int, str]:
        preferred = {
            "/idea": (
                "bib/RESEARCH_GAPS.md",
                "bib/EVIDENCE_MAP.md",
                "bib/LITERATURE_REVIEW.md",
            ),
            "/plan": (
                "idea/FINAL_IDEA.md",
                "idea/docs/research_contract.md",
                "bib/EVIDENCE_MAP.md",
                "bib/RESEARCH_GAPS.md",
            ),
            "/code": (
                "plan/EXECUTION_CHECKLIST.md",
                "plan/EXPERIMENT_PLAN.md",
                "plan/RESEARCH_BLUEPRINT.md",
            ),
        }.get(command, ())
        try:
            return preferred.index(relative_path), relative_path
        except ValueError:
            return len(preferred) + 1, relative_path

    async def _write_delivery_artifacts(self, task: TaskRun) -> list:
        revised = self._artifact_text(task, "paper/PAPER_REVISED.md")
        manuscript = revised or self._artifact_text(task, "paper/PAPER_DRAFT.md")
        if not manuscript:
            return []
        manuscript_stem = "PAPER_REVISED" if revised else "PAPER_DRAFT"
        manuscript_relative = f"paper/{manuscript_stem}.md"
        formats = detect_write_formats(task.objective)
        artifacts = []
        title = self._detect_title(manuscript) or "Research Draft"
        if "tex" in formats:
            tex = markdown_to_latex(manuscript, title=title)
            artifacts.append(
                self._write_text(
                    task,
                    f"paper/{manuscript_stem}.tex",
                    tex,
                    kind="document",
                    description="LaTeX export generated from the evidence-reviewed paper manuscript.",
                )
            )
        if "docx" in formats:
            docx_bytes = markdown_to_docx_bytes(manuscript)
            artifacts.append(
                self._write_bytes(
                    task,
                    f"paper/{manuscript_stem}.docx",
                    docx_bytes,
                    kind="document",
                    description="Word export generated from the evidence-reviewed paper manuscript.",
                )
            )
        if "pdf" in formats:
            pdf_bytes = markdown_to_pdf_bytes(manuscript)
            artifacts.append(
                self._write_bytes(
                    task,
                    f"paper/{manuscript_stem}.pdf",
                    pdf_bytes,
                    kind="document",
                    description="PDF export generated from the evidence-reviewed paper manuscript.",
                )
            )
        artifacts.extend(await self._compile_paper_manuscript(task, manuscript, manuscript_stem))
        evidence_records = self._paper_evidence_records(task)
        compile_status_relative = f"paper/{manuscript_stem}_COMPILE_STATUS.md"
        citation_audit, delivery_report = build_paper_quality_reports(
            Path(task.artifact_root),
            manuscript_relative,
            evidence_records,
            select_venue_profile(task.objective),
            compile_status_relative=compile_status_relative,
        )
        artifacts.append(
            self._write_text(
                task,
                "paper/CITATION_AUDIT.json",
                json.dumps(citation_audit, ensure_ascii=False, indent=2),
                kind="review",
                description="Deterministic citation-key audit for the revised paper.",
            )
        )
        artifacts.append(
            self._write_text(
                task,
                "paper/PAPER_DELIVERY_REPORT.json",
                json.dumps(delivery_report, ensure_ascii=False, indent=2),
                kind="review",
                description="Deterministic paper structure, evidence, citation, and compile delivery gate.",
            )
        )
        return artifacts

    def _write_rebuttal_delivery_artifacts(self, task: TaskRun) -> list:
        report = build_rebuttal_closure_report(Path(task.artifact_root))
        return [
            self._write_text(
                task,
                "rebuttal/REBUTTAL_CLOSURE_REPORT.json",
                json.dumps(report, ensure_ascii=False, indent=2),
                kind="review",
                description="Deterministic comment coverage and declared revision-status gate.",
            )
        ]

    async def _compile_paper_manuscript(self, task: TaskRun, manuscript: str, stem: str) -> list:
        tex = markdown_to_latex(
            manuscript,
            title=self._detect_title(manuscript) or "Research Draft",
        )
        build_dir = Path(task.artifact_root) / "paper" / ".compile"
        build_dir.mkdir(parents=True, exist_ok=True)
        source = build_dir / f"{stem}.tex"
        source.write_text(tex, encoding="utf-8")
        compiler = self._resolve_tex_compiler()
        if not compiler:
            return [
                self._write_text(
                    task,
                    f"paper/{stem}_COMPILE_STATUS.md",
                    "# Paper Compile Status\n\n- Status: failed\n- Reason: LaTeX compiler not available.\n",
                    kind="note",
                    description="Paper compile status report.",
                )
            ]
        outputs: list = []
        try:
            command = [
                compiler,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(build_dir),
                str(source),
            ]
            first = subprocess.run(command, check=False, capture_output=True, text=True, timeout=240)
            second = subprocess.run(command, check=False, capture_output=True, text=True, timeout=240)
            log_text = "\n\n".join(
                [
                    "## First pass stdout",
                    first.stdout or "",
                    "## First pass stderr",
                    first.stderr or "",
                    "## Second pass stdout",
                    second.stdout or "",
                    "## Second pass stderr",
                    second.stderr or "",
                ]
            )
            outputs.append(
                self._write_text(
                    task,
                    f"paper/{stem}_COMPILE_LOG.txt",
                    log_text,
                    kind="note",
                    description="LaTeX compilation log for the paper draft.",
                )
            )
            pdf_path = build_dir / f"{stem}.pdf"
            if pdf_path.exists():
                outputs.append(
                    self._write_bytes(
                        task,
                        f"paper/{stem}_COMPILED.pdf",
                        pdf_path.read_bytes(),
                        kind="document",
                        description=f"Compiled PDF generated via {Path(compiler).name}.",
                    )
                )
                outputs.append(
                    self._write_text(
                        task,
                        f"paper/{stem}_COMPILE_STATUS.md",
                        "# Paper Compile Status\n\n- Status: success\n- Compiler: "
                        + Path(compiler).name
                        + f"\n- Output: `paper/{stem}_COMPILED.pdf`\n",
                        kind="note",
                        description="Paper compile status report.",
                    )
                )
            else:
                outputs.append(
                    self._write_text(
                        task,
                        f"paper/{stem}_COMPILE_STATUS.md",
                        "# Paper Compile Status\n\n- Status: failed\n- Reason: PDF not produced.\n",
                        kind="note",
                        description="Paper compile status report.",
                    )
                )
        except Exception as exc:
            outputs.append(
                self._write_text(
                    task,
                    f"paper/{stem}_COMPILE_STATUS.md",
                    f"# Paper Compile Status\n\n- Status: failed\n- Reason: {exc.__class__.__name__}: {exc}\n",
                    kind="note",
                    description="Paper compile status report.",
                )
            )
        return outputs

    def _detect_title(self, markdown: str) -> str:
        lines = markdown.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped.lower() == "# title" and index + 2 < len(lines):
                candidate = lines[index + 2].strip("* ").strip()
                if candidate:
                    return candidate
        return ""

    def _skill_context(self, skill_paths: list[str]) -> str:
        contexts: list[str] = []
        aris_root = Path(config.aris_repo_root)
        total_chars = 0
        total_limit = 18000
        for relative in skill_paths:
            target = self._resolve_skill_path(aris_root, relative)
            if not target.exists():
                contexts.append(f"## {relative}\nMissing skill file.")
                continue
            excerpt = self._skill_excerpt(target, limit=2200)
            if not excerpt:
                continue
            block = f"## {relative}\n{excerpt}"
            total_chars += len(block)
            if total_chars > total_limit and contexts:
                contexts.append("## skill-budget\nAdditional ARIS skills were omitted after the prompt budget cap.")
                break
            contexts.append(block)
        return "\n\n".join(contexts) or "No skill context found."

    def _skill_names(self, skill_paths: list[str]) -> list[str]:
        names: list[str] = []
        for relative in skill_paths:
            names.append(Path(relative).parent.name)
        return names

    def _scholar_skill_names(self) -> set[str]:
        return {
            "research-lit",
            "prior-art-search",
            "openalex",
            "semantic-scholar",
            "arxiv",
            "deepxiv",
            "comm-lit-review",
            "novelty-check",
            "idea-discovery",
            "idea-creator",
            "idea-discovery-robot",
            "research-pipeline",
            "research-refine",
            "research-refine-pipeline",
            "paper-plan",
            "paper-writing",
            "paper-write",
            "claims-drafting",
            "research-review",
            "research-wiki",
            "wiki-enrich",
            "result-to-claim",
        }

    def _search_query_for_task(self, objective: str) -> str:
        cleaned = re.sub(r"^\s*/\w+\s*", "", objective).strip()
        return cleaned[:180] if cleaned else objective[:180]

    def _resolve_tex_compiler(self) -> str | None:
        for candidate in ("xelatex", "latexmk", "pdflatex"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    def _resolve_skill_path(self, aris_root: Path, relative_path: str) -> Path:
        return aris_root / relative_path

    def _skill_excerpt(self, path: Path, limit: int) -> str:
        text = path.read_text(encoding="utf-8", errors="ignore")
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        metadata = ""
        body = text
        if frontmatter_match:
            metadata_block = frontmatter_match.group(1)
            body = frontmatter_match.group(2)
            name_match = re.search(r'^name:\s*"?(.*?)"?\s*$', metadata_block, re.M)
            desc_match = re.search(r'^description:\s*"?(.*?)"?\s*$', metadata_block, re.M)
            parts: list[str] = []
            if name_match:
                parts.append(f"Skill name: {name_match.group(1).strip()}")
            if desc_match:
                parts.append(f"Skill purpose: {desc_match.group(1).strip()}")
            metadata = "\n".join(parts)

        lines: list[str] = []
        for raw_line in body.splitlines():
            stripped = raw_line.rstrip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if stripped.startswith("```"):
                continue
            lines.append(stripped)
            if len("\n".join(lines)) >= limit:
                break

        body_excerpt = "\n".join(lines)[:limit].strip()
        combined = "\n".join(part for part in [metadata, body_excerpt] if part).strip()
        return combined[:limit]

    def _file_excerpt(self, path: str, limit: int) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8", errors="ignore")[:limit]

    def _artifact_excerpt(self, task: TaskRun, relative_path: str) -> str:
        if not self._is_prompt_text_artifact(relative_path):
            return ""
        target = self._artifact_path(task, relative_path)
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8", errors="ignore")[:4000]

    @staticmethod
    def _is_prompt_text_artifact(relative_path: str) -> bool:
        return Path(relative_path).suffix.lower() in {
            ".bib",
            ".csv",
            ".enw",
            ".json",
            ".jsonl",
            ".md",
            ".nbib",
            ".ris",
            ".tex",
            ".tsv",
            ".txt",
            ".yaml",
            ".yml",
        }

    def _artifact_text(self, task: TaskRun, relative_path: str) -> str:
        target = self._artifact_path(task, relative_path)
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8", errors="ignore")

    def _artifact_path(self, task: TaskRun, relative_path: str) -> Path:
        root = Path(task.artifact_root)
        canonical = root / self._canonical_artifact_path(relative_path)
        if canonical.exists():
            return canonical
        return root / relative_path

    def _select_figure_image_size(self, objective: str, briefs: str) -> str:
        text = f"{objective}\n{briefs}".lower()
        portrait_keywords = ("poster", "vertical", "portrait", "竖", "海报")
        if any(keyword in text for keyword in portrait_keywords):
            return "1024x1536"
        square_keywords = ("icon", "logo", "square", "示意图标")
        if any(keyword in text for keyword in square_keywords):
            return "1024x1024"
        return "1536x1024"

    def _strip_command(self, message: str, command: str, alias: str | None = None) -> str:
        stripped = message.strip()
        tokens = [command]
        if alias and alias not in tokens:
            tokens.append(alias)
        for token in tokens:
            if stripped.startswith(token):
                return stripped[len(token) :].strip() or stripped
        return stripped
