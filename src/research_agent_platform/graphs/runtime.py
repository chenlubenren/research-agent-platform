from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..config import config
from ..models import ApprovalCheckpoint, TaskRun
from .checkpointer import PersistentMemorySaver
from .workflows import StageDefinition, WorkflowDefinition

if TYPE_CHECKING:
    from ..agent import ResearchAgentService


class WorkflowGraphState(TypedDict, total=False):
    task_id: str
    revision_feedback: str
    approval_decision: str


class LangGraphWorkflowRuntime:
    def __init__(
        self,
        service: ResearchAgentService,
        workflows: dict[str, WorkflowDefinition],
    ) -> None:
        self.service = service
        self.workflows = workflows
        self.checkpointer = PersistentMemorySaver(
            Path(config.state_root) / "langgraph-checkpoints.pkl"
        )
        self.graphs = {
            command: self._compile_workflow_graph(workflow)
            for command, workflow in workflows.items()
        }

    async def start_task(self, task: TaskRun, workflow: WorkflowDefinition) -> dict:
        await self.graphs[workflow.command].ainvoke(
            {"task_id": task.task_id, "revision_feedback": "", "approval_decision": ""},
            config=self._graph_config(task.task_id),
        )
        return self.build_reply(task.task_id, workflow)

    async def resume_task(
        self,
        task: TaskRun,
        workflow: WorkflowDefinition,
        *,
        approved: bool,
        feedback: str,
    ) -> dict:
        await self.graphs[workflow.command].ainvoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config=self._graph_config(task.task_id),
        )
        return self.build_reply(task.task_id, workflow)

    async def continue_task(self, task: TaskRun, workflow: WorkflowDefinition) -> dict:
        if not self.has_graph_state(task.task_id):
            raise ValueError(f"Task {task.task_id} has no workflow checkpoint to resume.")
        await self.graphs[workflow.command].ainvoke(
            None,
            config=self._graph_config(task.task_id),
        )
        return self.build_reply(task.task_id, workflow)

    def has_graph_state(self, task_id: str) -> bool:
        return self.checkpointer.get_tuple(self._graph_config(task_id)) is not None

    def build_reply(self, task_id: str, workflow: WorkflowDefinition) -> dict:
        task = self.service.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        checkpoint = self._pending_checkpoint(task)
        if task.status == "completed":
            final_artifact = self._final_delivery_artifact(task, workflow)
            text = self._completed_text(workflow.title, task, final_artifact)
        elif task.status == "failed":
            text = task.error or f"{workflow.title} 执行失败。"
        else:
            text = task.summary or f"{workflow.title} 正在执行。"
        return self.service._build_reply(task, text=text, checkpoint=checkpoint)

    def _compile_workflow_graph(self, workflow: WorkflowDefinition):
        builder = StateGraph(WorkflowGraphState)
        for index, stage in enumerate(workflow.stage_definitions):
            execute_name = self._execute_node_name(index)
            builder.add_node(execute_name, self._make_execute_stage_node(workflow, stage, index))
            if stage.hitl:
                approval_name = self._approval_node_name(index)
                builder.add_node(
                    approval_name,
                    self._make_approval_node(workflow, stage, index),
                )
                builder.add_conditional_edges(
                    execute_name,
                    self._make_checkpoint_router(stage),
                    {
                        "checkpoint": approval_name,
                        "continue": self._next_node_name(workflow, index),
                    },
                )
                builder.add_conditional_edges(
                    approval_name,
                    self._make_approval_router(index, workflow),
                    {
                        "approved": self._next_node_name(workflow, index),
                        "rejected": execute_name,
                    },
                )
            else:
                builder.add_edge(execute_name, self._next_node_name(workflow, index))

        builder.add_node("finalize", self._make_finalize_node(workflow))
        builder.add_edge(START, self._execute_node_name(0))
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer, name=workflow.command)

    def _make_execute_stage_node(
        self,
        workflow: WorkflowDefinition,
        stage: StageDefinition,
        stage_index: int,
    ):
        async def execute_stage(state: WorkflowGraphState) -> WorkflowGraphState:
            task = self._load_task(state)
            feedback = str(state.get("revision_feedback", ""))
            is_rerun = self._is_rejected_rerun(task, stage_index, feedback)

            task.current_stage_index = stage_index
            task.current_stage_name = stage.name
            task.status = "running"
            if is_rerun:
                self.service._log_progress(task, f"正在根据反馈重生成阶段: {stage.title}")
            self.service._log_progress(task, f"阶段开始: {stage.title}")
            self.service.store.save_task(task)

            artifact = await self.service._execute_stage(task, workflow, stage, feedback)
            task.artifacts.append(artifact)
            self.service._log_progress(task, f"阶段完成: {stage.title} -> {artifact.relative_path}")
            if is_rerun:
                self.service._log_progress(task, f"阶段已重生成: {stage.title}")
            self.service.store.save_task(task)
            await self.service.sync_task_workspace(task)

            return {
                "task_id": task.task_id,
                "revision_feedback": "",
                "approval_decision": "",
            }

        return execute_stage

    def _make_checkpoint_router(self, stage: StageDefinition):
        def route_after_stage(state: WorkflowGraphState) -> str:
            task = self._load_task(state)
            return "checkpoint" if self.service._stage_requires_checkpoint(task, stage) else "continue"

        return route_after_stage

    def _make_approval_node(
        self,
        workflow: WorkflowDefinition,
        stage: StageDefinition,
        stage_index: int,
    ):
        async def approval_gate(state: WorkflowGraphState) -> WorkflowGraphState:
            task = self._load_task(state)
            pending = self._pending_checkpoint(task)
            if pending is None or pending.stage_index != stage_index:
                pending = self.service._make_checkpoint(
                    task,
                    stage,
                    str(state.get("revision_feedback", "")),
                )
                task.approvals.append(pending)
                task.status = "waiting_human"
                task.summary = self._approval_waiting_summary(task, stage, stage_index)
                self.service._log_progress(task, f"等待人工审批: {pending.title}")
                self.service.store.save_task(task)
                await self.service.sync_task_workspace(task)

            decision = interrupt(
                {
                    "task_id": task.task_id,
                    "command": workflow.command,
                    "checkpoint": pending.model_dump(),
                }
            )

            task = self._load_task(state)
            pending = self._pending_checkpoint(task)
            if pending is None:
                raise ValueError(f"Task {task.task_id} has no pending checkpoint for approval.")
            feedback = str((decision or {}).get("feedback", ""))
            approved = bool((decision or {}).get("approved"))
            pending.feedback = feedback
            pending.resolved_at = pending.resolved_at or pending.created_at
            task.status = "running"

            if approved:
                pending.status = "approved"
                self.service._log_progress(task, f"已批准 checkpoint: {pending.title}")
                task.current_stage_index = stage_index + 1
                task.current_stage_name = (
                    workflow.stage_definitions[stage_index + 1].name
                    if stage_index + 1 < len(workflow.stage_definitions)
                    else ""
                )
            else:
                pending.status = "rejected"
                self.service._log_progress(task, f"已打回 checkpoint: {pending.title} | 反馈: {feedback}")
                task.current_stage_index = stage_index
                task.current_stage_name = stage.name

            self.service.store.save_task(task)
            return {
                "task_id": task.task_id,
                "approval_decision": "approved" if approved else "rejected",
                "revision_feedback": feedback,
            }

        return approval_gate

    def _make_approval_router(self, stage_index: int, workflow: WorkflowDefinition):
        def route_after_approval(state: WorkflowGraphState) -> str:
            decision = state.get("approval_decision", "approved")
            if decision == "rejected":
                return "rejected"
            return "approved"

        return route_after_approval

    def _make_finalize_node(self, workflow: WorkflowDefinition):
        async def finalize(state: WorkflowGraphState) -> WorkflowGraphState:
            task = self._load_task(state)
            if task.command == "/idea":
                await self.service._write_research_contract(task)
            if task.command == "/write":
                task.artifacts.extend(await self.service._write_delivery_artifacts(task))
            if task.command == "/rebuttal":
                task.artifacts.extend(self.service._write_rebuttal_delivery_artifacts(task))
            if task.command == "/review":
                task.artifacts.extend(self.service._write_review_delivery_artifacts(task))
            if task.command == "/present":
                await self.service._write_presentation_delivery_artifacts(task)

            self.service._finalize_task_record(task, workflow)
            self.service.store.save_task(task)
            cloud_workspace = await self.service.sync_task_workspace(task)
            self.service.enforce_cloud_delivery(task, cloud_workspace)
            if task.status != "failed":
                self.service._mark_task_completed(task, workflow)

            session = self.service.store.load_session(task.session_id)
            if session:
                session.active_task_id = None
                self.service.store.save_session(session)

            return {"task_id": task.task_id, "revision_feedback": "", "approval_decision": ""}

        return finalize

    def _load_task(self, state: WorkflowGraphState) -> TaskRun:
        task_id = str(state.get("task_id", ""))
        task = self.service.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        return task

    def _graph_config(self, task_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": task_id}}

    def _pending_checkpoint(self, task: TaskRun) -> ApprovalCheckpoint | None:
        if not task.approvals:
            return None
        latest = task.approvals[-1]
        return latest if latest.status == "pending" else None

    def _latest_wiki_note(self, task: TaskRun) -> str:
        for note in reversed(task.notes):
            if note.startswith("Wiki note: "):
                return note.replace("Wiki note: ", "", 1)
        return ""

    def _session_relative_path(self, task: TaskRun, path: str) -> str:
        try:
            return Path(path).resolve().relative_to(Path(task.artifact_root).resolve()).as_posix()
        except ValueError:
            return path

    def _completed_text(self, title: str, task: TaskRun, final_artifact: str) -> str:
        artifact_path = self._session_relative_path(task, final_artifact) if final_artifact else ""
        if artifact_path:
            return f"{title} 已完成。最终产物位于 `{artifact_path}`。"
        return f"{title} 已完成。最终产物已写入本会话工作区。"

    def _final_delivery_artifact(self, task: TaskRun, workflow: WorkflowDefinition) -> str:
        command = workflow.command
        if command == "/fig":
            for path in ("figures/generated/FIGURE_01.png", "figures/FIGURE_01.png"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/present":
            for path in ("presentation/PAPER_TALK.pptx", "presentation/STAGE_REPORT.pptx"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/write":
            for path in ("paper/PAPER_REVISED_AFTER_REVIEW.md", "paper/PAPER_REVISED.md", "paper/PAPER_DRAFT.md"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/idea":
            for path in ("idea/FINAL_IDEA.md", "idea/docs/research_contract.md"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/plan":
            for path in ("plan/EXECUTION_CHECKLIST.md", "plan/EXPERIMENT_PLAN.md", "plan/RESEARCH_BLUEPRINT.md"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/rebuttal":
            for path in ("rebuttal/REVISION_LEDGER.md", "rebuttal/REBUTTAL_DRAFT.md"):
                if self._artifact_exists(task, path):
                    return path
        if command == "/review":
            for path in ("bib/RESEARCH_GAPS.md", "bib/EVIDENCE_MAP.md", "bib/LITERATURE_REVIEW.md"):
                if self._artifact_exists(task, path):
                    return path
        return ""

    def _artifact_exists(self, task: TaskRun, relative_path: str) -> bool:
        return any(artifact.relative_path == relative_path for artifact in task.artifacts)

    def _approval_waiting_summary(
        self,
        task: TaskRun,
        stage: StageDefinition,
        stage_index: int,
    ) -> str:
        latest_rejected = any(
            checkpoint.stage_index == stage_index and checkpoint.status == "rejected"
            for checkpoint in task.approvals
        )
        if latest_rejected:
            return f"{stage.title} 已根据反馈重生成。请检查更新后的产物，确认后继续，或再次打回。"
        return f"{stage.title} 已完成，正在等待你的审核。你可以批准继续，或填写修改意见后打回本阶段。"

    def _is_rejected_rerun(self, task: TaskRun, stage_index: int, feedback: str) -> bool:
        if not feedback or not task.approvals:
            return False
        latest = task.approvals[-1]
        return latest.stage_index == stage_index and latest.status == "rejected"

    def _execute_node_name(self, stage_index: int) -> str:
        return f"execute_stage_{stage_index}"

    def _approval_node_name(self, stage_index: int) -> str:
        return f"approval_stage_{stage_index}"

    def _next_node_name(self, workflow: WorkflowDefinition, stage_index: int) -> str:
        next_index = stage_index + 1
        if next_index >= len(workflow.stage_definitions):
            return "finalize"
        return self._execute_node_name(next_index)
