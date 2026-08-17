from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


TaskStatus = Literal["running", "waiting_human", "completed", "failed"]
CheckpointStatus = Literal["pending", "approved", "rejected"]
RouteSource = Literal["explicit", "implicit_heuristic", "implicit_llm", "approval", "chat"]
PresentationType = Literal["paper", "stage"]
PresentationSourceScope = Literal["auto", "attachments", "selected", "session", "workspace"]
DownloadSourceScope = Literal["auto", "attachments", "selected", "session", "workspace"]
FigureKind = Literal[
    "data_plot",
    "framework_diagram",
    "mechanism_diagram",
    "reference_reproduction",
]
FigureRenderer = Literal["matplotlib", "academic_svg", "drawio", "figurespec"]
FigureEditability = Literal["publication_vector", "structural"]
FigureSourceScope = Literal["auto", "attachments", "selected", "workspace"]
CloudSyncStatus = Literal["disabled", "pending", "synced", "error"]
ArtifactKind = Literal[
    "report",
    "plan",
    "review",
    "slides",
    "wiki",
    "contract",
    "manifest",
    "checkpoint",
    "note",
    "document",
    "image",
    "presentation",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class MessageRecord(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=utc_now)


class ArtifactRecord(BaseModel):
    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    name: str
    kind: ArtifactKind
    relative_path: str
    absolute_path: str
    url_path: str
    description: str
    created_at: str = Field(default_factory=utc_now)


class UploadBatchRecord(BaseModel):
    upload_batch_id: str = Field(default_factory=lambda: new_id("upload"))
    relative_paths: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class PresentationSourceConfig(BaseModel):
    presentation_type: PresentationType
    requested_scope: PresentationSourceScope = "auto"
    resolved_scope: PresentationSourceScope
    source_refs: list[str] = Field(default_factory=list)
    upload_batch_id: str = ""
    selection_reason: str = ""


class RebuttalSourceConfig(BaseModel):
    paper_refs: list[str] = Field(default_factory=list)
    review_refs: list[str] = Field(default_factory=list)
    paper_selection_reason: str = ""
    review_selection_reason: str = ""
    upload_batch_ids: list[str] = Field(default_factory=list)


class WriteSourceConfig(BaseModel):
    requested_scope: PresentationSourceScope = "auto"
    resolved_scope: PresentationSourceScope
    source_refs: list[str] = Field(default_factory=list)
    upload_batch_id: str = ""
    selection_reason: str = ""


class DownloadSourceConfig(BaseModel):
    requested_scope: DownloadSourceScope = "auto"
    resolved_scope: DownloadSourceScope
    source_refs: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    upload_batch_id: str = ""
    selection_reason: str = ""


class FigureSourceConfig(BaseModel):
    requested_scope: FigureSourceScope = "auto"
    resolved_scope: FigureSourceScope
    source_refs: list[str] = Field(default_factory=list)
    data_refs: list[str] = Field(default_factory=list)
    image_refs: list[str] = Field(default_factory=list)
    upload_batch_id: str = ""
    selection_reason: str = ""


class FigurePanel(BaseModel):
    panel_id: str
    title: str
    purpose: str = ""


class FigureEntity(BaseModel):
    entity_id: str
    label: str
    role: str = "component"
    panel_id: str = "a"


class FigureRelation(BaseModel):
    source: str
    target: str
    label: str = ""
    relation_type: Literal["flow", "inhibition", "feedback", "association"] = "flow"


class FigureContract(BaseModel):
    schema_version: str = "1.0"
    figure_id: str = "FIGURE_01"
    kind: FigureKind
    renderer: FigureRenderer
    route_reason: str
    purpose: str
    core_claim: str
    panels: list[FigurePanel] = Field(default_factory=list)
    entities: list[FigureEntity] = Field(default_factory=list)
    relations: list[FigureRelation] = Field(default_factory=list)
    label_allowlist: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    language: str = "English"
    target_width_px: int = 1600
    target_height_px: int = 960
    editability: FigureEditability = "publication_vector"
    editability_required: bool = True
    prohibited_content: list[str] = Field(default_factory=list)
    decision_required: bool = False
    decision_question: str = ""


class VisualStyleSpec(BaseModel):
    schema_version: str = "1.0"
    preset: Literal["academic_clean", "playful_academic"] = "academic_clean"
    palette: list[str] = Field(default_factory=list)
    font_family: str = "Arial"
    cjk_font_family: str = "PingFang SC"
    corner_radius: int = 12
    hatch_backgrounds: bool = False
    icon_density: Literal["none", "low", "medium"] = "low"
    arrow_style: str = "academic"
    panel_style: str = "subtle"


class LayoutPlan(BaseModel):
    """Renderer-neutral topology and reading-order intent, without final coordinates."""

    schema_version: str = "1.0"
    reading_direction: Literal["RIGHT", "DOWN"] = "RIGHT"
    panel_order: list[str] = Field(default_factory=list)
    node_order: list[str] = Field(default_factory=list)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    port_preferences: dict[str, Literal["NORTH", "SOUTH", "EAST", "WEST"]] = Field(
        default_factory=dict
    )
    edge_types: dict[str, str] = Field(default_factory=dict)


class FigureDeliveryManifest(BaseModel):
    schema_version: Literal["3.0"] = "3.0"
    mode: str
    figure_id: str
    figure_kind: FigureKind
    renderer: FigureRenderer
    backend_skill: str
    route_reason: str
    input_files: list[str] = Field(default_factory=list)
    source_config: FigureSourceConfig
    contract: str
    visual_style: str
    layout_plan: str = ""
    authoritative_source: str
    derived_outputs: list[str] = Field(default_factory=list)
    assist_assets: list[str] = Field(default_factory=list)
    asset_sources: list[dict[str, Any]] = Field(default_factory=list)
    editability: FigureEditability
    layout_engine: str = "not_applicable"
    canonical: bool = True
    topology_verified: bool = True
    raster_inside_drawio: bool = False
    edit_banana_called: bool = False
    vlm_called: bool = False
    caption: str
    qa: str
    publication_status: str
    warnings: list[str] = Field(default_factory=list)
    wps_handoff: str = ""
    render_spec: str = ""
    sheet: str = ""
    chart_type: str = ""
    x_column: str = ""
    y_columns: list[str] = Field(default_factory=list)
    row_count: int = 0


class CloudWorkspaceState(BaseModel):
    provider: str = "seafile"
    status: CloudSyncStatus = "disabled"
    configured: bool = False
    configuration_hint: str = ""
    auth_mode: str = ""
    remote_path: str = ""
    share_url: str = ""
    preview_url: str = ""
    download_url: str = ""
    repo_id: str = ""
    synced_files: int = 0
    uploaded_files: int = 0
    last_synced_at: str = ""
    error: str = ""


class CloudAuthProfile(BaseModel):
    user_id: str
    provider: str = "seafile"
    base_url: str = ""
    api_token: str = ""
    repo_id: str = ""
    repo_name: str = "Research Agent"
    remote_root: str = "research-agent"
    share_links: bool = True
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class ProgressEvent(BaseModel):
    sequence: int
    kind: str = "progress"
    message: str
    stage: str = ""
    timestamp: str = Field(default_factory=utc_now)


class ApprovalCheckpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: new_id("checkpoint"))
    stage_name: str
    stage_index: int
    title: str
    prompt: str
    status: CheckpointStatus = "pending"
    feedback: str = ""
    created_at: str = Field(default_factory=utc_now)
    resolved_at: str | None = None


class TaskRun(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    session_id: str
    user_id: str = "local"
    command: str
    objective: str
    route_source: RouteSource
    workflow_title: str
    status: TaskStatus = "running"
    current_stage_index: int = 0
    current_stage_name: str = ""
    artifact_root: str = ""
    summary: str = ""
    response_text: str = ""
    error: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    approvals: list[ApprovalCheckpoint] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    progress_log: list[str] = Field(default_factory=list)
    progress_events: list[ProgressEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    presentation_source: PresentationSourceConfig | None = None
    rebuttal_source: RebuttalSourceConfig | None = None
    write_source: WriteSourceConfig | None = None
    download_source: DownloadSourceConfig | None = None
    figure_source: FigureSourceConfig | None = None


class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("session"))
    user_id: str = "local"
    workspace_root: str = ""
    active_task_id: str | None = None
    history: list[MessageRecord] = Field(default_factory=list)
    upload_batches: list[UploadBatchRecord] = Field(default_factory=list)
    cloud_workspace: CloudWorkspaceState = Field(default_factory=CloudWorkspaceState)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
