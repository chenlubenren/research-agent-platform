from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .models import (
    FigureContract,
    FigureEntity,
    FigurePanel,
    FigureRelation,
    FigureSourceConfig,
    UploadBatchRecord,
    VisualStyleSpec,
)


DATA_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
REFERENCE_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}
SOURCE_EXTENSIONS = DATA_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf", ".md", ".txt", ".docx", ".pptx"}

MECHANISM_TERMS = (
    "mechanism",
    "机制",
    "机理",
    "交互",
    "生动",
    "图标",
    "插画",
    "illustrated",
)
STRUCTURAL_EDIT_TERMS = (
    "draw.io",
    "drawio",
    "拖拽",
    "移动节点",
    "改结构",
    "替换节点",
    "箭头跟随",
    "结构可编辑",
)
PUBLICATION_VECTOR_TERMS = ("academic svg", "svg", "投稿", "出版矢量")
FRAMEWORK_TERMS = (
    "framework",
    "architecture",
    "pipeline",
    "workflow",
    "flowchart",
    "架构",
    "框架",
    "流程",
    "技术路线",
    "系统图",
)
REFERENCE_TERMS = (
    "参考图",
    "复刻",
    "仿照",
    "同款",
    "style reference",
    "reproduce",
    "replicate",
)
DATA_TERMS = (
    "plot",
    "chart",
    "data",
    "accuracy",
    "loss",
    "trend",
    "distribution",
    "compare",
    "数据",
    "图表",
    "曲线",
    "柱状",
    "散点",
    "热力",
    "箱线",
    "小提琴",
    "误差",
    "趋势",
    "分布",
    "对比结果",
)


def resolve_figure_source_config(
    objective: str,
    workspace_root: Path,
    upload_batches: Iterable[UploadBatchRecord] = (),
) -> FigureSourceConfig:
    batches = list(upload_batches)
    latest_batch = batches[-1] if batches else None
    explicit_refs = _extract_explicit_refs(objective, workspace_root)

    if explicit_refs:
        requested_scope = "selected"
        resolved_scope = "selected"
        source_refs = explicit_refs
        upload_batch_id = ""
        reason = "用户在 /fig 指令中明确指定了输入文件。"
    elif latest_batch and latest_batch.relative_paths:
        requested_scope = "auto"
        resolved_scope = "attachments"
        source_refs = _existing_refs(workspace_root, latest_batch.relative_paths)
        upload_batch_id = latest_batch.upload_batch_id
        reason = "自动模式优先冻结当前会话最近一次上传批次。"
    else:
        requested_scope = "auto"
        resolved_scope = "workspace"
        source_refs = _workspace_sources(workspace_root)
        upload_batch_id = ""
        reason = "未发现显式路径或上传批次，按限定目录检索工作区证据。"

    data_refs = [ref for ref in source_refs if Path(ref).suffix.lower() in DATA_EXTENSIONS]
    image_refs = [ref for ref in source_refs if Path(ref).suffix.lower() in REFERENCE_EXTENSIONS]
    return FigureSourceConfig(
        requested_scope=requested_scope,
        resolved_scope=resolved_scope,
        source_refs=source_refs,
        data_refs=data_refs,
        image_refs=image_refs,
        upload_batch_id=upload_batch_id,
        selection_reason=reason,
    )


def build_figure_contract(objective: str, sources: FigureSourceConfig) -> FigureContract:
    normalized = objective.lower()
    has_reference_request = any(term in normalized for term in REFERENCE_TERMS)
    has_mechanism_request = any(term in normalized for term in MECHANISM_TERMS)
    has_data_request = any(term in normalized for term in DATA_TERMS)
    wants_structural_editing = any(term in normalized for term in STRUCTURAL_EDIT_TERMS)
    wants_publication_vector = any(term in normalized for term in PUBLICATION_VECTOR_TERMS)
    requests_legacy_figurespec = "figurespec" in normalized

    if wants_structural_editing:
        kind = (
            "reference_reproduction"
            if has_reference_request
            else "mechanism_diagram"
            if has_mechanism_request
            else "framework_diagram"
        )
        renderer = "drawio"
        route_reason = "用户明确要求结构级编辑、节点拖拽或 Draw.io 格式。"
        editability = "structural"
    elif requests_legacy_figurespec:
        kind = "mechanism_diagram" if has_mechanism_request else "framework_diagram"
        renderer = "academic_svg"
        route_reason = "旧 FigureSpec 请求已映射到 Academic SVG；FigureSpec 渲染路线已弃用。"
        editability = "publication_vector"
    elif has_reference_request and sources.image_refs:
        kind = "reference_reproduction"
        renderer = "drawio"
        route_reason = "检测到参考图附件和拆解请求，交由可选 Edit Banana 服务生成 Draw.io 草稿。"
        editability = "structural"
    elif sources.resolved_scope == "selected" and sources.data_refs:
        kind = "data_plot"
        renderer = "matplotlib"
        route_reason = "用户明确指定了表格数据文件，且未提出相冲突的机制图或框架图要求。"
        editability = "publication_vector"
    elif has_data_request and sources.data_refs:
        kind = "data_plot"
        renderer = "matplotlib"
        route_reason = "请求明确表达数值可视化意图，且冻结输入中存在表格数据。"
        editability = "publication_vector"
    elif has_reference_request:
        kind = "reference_reproduction"
        renderer = "drawio"
        route_reason = "请求拆解参考视觉；等待冻结参考图后调用可选 Edit Banana 服务。"
        editability = "structural"
    elif has_data_request:
        kind = "data_plot"
        renderer = "matplotlib"
        route_reason = "请求明确表达数值可视化意图；等待冻结可读取的数据源。"
        editability = "publication_vector"
    else:
        kind = "mechanism_diagram" if has_mechanism_request else "framework_diagram"
        renderer = "academic_svg"
        route_reason = (
            "请求包含机制或交互语义；采用机制模板，但默认仍输出投稿级 Academic SVG。"
            if has_mechanism_request
            else "未检测到数据绘图或强结构编辑要求，默认输出投稿级 Academic SVG。"
        )
        if wants_publication_vector:
            route_reason = "用户明确要求 SVG/投稿矢量输出，使用 Academic SVG。"
        editability = "publication_vector"

    language = "Chinese" if re.search(r"[\u4e00-\u9fff]", objective) else "English"
    purpose = _clean_objective(objective) or "Create one evidence-grounded research figure."
    panels = _infer_panels(objective, purpose, language)
    entities = _infer_entities(objective, kind)
    for index, entity in enumerate(entities):
        panel_index = min(len(panels) - 1, index * len(panels) // max(1, len(entities)))
        entity.panel_id = panels[panel_index].panel_id
    relation_label = "流向" if language == "Chinese" else "flows to"
    relations = [
        FigureRelation(source=left.entity_id, target=right.entity_id, label=relation_label)
        for left, right in zip(entities, entities[1:])
    ]
    labels = _deduplicate(
        [entity.label for entity in entities]
        + [relation.label for relation in relations if relation.label]
        + [panel.title for panel in panels]
    )
    decision_required = False
    decision_question = ""
    if has_data_request and not sources.data_refs:
        decision_required = True
        decision_question = "数据图请求尚未冻结到可读取的 CSV/TSV/XLSX 数据源，请指定或上传数据文件。"
    elif has_reference_request and not sources.image_refs:
        decision_required = True
        decision_question = "参考图复刻请求尚未冻结到参考图片，请指定或上传图片。"
    return FigureContract(
        kind=kind,
        renderer=renderer,
        route_reason=route_reason,
        purpose=purpose,
        core_claim=purpose,
        panels=panels,
        entities=entities,
        relations=relations,
        label_allowlist=labels,
        evidence_refs=sources.source_refs,
        language=language,
        editability=editability,
        prohibited_content=[
            "labels not present in label_allowlist",
            "untraceable quantitative claims",
            "rasterized structural text or arrows",
        ],
        decision_required=decision_required,
        decision_question=decision_question,
    )


def normalize_figure_renderer(renderer: str) -> tuple[str, list[str]]:
    if renderer == "figurespec":
        return "academic_svg", ["renderer 'figurespec' is deprecated; mapped to 'academic_svg'"]
    return renderer, []


def build_visual_style_spec(contract: FigureContract) -> VisualStyleSpec:
    playful = contract.kind in {"mechanism_diagram", "reference_reproduction"}
    return VisualStyleSpec(
        preset="playful_academic" if playful else "academic_clean",
        palette=(
            ["#E85D4A", "#35A853", "#3A9ACD", "#D99A1E", "#8A68B8"]
            if playful
            else ["#2463A7", "#2D8A63", "#8A6BBE", "#D4553D"]
        ),
        font_family="Arial",
        cjk_font_family="PingFang SC",
        corner_radius=14 if playful else 10,
        hatch_backgrounds=playful,
        icon_density="medium" if playful else "low",
        arrow_style="semantic" if playful else "academic",
        panel_style="colored-tabs" if playful else "subtle",
    )


def contract_to_json(contract: FigureContract) -> str:
    return json.dumps(contract.model_dump(), ensure_ascii=False, indent=2)


def style_to_json(style: VisualStyleSpec) -> str:
    return json.dumps(style.model_dump(), ensure_ascii=False, indent=2)


def _extract_explicit_refs(objective: str, workspace_root: Path) -> list[str]:
    candidates = re.findall(
        r"[^\s`\"']+\.(?:csv|tsv|xlsx|xls|png|jpe?g|webp|gif|svg|pdf|md|txt|docx|pptx)",
        objective,
        re.I,
    )
    return _existing_refs(workspace_root, candidates)


def _existing_refs(workspace_root: Path, refs: Iterable[str]) -> list[str]:
    selected: list[str] = []
    resolved_root = workspace_root.resolve()
    for ref in refs:
        cleaned = ref.replace("\\", "/").strip(".,;:，。；：")
        supplied_path = Path(cleaned)
        normalized = cleaned.lstrip("/")
        path = supplied_path if supplied_path.is_absolute() else workspace_root / normalized
        if not path.is_file() and "/" not in normalized:
            matches = [
                candidate
                for candidate in workspace_root.rglob(normalized)
                if candidate.is_file() and candidate.suffix.lower() in SOURCE_EXTENSIONS
            ]
            if len(matches) == 1:
                path = matches[0]
                normalized = path.relative_to(workspace_root).as_posix()
        try:
            resolved_path = path.resolve()
            inside_workspace = resolved_path.is_relative_to(resolved_root)
        except OSError:
            inside_workspace = False
        if inside_workspace and path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
            selected.append(resolved_path.relative_to(resolved_root).as_posix())
    return _deduplicate(selected)


def _workspace_sources(workspace_root: Path, limit: int = 40) -> list[str]:
    selected: list[str] = []
    for directory_name in ("figures", "Content", "paper", "plan", "code"):
        directory = workspace_root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_EXTENSIONS
                and "generated" not in {part.lower() for part in path.parts}
            ):
                selected.append(path.relative_to(workspace_root).as_posix())
                if len(selected) >= limit:
                    return selected
    return selected


def _infer_entities(objective: str, kind: str) -> list[FigureEntity]:
    cleaned = _clean_objective(objective)
    parts = [
        _short_label(part)
        for part in re.split(r"(?:->|→|=>|⇒|\bthen\b|然后|再到|，|,|；|;|、)", cleaned, flags=re.I)
    ]
    labels = [part for part in parts if 1 < len(part) <= 36]
    if len(labels) < 2:
        defaults = {
            "data_plot": ["Input Data", "Analysis", "Result"],
            "framework_diagram": ["Input", "Core Method", "Output"],
            "mechanism_diagram": ["Stimulus", "Mechanism", "Outcome"],
            "reference_reproduction": ["Input", "Process", "Result"],
        }
        labels = defaults[kind]
    labels = _deduplicate(labels[:6])
    return [
        FigureEntity(entity_id=f"node_{index + 1}", label=label)
        for index, label in enumerate(labels)
    ]


def _infer_panels(objective: str, purpose: str, language: str) -> list[FigurePanel]:
    normalized = objective.lower()
    match = re.search(r"([2-4])\s*(?:个\s*)?(?:panels?|面板|子图)", normalized)
    if match:
        count = int(match.group(1))
    elif any(term in normalized for term in ("multi-panel", "multipanel", "多面板", "多子图")):
        count = 2
    else:
        count = 1
    if count == 1:
        title = "主图" if language == "Chinese" else "Main Figure"
        return [FigurePanel(panel_id="a", title=title, purpose=purpose)]
    return [
        FigurePanel(
            panel_id=chr(ord("a") + index),
            title=(
                f"面板 {chr(ord('A') + index)}"
                if language == "Chinese"
                else f"Panel {chr(ord('A') + index)}"
            ),
            purpose=purpose,
        )
        for index in range(count)
    ]


def _clean_objective(objective: str) -> str:
    cleaned = re.sub(r"^/(?:fig|figure)\s*", "", objective.strip(), flags=re.I)
    cleaned = re.sub(r"\b(?:draw|generate|create|make)\b", "", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip(" -:：")


def _short_label(value: str) -> str:
    value = re.sub(r"(?:，|,)?(?:要有|采用|使用).*$", "", value)
    value = re.sub(r"(?:的)?(?:多面板|多子图|面板|子图)(?:框架图|机制图|流程图)?$", "", value)
    value = re.sub(
        r"(?:请|帮我|画一张|画一个|生成一张|生成一个|科研|论文|机制图|框架图|流程图|架构图|图表)",
        "",
        value,
    )
    value = re.sub(r"^(?:画|生成|制作)", "", value)
    return re.sub(r"\s+", " ", value).strip(" -:：的")[:36]


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
