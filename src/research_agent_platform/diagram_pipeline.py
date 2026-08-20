from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

from PIL import ImageFont

from .figure_contracts import normalize_figure_renderer
from .models import FigureContract, LayoutPlan, VisualStyleSpec


@dataclass
class DiagramRenderResult:
    renderer: str
    render_spec: dict[str, Any]
    editable_extension: str
    editable_bytes: bytes
    svg_bytes: bytes
    png_bytes: bytes
    pdf_bytes: bytes
    qa: dict[str, Any]


def render_diagram(contract: FigureContract, style: VisualStyleSpec) -> DiagramRenderResult:
    render_spec = build_diagram_render_spec(contract, style)
    renderer, renderer_issues = normalize_figure_renderer(contract.renderer)
    svg_bytes = _render_academic_svg(render_spec, style).encode("utf-8")
    if renderer == "drawio":
        editable_extension = "drawio"
        editable_bytes = _render_drawio_xml(render_spec, style).encode("utf-8")
    else:
        editable_extension = "svg"
        editable_bytes = svg_bytes
    png_bytes, pdf_bytes = svg_to_png_pdf(svg_bytes)
    qa = validate_diagram(
        contract,
        style,
        render_spec,
        svg_bytes,
        editable_bytes,
        renderer_issues=renderer_issues,
    )
    return DiagramRenderResult(
        renderer=renderer,
        render_spec=render_spec,
        editable_extension=editable_extension,
        editable_bytes=editable_bytes,
        svg_bytes=svg_bytes,
        png_bytes=png_bytes,
        pdf_bytes=pdf_bytes,
        qa=qa,
    )


def build_diagram_render_spec(
    contract: FigureContract,
    style: VisualStyleSpec,
) -> dict[str, Any]:
    width = contract.target_width_px
    height = contract.target_height_px
    entities = contract.entities
    layout_plan = build_layout_plan(contract)
    panels = _layout_panels(contract, style)
    panel_map = {panel["id"]: panel for panel in panels}
    default_panel_id = panels[0]["id"]
    grouped_entities: dict[str, list[Any]] = {panel["id"]: [] for panel in panels}
    for entity in entities:
        grouped_entities.get(entity.panel_id, grouped_entities[default_panel_id]).append(entity)
    font_path = _resolve_font_path(
        style.cjk_font_family if contract.language == "Chinese" else style.font_family,
        require_cjk=contract.language == "Chinese",
    )
    font_size = 22
    nodes: list[dict[str, Any]] = []
    palette_index = 0
    for panel_id, panel_entities in grouped_entities.items():
        for entity in panel_entities:
            lines, text_width, text_height = _measure_wrapped_text(
                entity.label,
                font_path,
                font_size,
                max_text_width=220,
            )
            node_width = max(180, min(330, math.ceil(text_width + 116)))
            node_height = max(104, math.ceil(text_height + 48))
            icon_name = _icon_name(palette_index, entity.role, entity.label)
            icon_asset = _load_icon_asset(icon_name)
            nodes.append(
                {
                    "id": entity.entity_id,
                    "label": entity.label,
                    "role": entity.role,
                    "panel_id": panel_id,
                    "x": 0.0,
                    "y": 0.0,
                    "width": node_width,
                    "height": node_height,
                    "text_lines": lines,
                    "text_width": round(text_width, 1),
                    "text_height": round(text_height, 1),
                    "shape": "rounded",
                    "fill": _lighten(style.palette[palette_index % len(style.palette)]),
                    "stroke": style.palette[palette_index % len(style.palette)],
                    "icon": icon_name,
                    "icon_asset": icon_asset,
                }
            )
            palette_index += 1
    edges = [
        {
            "id": f"edge_{index + 1}",
            "from": relation.source,
            "to": relation.target,
            "label": relation.label,
            "semantic": relation.relation_type,
            "color": _edge_color(relation.relation_type),
            "style": _edge_style(relation.relation_type),
            "thickness": 3 if style.preset == "playful_academic" else 2,
        }
        for index, relation in enumerate(contract.relations)
    ]
    layout_engine, layout_warnings = _place_nodes(nodes, edges, panels, layout_plan)
    return {
        "schema_version": "2.0",
        "canvas": {"width": width, "height": height},
        "layout_plan": layout_plan.model_dump(),
        "layout_engine": layout_engine,
        "layout_warnings": layout_warnings,
        "style": {
            "font_family": (
                style.cjk_font_family if contract.language == "Chinese" else style.font_family
            ),
            "font_size": font_size,
            "font_path": str(font_path) if font_path else "",
            "font_resolved": font_path is not None,
            "bg_color": "#FFFFFF",
            "palette": style.palette,
            "preset": style.preset,
            "hatch_backgrounds": style.hatch_backgrounds,
            "corner_radius": style.corner_radius,
        },
        "panels": panels,
        "nodes": nodes,
        "edges": edges,
        "ports": [
            {"id": f"{node['id']}_east", "node_id": node["id"], "side": "EAST"}
            for node in nodes
        ]
        + [
            {"id": f"{node['id']}_west", "node_id": node["id"], "side": "WEST"}
            for node in nodes
        ],
        "groups": [
            {"id": f"group_{panel_id}", "panel_id": panel_id, "node_ids": node_ids}
            for panel_id, node_ids in layout_plan.groups.items()
        ],
        "labels": [],
        "asset_sources": _deduplicate_dicts(
            [node["icon_asset"]["provenance"] for node in nodes if node.get("icon_asset")]
        ),
    }


def build_layout_plan(contract: FigureContract) -> LayoutPlan:
    edge_types = {
        f"edge_{index + 1}": relation.relation_type
        for index, relation in enumerate(contract.relations)
    }
    return LayoutPlan(
        reading_direction="DOWN" if len(contract.panels) > 2 else "RIGHT",
        panel_order=[panel.panel_id for panel in contract.panels],
        node_order=[entity.entity_id for entity in contract.entities],
        groups={panel.panel_id: [entity.entity_id for entity in contract.entities if entity.panel_id == panel.panel_id] for panel in contract.panels},
        edge_types=edge_types,
    )


def _place_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    plan: LayoutPlan,
) -> tuple[str, list[str]]:
    used_elk = True
    warnings: list[str] = []
    for panel in panels:
        panel_nodes = [node for node in nodes if node["panel_id"] == panel["id"]]
        if not panel_nodes:
            continue
        node_ids = {node["id"] for node in panel_nodes}
        panel_edges = [
            edge for edge in edges if edge["from"] in node_ids and edge["to"] in node_ids
        ]
        positions = _run_elk_panel(panel_nodes, panel_edges, plan.reading_direction)
        if positions is None or not _positions_fit_panel(positions, panel_nodes, panel):
            used_elk = False
            _fallback_grid_panel(panel_nodes, panel)
        else:
            graph_width = max(
                float(positions[node["id"]]["x"]) + float(node["width"])
                for node in panel_nodes
            )
            graph_height = max(
                float(positions[node["id"]]["y"]) + float(node["height"])
                for node in panel_nodes
            )
            content_top = float(panel["y"]) + 58
            content_height = float(panel["height"]) - 76
            origin_x = float(panel["x"]) + max(34, (float(panel["width"]) - graph_width) / 2)
            origin_y = content_top + max(14, (content_height - graph_height) / 2)
            for node in panel_nodes:
                placed = positions[node["id"]]
                node["x"] = round(origin_x + float(placed["x"]) + float(node["width"]) / 2, 1)
                node["y"] = round(origin_y + float(placed["y"]) + float(node["height"]) / 2, 1)
    if not used_elk:
        warnings.append("ELK unavailable or layout exceeded panel bounds; used deterministic fallback grid")
    node_map = {node["id"]: node for node in nodes}
    for edge in edges:
        source = node_map.get(edge["from"])
        target = node_map.get(edge["to"])
        edge["waypoints"] = _orthogonal_waypoints(source, target) if source and target else []
        edge["source_port"] = "EAST"
        edge["target_port"] = "WEST"
    return ("elkjs" if used_elk else "fallback_grid"), warnings


def _run_elk_panel(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    direction: str,
) -> dict[str, dict[str, float]] | None:
    node_binary = shutil.which("node")
    script = Path(__file__).resolve().parents[2] / "scripts" / "elk_layout.mjs"
    elk_module = Path(__file__).resolve().parents[2] / "node_modules" / "elkjs"
    if not node_binary or not script.is_file() or not elk_module.exists():
        return None
    payload = {
        "direction": direction,
        "nodes": [
            {"id": node["id"], "width": node["width"], "height": node["height"]}
            for node in nodes
        ],
        "edges": [
            {"id": edge["id"], "source": edge["from"], "target": edge["to"]}
            for edge in edges
        ],
    }
    try:
        completed = subprocess.run(
            [node_binary, str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return None
        result = json.loads(completed.stdout)
        return {
            str(node["id"]): {"x": float(node["x"]), "y": float(node["y"])}
            for node in result.get("nodes", [])
        }
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _positions_fit_panel(
    positions: dict[str, dict[str, float]],
    nodes: list[dict[str, Any]],
    panel: dict[str, Any],
) -> bool:
    if set(positions) != {node["id"] for node in nodes}:
        return False
    max_right = max(positions[node["id"]]["x"] + float(node["width"]) for node in nodes)
    max_bottom = max(positions[node["id"]]["y"] + float(node["height"]) for node in nodes)
    return max_right <= float(panel["width"]) - 68 and max_bottom <= float(panel["height"]) - 100


def _fallback_grid_panel(nodes: list[dict[str, Any]], panel: dict[str, Any]) -> None:
    available_width = float(panel["width"]) - 68
    available_height = float(panel["height"]) - 100
    max_width = max(float(node["width"]) for node in nodes)
    max_height = max(float(node["height"]) for node in nodes)
    columns = max(1, min(len(nodes), int((available_width + 28) // (max_width + 28))))
    rows = math.ceil(len(nodes) / columns)
    horizontal_step = available_width / columns
    vertical_step = available_height / rows
    for index, node in enumerate(nodes):
        column = index % columns
        row = index // columns
        node["x"] = round(float(panel["x"]) + 34 + horizontal_step * (column + 0.5), 1)
        node["y"] = round(float(panel["y"]) + 72 + vertical_step * (row + 0.5), 1)


def _orthogonal_waypoints(
    source: dict[str, Any], target: dict[str, Any]
) -> list[dict[str, float]]:
    sx, sy, tx, ty = _edge_endpoints(source, target)
    if abs(tx - sx) >= abs(ty - sy):
        middle = (sx + tx) / 2
        points = [(sx, sy), (middle, sy), (middle, ty), (tx, ty)]
    else:
        middle = (sy + ty) / 2
        points = [(sx, sy), (sx, middle), (tx, middle), (tx, ty)]
    compact: list[dict[str, float]] = []
    for x, y in points:
        point = {"x": round(x, 1), "y": round(y, 1)}
        if not compact or point != compact[-1]:
            compact.append(point)
    return compact


def _polyline_midpoint(points: list[dict[str, float]]) -> dict[str, float]:
    segments: list[tuple[dict[str, float], dict[str, float], float]] = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        length = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
        segments.append((start, end, length))
        total += length
    target = total / 2
    traversed = 0.0
    for start, end, length in segments:
        if traversed + length >= target and length > 0:
            ratio = (target - traversed) / length
            return {
                "x": float(start["x"]) + (float(end["x"]) - float(start["x"])) * ratio,
                "y": float(start["y"]) + (float(end["y"]) - float(start["y"])) * ratio,
            }
        traversed += length
    return points[-1] if points else {"x": 0.0, "y": 0.0}


def _arrowhead_points(points: list[dict[str, float]]) -> str:
    if len(points) < 2:
        return "0,0 0,0 0,0"
    start, end = points[-2], points[-1]
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    base_x, base_y = float(end["x"]) - ux * 14, float(end["y"]) - uy * 14
    px, py = -uy * 7, ux * 7
    return (
        f'{float(end["x"]):.1f},{float(end["y"]):.1f} '
        f"{base_x + px:.1f},{base_y + py:.1f} {base_x - px:.1f},{base_y - py:.1f}"
    )


def _resolve_font_path(family: str, *, require_cjk: bool) -> Path | None:
    preferred = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    if not require_cjk:
        preferred.extend(
            [
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]
        )
    for path in preferred:
        if path.is_file():
            return path
    compact = re.sub(r"[^a-z0-9]", "", family.lower())
    for root in (Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path("/usr/share/fonts")):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and compact in re.sub(r"[^a-z0-9]", "", path.stem.lower()):
                return path
    return None


def _measure_wrapped_text(
    text: str,
    font_path: Path | None,
    font_size: int,
    *,
    max_text_width: int,
) -> tuple[list[str], float, float]:
    font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

    def width(value: str) -> float:
        bounds = font.getbbox(value or " ")
        return float(bounds[2] - bounds[0])

    tokens = text.split(" ") if " " in text else list(text)
    separator = " " if " " in text else ""
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{separator if current else ''}{token}"
        if current and width(candidate) > max_text_width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    lines = lines or [text]
    line_height = max(font_size * 1.25, float(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.25)
    return lines, max(width(line) for line in lines), line_height * len(lines)


def _layout_panels(contract: FigureContract, style: VisualStyleSpec) -> list[dict[str, Any]]:
    width = contract.target_width_px
    height = contract.target_height_px
    contract_panels = contract.panels[:4] or []
    if not contract_panels:
        return []
    count = len(contract_panels)
    outer_x, outer_y, gap = 35, 45, 24
    columns = 1 if count == 1 else 2
    rows = (count + columns - 1) // columns
    panel_width = (width - 2 * outer_x - gap * (columns - 1)) / columns
    panel_height = (height - 2 * outer_y - gap * (rows - 1)) / rows
    return [
        {
            "id": panel.panel_id,
            "title": panel.title,
            "x": round(outer_x + (index % columns) * (panel_width + gap), 1),
            "y": round(outer_y + (index // columns) * (panel_height + gap), 1),
            "width": round(panel_width, 1),
            "height": round(panel_height, 1),
            "fill": "#FAFBFC",
            "stroke": style.palette[index % len(style.palette)],
        }
        for index, panel in enumerate(contract_panels)
    ]


def svg_to_png_pdf(svg_bytes: bytes) -> tuple[bytes, bytes]:
    import fitz

    svg_document = fitz.open(stream=svg_bytes, filetype="svg")
    try:
        page = svg_document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        png_bytes = pixmap.tobytes("png")
        pdf_bytes = svg_document.convert_to_pdf()
    finally:
        svg_document.close()
    return png_bytes, pdf_bytes


def validate_diagram(
    contract: FigureContract,
    style: VisualStyleSpec,
    render_spec: dict[str, Any],
    svg_bytes: bytes,
    editable_bytes: bytes,
    *,
    renderer_issues: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = [*(renderer_issues or []), *render_spec.get("layout_warnings", [])]
    nodes = render_spec.get("nodes", [])
    edges = render_spec.get("edges", [])
    node_ids = {str(node.get("id", "")) for node in nodes}
    expected_ids = {entity.entity_id for entity in contract.entities}
    if node_ids != expected_ids:
        errors.append("rendered entities do not exactly match FigureContract entities")
    disconnected = [
        str(edge.get("id", ""))
        for edge in edges
        if str(edge.get("from", "")) not in node_ids or str(edge.get("to", "")) not in node_ids
    ]
    if disconnected:
        errors.append(f"disconnected edges: {disconnected}")

    rendered_relations = {
        (
            str(edge.get("from", "")),
            str(edge.get("to", "")),
            str(edge.get("label", "")),
            str(edge.get("semantic", "flow")),
        )
        for edge in edges
    }
    expected_relations = {
        (relation.source, relation.target, relation.label, relation.relation_type)
        for relation in contract.relations
    }
    if rendered_relations != expected_relations:
        errors.append("rendered arrow directions do not exactly match FigureContract relations")
    crossing_edges = [
        str(edge.get("id", ""))
        for edge in edges
        if _edge_crosses_non_target_node(edge, nodes)
    ]
    if crossing_edges:
        errors.append(f"edges cross unrelated nodes: {crossing_edges}")

    allowed = set(contract.label_allowlist)
    rendered_labels = {str(node.get("label", "")) for node in nodes}
    rendered_labels.update(str(edge.get("label", "")) for edge in edges if edge.get("label"))
    rendered_labels.update(
        str(panel.get("title", "")) for panel in render_spec.get("panels", []) if panel.get("title")
    )
    unexpected_labels = sorted(label for label in rendered_labels if label and label not in allowed)
    if unexpected_labels:
        errors.append(f"labels outside allowlist: {unexpected_labels}")

    canvas = render_spec.get("canvas", {})
    width = float(canvas.get("width", 0))
    height = float(canvas.get("height", 0))
    for node in nodes:
        x, y = float(node["x"]), float(node["y"])
        half_w, half_h = float(node["width"]) / 2, float(node["height"]) / 2
        if x - half_w < 0 or y - half_h < 0 or x + half_w > width or y + half_h > height:
            errors.append(f"node {node['id']} exceeds canvas bounds")
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if _nodes_overlap(left, right):
                errors.append(f"nodes {left['id']} and {right['id']} overlap")
    for node in nodes:
        lines = node.get("text_lines") or []
        compact_label = re.sub(r"\s+", "", str(node.get("label", "")))
        compact_rendered = re.sub(r"\s+", "", "".join(lines))
        if (
            compact_rendered != compact_label
            or float(node.get("text_width", 0)) > float(node["width"]) - 100
            or float(node.get("text_height", 0)) > float(node["height"]) - 30
        ):
            errors.append(f"label text exceeds node box: {node['id']}")
    panels = render_spec.get("panels", [])
    if panels:
        panels_by_id = {str(panel["id"]): panel for panel in panels}
        for node in nodes:
            panel = panels_by_id.get(str(node.get("panel_id", "")))
            if panel is None:
                errors.append(f"node {node['id']} has no valid panel assignment")
                continue
            panel_left = float(panel["x"])
            panel_top = float(panel["y"])
            panel_right = panel_left + float(panel["width"])
            panel_bottom = panel_top + float(panel["height"])
            x, y = float(node["x"]), float(node["y"])
            half_w, half_h = float(node["width"]) / 2, float(node["height"]) / 2
            if (
                x - half_w < panel_left
                or y - half_h < panel_top
                or x + half_w > panel_right
                or y + half_h > panel_bottom
            ):
                errors.append(f"node {node['id']} exceeds panel bounds")

    if not svg_bytes.lstrip().startswith(b"<svg") and b"<svg" not in svg_bytes[:500]:
        errors.append("SVG output is not parseable")
    editable_parseable = True
    try:
        editable_root = ET.fromstring(editable_bytes)
    except ET.ParseError:
        editable_parseable = False
        editable_root = None
        errors.append("editable source XML is not parseable")
    renderer, _ = normalize_figure_renderer(contract.renderer)
    native_edges = True
    if renderer == "drawio" and editable_root is not None:
        native_edges = all(
            cell.get("source") and cell.get("target")
            for cell in editable_root.findall(".//mxCell[@edge='1']")
        )
        if not native_edges:
            errors.append("Draw.io contains an edge without native source/target connectors")
    stable_svg_ids = all(
        f'id="{entity.entity_id}"'.encode("utf-8") in svg_bytes for entity in contract.entities
    ) and all(f'id="edge_{index + 1}"'.encode("utf-8") in svg_bytes for index, _ in enumerate(contract.relations))
    if not stable_svg_ids:
        errors.append("SVG critical objects are missing stable IDs")
    font_resolved = bool(render_spec.get("style", {}).get("font_resolved"))
    if not font_resolved:
        message = f"font unavailable: {style.cjk_font_family if contract.language == 'Chinese' else style.font_family}"
        if contract.language == "Chinese":
            errors.append(message)
        else:
            warnings.append(message)
    low_contrast_colors: list[str] = []
    for color in style.palette:
        if _contrast_ratio(color, "#FFFFFF") < 2.0:
            low_contrast_colors.append(color)
            warnings.append(f"low contrast accent on white: {color}")

    status = "pass" if not errors else "fail"
    return {
        "schema_version": "1.0",
        "hard_status": status,
        "errors": errors,
        "warnings": _deduplicate(warnings),
        "checks": {
            "entity_coverage": node_ids == expected_ids,
            "relation_direction": rendered_relations == expected_relations,
            "label_allowlist": not unexpected_labels,
            "canvas_bounds": not any("bounds" in error for error in errors),
            "overlap": not any("overlap" in error for error in errors),
            "text_overflow": not any("text exceeds" in error for error in errors),
            "edge_connectivity": not disconnected,
            "edge_clearance": not crossing_edges,
            "editable_source_parseable": editable_parseable,
            "native_drawio_edges": native_edges,
            "stable_svg_ids": stable_svg_ids,
            "font_resolved": font_resolved,
            "contrast": not low_contrast_colors,
            "grayscale_redundancy": all(node.get("label") and node.get("icon") for node in nodes),
        },
        "visual_review": "advisory_not_run",
        "publication_status": "needs_human_visual_review",
    }


def _render_drawio_xml(render_spec: dict[str, Any], style: VisualStyleSpec) -> str:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "24.7.17"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "figure-01", "name": "Figure 01"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {"dx": "1600", "dy": "960", "grid": "1", "gridSize": "10", "page": "1"},
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    for panel in render_spec.get("panels", []):
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"panel_{panel['id']}",
                "value": panel["title"],
                "style": (
                    "rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
                    f"spacingTop=12;spacingLeft=16;fillColor={panel['fill']};strokeColor={panel['stroke']};"
                    "fontStyle=1;fontSize=20;dashed=1;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(panel["x"]),
                "y": str(panel["y"]),
                "width": str(panel["width"]),
                "height": str(panel["height"]),
                "as": "geometry",
            },
        )

    for node in render_spec.get("nodes", []):
        left = float(node["x"]) - float(node["width"]) / 2
        top = float(node["y"]) - float(node["height"]) / 2
        background = ET.SubElement(
            root,
            "mxCell",
            {
                "id": node["id"],
                "value": "",
                "style": (
                    "rounded=1;whiteSpace=wrap;html=1;"
                    f"arcSize={style.corner_radius};fillColor={node['fill']};strokeColor={node['stroke']};"
                    "strokeWidth=2;shadow=0;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            background,
            "mxGeometry",
            {
                "x": f"{left:.1f}",
                "y": f"{top:.1f}",
                "width": str(node["width"]),
                "height": str(node["height"]),
                "as": "geometry",
            },
        )
        icon = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"{node['id']}_icon",
                "value": "",
                "style": (
                    (
                        f"shape=image;imageAspect=0;aspect=fixed;image={node['icon_asset']['data_uri']};"
                        if node.get("icon_asset")
                        else f"{_drawio_icon_style(node['icon'])};fillColor={node['stroke']};strokeColor={node['stroke']};"
                    )
                    + "whiteSpace=wrap;html=1;align=center;verticalAlign=middle;fontColor=#FFFFFF;"
                ),
                "vertex": "1",
                "parent": node["id"],
            },
        )
        ET.SubElement(
            icon,
            "mxGeometry",
            {
                "x": "18",
                "y": "18",
                "width": "48",
                "height": "48",
                "as": "geometry",
            },
        )
        label = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"{node['id']}_label",
                "value": node["label"],
                "style": (
                    "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
                    "whiteSpace=wrap;rounded=0;fontSize=20;fontStyle=1;fontColor=#20242A;"
                ),
                "vertex": "1",
                "parent": node["id"],
            },
        )
        ET.SubElement(
            label,
            "mxGeometry",
            {
                "x": "72",
                "y": "18",
                "width": f"{float(node['width']) - 90:.1f}",
                "height": f"{float(node['height']) - 36:.1f}",
                "as": "geometry",
            },
        )

    for edge in render_spec.get("edges", []):
        dashed = "dashed=1;dashPattern=8 5;" if edge.get("style") != "solid" else ""
        end_fill = "0" if edge.get("semantic") == "association" else "1"
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": edge["id"],
                "value": edge.get("label", ""),
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
                    f"strokeColor={edge['color']};strokeWidth={edge['thickness']};{dashed}endArrow=block;endFill={end_fill};"
                    "fontSize=16;labelBackgroundColor=#FFFFFF;"
                ),
                "edge": "1",
                "parent": "1",
                "source": edge["from"],
                "target": edge["to"],
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        interior_points = list(edge.get("waypoints", []))[1:-1]
        if interior_points:
            points = ET.SubElement(geometry, "Array", {"as": "points"})
            for point in interior_points:
                ET.SubElement(
                    points,
                    "mxPoint",
                    {"x": str(point["x"]), "y": str(point["y"])},
                )

    return ET.tostring(mxfile, encoding="unicode", xml_declaration=True)


def _render_academic_svg(render_spec: dict[str, Any], style: VisualStyleSpec) -> str:
    canvas = render_spec["canvas"]
    width, height = canvas["width"], canvas["height"]
    family = escape(render_spec["style"]["font_family"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{family}, Arial, sans-serif">',
        "<defs>",
        '<marker id="arrow-flow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#333333"/></marker>',
        '<marker id="arrow-inhibition" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#C8453C"/></marker>',
        '<marker id="arrow-feedback" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#7653A6"/></marker>',
    ]
    for panel in render_spec.get("panels", []):
        parts.append(
            f'<clipPath id="clip_{escape(panel["id"])}"><rect x="{panel["x"]}" y="{panel["y"]}" width="{panel["width"]}" height="{panel["height"]}" rx="18"/></clipPath>'
        )
    parts.extend(["</defs>", f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>'])
    for panel in render_spec.get("panels", []):
        parts.append(f'<g id="panel-{escape(panel["id"])}" data-kind="panel">')
        parts.append(
            f'<rect x="{panel["x"]}" y="{panel["y"]}" width="{panel["width"]}" height="{panel["height"]}" rx="18" fill="{panel["fill"]}" stroke="{panel["stroke"]}" stroke-width="2" stroke-dasharray="8 5"/>'
        )
        if style.hatch_backgrounds:
            panel_clip_id = f"clip_{panel['id']}"
            parts.append(f'<g clip-path="url(#{escape(panel_clip_id)})">')
            panel_x = float(panel["x"])
            panel_y = float(panel["y"])
            panel_width = float(panel["width"])
            panel_height = float(panel["height"])
            for offset in range(-int(panel_height), int(panel_width), 24):
                parts.append(
                    f'<line x1="{panel_x + offset:.1f}" y1="{panel_y + panel_height:.1f}" x2="{panel_x + offset + panel_height:.1f}" y2="{panel_y:.1f}" stroke="#E4E8ED" stroke-width="2" opacity="0.72"/>'
                )
            parts.append("</g>")
        parts.append(
            f'<rect x="{panel["x"]}" y="{panel["y"]}" width="190" height="44" rx="10" fill="{panel["stroke"]}"/>'
        )
        parts.append(
            f'<text id="panel-{escape(panel["id"])}-title" x="{panel["x"] + 16}" y="{panel["y"] + 30}" font-size="22" font-weight="700" fill="#FFFFFF">{escape(panel["title"])}</text>'
        )
        parts.append("</g>")

    node_map = {node["id"]: node for node in render_spec.get("nodes", [])}
    for edge in render_spec.get("edges", []):
        points = edge.get("waypoints") or [
            {"x": value, "y": other}
            for value, other in zip(_edge_endpoints(node_map[edge["from"]], node_map[edge["to"]])[::2], _edge_endpoints(node_map[edge["from"]], node_map[edge["to"]])[1::2])
        ]
        path_data = " ".join(
            ("M" if index == 0 else "L") + f" {point['x']:.1f} {point['y']:.1f}"
            for index, point in enumerate(points)
        )
        marker = (
            ""
            if edge.get("semantic") == "association"
            else f' marker-end="url(#arrow-{edge.get("semantic", "flow")})"'
        )
        parts.append(
            f'<g id="{escape(edge["id"])}" data-kind="edge" data-source="{escape(edge["from"])}" data-target="{escape(edge["to"])}">'
        )
        parts.append(
            f'<path id="{escape(edge["id"])}-path" d="{path_data}" fill="none" stroke="{edge["color"]}" stroke-width="{edge["thickness"]}" stroke-linejoin="round"{_svg_dash(edge.get("style", "solid"))}{marker}/>'
        )
        if edge.get("semantic") != "association":
            parts.append(
                f'<polygon id="{escape(edge["id"])}-arrowhead" points="{_arrowhead_points(points)}" fill="{edge["color"]}"/>'
            )
        if edge.get("label"):
            middle_point = _polyline_midpoint(points)
            parts.append(
                f'<rect x="{middle_point["x"] - 54:.1f}" y="{middle_point["y"] - 28:.1f}" width="108" height="30" rx="8" fill="#FFFFFF"/>'
            )
            parts.append(
                f'<text id="{escape(edge["id"])}-label" x="{middle_point["x"]:.1f}" y="{middle_point["y"] - 7:.1f}" text-anchor="middle" font-size="15" fill="#333333">{escape(edge["label"])}</text>'
            )
        parts.append("</g>")

    for node in render_spec.get("nodes", []):
        left = float(node["x"]) - float(node["width"]) / 2
        top = float(node["y"]) - float(node["height"]) / 2
        parts.append(f'<g id="{escape(node["id"])}" data-kind="node" data-panel="{escape(node["panel_id"])}">')
        parts.append(
            f'<rect id="{escape(node["id"])}-shape" x="{left:.1f}" y="{top:.1f}" width="{node["width"]}" height="{node["height"]}" rx="{style.corner_radius}" fill="{node["fill"]}" stroke="{node["stroke"]}" stroke-width="2"/>'
        )
        icon_x, icon_y = left + 52, top + float(node["height"]) / 2
        parts.append(f'<g id="{escape(node["id"])}-icon" data-kind="icon">')
        if node.get("icon_asset"):
            parts.append(
                f'<circle cx="{icon_x:.1f}" cy="{icon_y:.1f}" r="31" fill="#FFFFFF" stroke="{node["stroke"]}" stroke-width="2"/>'
            )
            parts.append(
                f'<g transform="translate({icon_x - 22:.1f} {icon_y - 22:.1f}) scale(1.8333)">{node["icon_asset"]["inner_svg"]}</g>'
            )
        else:
            parts.extend(_svg_icon(node["icon"], icon_x, icon_y, node["stroke"]))
        parts.append("</g>")
        lines = node.get("text_lines") or [node["label"]]
        start_y = float(node["y"]) - (len(lines) - 1) * 14
        label_x = left + 82 + (float(node["width"]) - 96) / 2
        parts.append(f'<g id="{escape(node["id"])}-label" data-kind="text"><text x="{label_x:.1f}" y="{start_y:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="22" font-weight="700" fill="#20242A">')
        for line_index, line in enumerate(lines):
            dy = "0" if line_index == 0 else "28"
            parts.append(f'<tspan x="{label_x:.1f}" dy="{dy}">{escape(line)}</tspan>')
        parts.append("</text></g></g>")
    parts.append("</svg>")
    return "".join(parts)


def _svg_icon(icon: str, x: float, y: float, color: str) -> list[str]:
    base = [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="30" fill="{color}"/>']
    if icon == "database":
        base.extend(
            [
                f'<ellipse cx="{x:.1f}" cy="{y - 10:.1f}" rx="15" ry="6" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
                f'<path d="M {x - 15:.1f} {y - 10:.1f} V {y + 12:.1f} C {x - 15:.1f} {y + 20:.1f}, {x + 15:.1f} {y + 20:.1f}, {x + 15:.1f} {y + 12:.1f} V {y - 10:.1f}" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
            ]
        )
    elif icon == "robot":
        base.extend(
            [
                f'<rect x="{x - 16:.1f}" y="{y - 13:.1f}" width="32" height="27" rx="6" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
                f'<circle cx="{x - 7:.1f}" cy="{y:.1f}" r="3" fill="#FFFFFF"/><circle cx="{x + 7:.1f}" cy="{y:.1f}" r="3" fill="#FFFFFF"/>',
                f'<path d="M {x:.1f} {y - 13:.1f} V {y - 22:.1f}" stroke="#FFFFFF" stroke-width="3"/><circle cx="{x:.1f}" cy="{y - 24:.1f}" r="3" fill="#FFFFFF"/>',
            ]
        )
    elif icon == "person":
        base.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y - 9:.1f}" r="8" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
                f'<path d="M {x - 15:.1f} {y + 16:.1f} Q {x:.1f} {y:.1f} {x + 15:.1f} {y + 16:.1f}" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
            ]
        )
    elif icon == "document":
        base.append(
            f'<path d="M {x - 13:.1f} {y - 18:.1f} H {x + 7:.1f} L {x + 15:.1f} {y - 10:.1f} V {y + 18:.1f} H {x - 13:.1f} Z M {x - 7:.1f} {y - 2:.1f} H {x + 9:.1f} M {x - 7:.1f} {y + 7:.1f} H {x + 9:.1f}" fill="none" stroke="#FFFFFF" stroke-width="3"/>'
        )
    elif icon == "security":
        base.append(
            f'<path d="M {x:.1f} {y - 19:.1f} L {x + 16:.1f} {y - 12:.1f} V {y:.1f} Q {x + 14:.1f} {y + 15:.1f} {x:.1f} {y + 21:.1f} Q {x - 14:.1f} {y + 15:.1f} {x - 16:.1f} {y:.1f} V {y - 12:.1f} Z" fill="none" stroke="#FFFFFF" stroke-width="3"/>'
        )
    elif icon == "attack":
        base.extend(
            [
                f'<path d="M {x - 5:.1f} {y - 20:.1f} L {x + 13:.1f} {y - 2:.1f} L {x + 3:.1f} {y + 1:.1f} L {x + 8:.1f} {y + 20:.1f} L {x - 13:.1f} {y:.1f} L {x - 3:.1f} {y - 3:.1f} Z" fill="#FFFFFF"/>',
            ]
        )
    elif icon == "environment":
        base.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="17" fill="none" stroke="#FFFFFF" stroke-width="3"/>',
                f'<path d="M {x - 16:.1f} {y:.1f} H {x + 16:.1f} M {x:.1f} {y - 17:.1f} C {x - 8:.1f} {y - 6:.1f}, {x - 8:.1f} {y + 6:.1f}, {x:.1f} {y + 17:.1f} M {x:.1f} {y - 17:.1f} C {x + 8:.1f} {y - 6:.1f}, {x + 8:.1f} {y + 6:.1f}, {x:.1f} {y + 17:.1f}" fill="none" stroke="#FFFFFF" stroke-width="2"/>',
            ]
        )
    elif icon == "experiment":
        base.append(
            f'<path d="M {x - 7:.1f} {y - 19:.1f} H {x + 7:.1f} M {x - 3:.1f} {y - 19:.1f} V {y - 4:.1f} L {x - 15:.1f} {y + 17:.1f} H {x + 15:.1f} L {x + 3:.1f} {y - 4:.1f} V {y - 19:.1f} M {x - 8:.1f} {y + 7:.1f} H {x + 8:.1f}" fill="none" stroke="#FFFFFF" stroke-width="3"/>'
        )
    else:
        base.append(
            f'<path d="M {x - 14:.1f} {y:.1f} L {x - 3:.1f} {y + 12:.1f} L {x + 16:.1f} {y - 13:.1f}" fill="none" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return base


def _icon_name(index: int, role: str, label: str) -> str:
    normalized = f"{role} {label}".lower()
    if any(term in normalized for term in ("attack", "threat", "poison", "攻击", "威胁", "投毒")):
        return "attack"
    if any(term in normalized for term in ("security", "safe", "defense", "privacy", "安全", "防御", "隐私")):
        return "security"
    if any(term in normalized for term in ("user", "person", "human", "用户", "人员")):
        return "person"
    if any(term in normalized for term in ("document", "paper", "file", "文档", "论文", "文件")):
        return "document"
    if any(term in normalized for term in ("environment", "world", "生态", "环境", "世界")):
        return "environment"
    if any(term in normalized for term in ("experiment", "lab", "实验", "试验")):
        return "experiment"
    if "data" in normalized or "数据" in normalized or index == 0:
        return "database"
    if "agent" in normalized or "model" in normalized or index == 1:
        return "robot"
    return "result"


def _load_icon_asset(icon: str) -> dict[str, Any] | None:
    aliases = {
        "database": "database",
        "robot": "cpu",
        "person": "user",
        "document": "file",
        "security": "shield",
        "attack": "bolt",
        "environment": "world",
        "experiment": "flask",
        "result": "circle-check",
    }
    name = aliases.get(icon, "")
    path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "drawio-figure"
        / "assets"
        / "iconify"
        / "tabler"
        / f"{name}.svg"
    )
    if not name or not path.is_file():
        return None
    source_bytes = path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    inner_svg = re.sub(r"^\s*<svg[^>]*>|</svg>\s*$", "", source_text, flags=re.S)
    return {
        "collection": "tabler",
        "name": name,
        "data_uri": f"data:image/svg+xml,{quote(source_text, safe='')}",
        "inner_svg": inner_svg,
        "provenance": {
            "asset": f"tabler:{name}",
            "source": "Iconify-compatible local cache",
            "license": "MIT",
            "upstream": "https://github.com/tabler/tabler-icons",
        },
    }


def _drawio_icon_style(icon: str) -> str:
    return {
        "database": "shape=cylinder3;size=8",
        "robot": "rounded=1;arcSize=25",
        "person": "shape=actor",
        "document": "shape=document;boundedLbl=1",
        "security": "shape=hexagon;perimeter=hexagonPerimeter2;fixedSize=1",
        "attack": "triangle;direction=east",
        "environment": "ellipse",
        "experiment": "shape=parallelogram;perimeter=parallelogramPerimeter",
        "result": "ellipse",
    }.get(icon, "ellipse")


def _edge_color(relation_type: str) -> str:
    return {
        "flow": "#333333",
        "inhibition": "#C8453C",
        "feedback": "#7653A6",
        "association": "#5D6B78",
    }.get(relation_type, "#333333")


def _edge_style(relation_type: str) -> str:
    return "solid" if relation_type == "flow" else ("dotted" if relation_type == "association" else "dashed")


def _svg_dash(style: str) -> str:
    if style == "dashed":
        return ' stroke-dasharray="10 7"'
    if style == "dotted":
        return ' stroke-dasharray="3 7" stroke-linecap="round"'
    return ""


def _wrap_label(label: str, max_units: int) -> list[str]:
    if len(label) <= max_units:
        return [label]
    if " " in label:
        words = label.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_units:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines[:3]
    return [label[index : index + max_units] for index in range(0, len(label), max_units)][:3]


def _label_wrap_units(node: dict[str, Any]) -> int:
    return max(3, int((float(node["width"]) - 98) / 11))


def _edge_endpoints(
    source: dict[str, Any],
    target: dict[str, Any],
) -> tuple[float, float, float, float]:
    source_x, source_y = float(source["x"]), float(source["y"])
    target_x, target_y = float(target["x"]), float(target["y"])
    dx, dy = target_x - source_x, target_y - source_y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return source_x, source_y, target_x, target_y

    def boundary_scale(node: dict[str, Any]) -> float:
        candidates: list[float] = []
        if abs(dx) >= 1e-6:
            candidates.append(float(node["width"]) / 2 / abs(dx))
        if abs(dy) >= 1e-6:
            candidates.append(float(node["height"]) / 2 / abs(dy))
        return min(candidates)

    source_scale = boundary_scale(source)
    target_scale = boundary_scale(target)
    return (
        source_x + dx * source_scale,
        source_y + dy * source_scale,
        target_x - dx * target_scale,
        target_y - dy * target_scale,
    )


def _nodes_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        abs(float(left["x"]) - float(right["x"]))
        < (float(left["width"]) + float(right["width"])) / 2
        and abs(float(left["y"]) - float(right["y"]))
        < (float(left["height"]) + float(right["height"])) / 2
    )


def _edge_crosses_non_target_node(
    edge: dict[str, Any], nodes: list[dict[str, Any]]
) -> bool:
    points = list(edge.get("waypoints", []))
    if len(points) < 2:
        return False
    ignored = {str(edge.get("from", "")), str(edge.get("to", ""))}
    for node in nodes:
        if str(node.get("id", "")) in ignored:
            continue
        left = float(node["x"]) - float(node["width"]) / 2
        right = float(node["x"]) + float(node["width"]) / 2
        top = float(node["y"]) - float(node["height"]) / 2
        bottom = float(node["y"]) + float(node["height"]) / 2
        for start, end in zip(points, points[1:]):
            x1, y1 = float(start["x"]), float(start["y"])
            x2, y2 = float(end["x"]), float(end["y"])
            if abs(x1 - x2) < 1e-6 and left < x1 < right and max(y1, y2) > top and min(y1, y2) < bottom:
                return True
            if abs(y1 - y2) < 1e-6 and top < y1 < bottom and max(x1, x2) > left and min(x1, x2) < right:
                return True
    return False


def _font_available(family: str) -> bool:
    fc_list = shutil.which("fc-list")
    if fc_list:
        try:
            result = subprocess.run(
                [fc_list, ":", "family"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if family.lower() in result.stdout.lower():
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    compact = re.sub(r"[^a-z0-9]", "", family.lower())
    for root in (Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path("/usr/share/fonts")):
        if root.exists() and any(compact in re.sub(r"[^a-z0-9]", "", path.stem.lower()) for path in root.rglob("*")):
            return True
    return False


def _contrast_ratio(first: str, second: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _lighten(color: str, factor: float = 0.86) -> str:
    channels = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    light = [min(255, round(value + (255 - value) * factor)) for value in channels]
    return "#" + "".join(f"{value:02X}" for value in light)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _deduplicate_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        key = (str(value.get("source", "")), str(value.get("asset", "")))
        unique[key] = value
    return list(unique.values())
