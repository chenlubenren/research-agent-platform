from __future__ import annotations

import asyncio
import io
import json
import runpy
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl import Workbook
from PIL import Image

from research_agent_platform import agent as agent_module
from research_agent_platform import diagram_pipeline as diagram_module
from research_agent_platform.diagram_pipeline import build_diagram_render_spec, render_diagram, validate_diagram
from research_agent_platform.edit_banana import inspect_edit_banana_drawio
from research_agent_platform.figure_contracts import (
    build_figure_contract,
    build_visual_style_spec,
    resolve_figure_source_config,
)
from research_agent_platform.figure_pipeline import render_code_figure, validate_code_figure
from research_agent_platform.models import (
    ArtifactRecord,
    FigureContract,
    FigureDeliveryManifest,
    FigureSourceConfig,
    TaskRun,
    UploadBatchRecord,
)
from research_agent_platform.router.intent import explicit_route
from research_agent_platform.upstream import GeneratedImage


FIGURE_FIXTURES = Path(__file__).parent / "fixtures" / "figures"


def test_fig_routes_explicitly():
    route = explicit_route("/fig 根据数据生成图")

    assert route is not None
    assert route.command == "/fig"


def test_csv_category_and_value_generates_bar_chart(tmp_path: Path):
    source = tmp_path / "Content" / "uploads" / "results.csv"
    source.parent.mkdir(parents=True)
    source.write_text("method,accuracy\nbaseline,0.81\nours,0.92\n", encoding="utf-8")

    result = render_code_figure(tmp_path, objective="/fig 模型准确率")

    assert result.chart_type == "bar"
    assert result.source_ref == "Content/uploads/results.csv"
    assert result.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in result.svg_bytes[:500]


def test_two_numeric_columns_generate_scatter_chart(tmp_path: Path):
    source = tmp_path / "figures" / "uploads" / "measurements.tsv"
    source.parent.mkdir(parents=True)
    source.write_text("rainfall\trunoff\n10\t3\n20\t8\n30\t15\n", encoding="utf-8")

    result = render_code_figure(tmp_path)

    assert result.chart_type == "scatter"
    assert result.x_column == "rainfall"
    assert result.y_columns == ["runoff"]


def test_xlsx_time_series_generates_line_chart(tmp_path: Path):
    source = tmp_path / "paper" / "uploads" / "experiment.xlsx"
    source.parent.mkdir(parents=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "metrics"
    worksheet.append(["epoch", "loss", "accuracy"])
    worksheet.append([1, 0.8, 0.6])
    worksheet.append([2, 0.5, 0.76])
    worksheet.append([3, 0.3, 0.86])
    workbook.save(source)

    result = render_code_figure(tmp_path)

    assert result.chart_type == "line"
    assert result.sheet_name == "metrics"
    assert result.y_columns == ["loss", "accuracy"]


def test_grouped_bar_uses_error_column_without_plotting_it_as_series(tmp_path: Path):
    source = tmp_path / "figures" / "uploads" / "ablation.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes((FIGURE_FIXTURES / "grouped_metrics.csv").read_bytes())

    result = render_code_figure(tmp_path, objective="画带误差棒的分组柱状图")

    assert result.chart_type == "bar"
    assert result.y_columns == ["accuracy", "latency_ms"]
    assert result.pdf_bytes.startswith(b"%PDF")
    assert validate_code_figure(result)["checks"]["error_expression"] is True

    script = tmp_path / "figures" / "generated" / "FIGURE_01.py"
    script.parent.mkdir(parents=True)
    script.write_text(result.source_code, encoding="utf-8")
    runpy.run_path(str(script), run_name="__main__")
    assert (script.parent / "FIGURE_01.svg").exists()


def test_data_qa_rejects_requested_error_bars_without_uncertainty_column(tmp_path: Path):
    source = tmp_path / "Content" / "uploads" / "results.csv"
    source.parent.mkdir(parents=True)
    source.write_text("method,score\nA,0.7\nB,0.9\n", encoding="utf-8")

    result = render_code_figure(tmp_path, objective="画带误差棒的分组柱状图")
    qa = validate_code_figure(result)

    assert qa["hard_status"] == "fail"
    assert qa["checks"]["error_expression"] is False


def test_heatmap_box_violin_and_small_multiples_routes(tmp_path: Path):
    source = tmp_path / "Content" / "uploads" / "metrics.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "method,metric_a,metric_b,metric_c\nA,1,2,3\nB,2,3,4\nC,3,4,5\n",
        encoding="utf-8",
    )

    assert render_code_figure(tmp_path, objective="生成热力图").chart_type == "heatmap"
    assert render_code_figure(tmp_path, objective="生成箱线图").chart_type == "box"
    assert render_code_figure(tmp_path, objective="生成小提琴图").chart_type == "violin"
    assert render_code_figure(tmp_path, objective="生成多面板小多图").chart_type == "small_multiples"


def test_figure_workflow_uses_code_for_valid_uploaded_data(service, monkeypatch):
    session = service.store.create_session()
    source = Path(session.workspace_root, "Content", "uploads", "results.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("method,score\nA,0.7\nB,0.9\n", encoding="utf-8")
    session.upload_batches.append(
        UploadBatchRecord(relative_paths=["Content/uploads/results.csv"])
    )
    service.store.save_session(session)

    async def unexpected_generate_image(**kwargs):
        raise AssertionError("Image-2 must not render precise tabular data")

    monkeypatch.setattr(agent_module, "generate_image", unexpected_generate_image)
    result = asyncio.run(
        service.chat(
            session.session_id,
            "/fig 根据 Content/uploads/results.csv 数据画对比柱状图",
        )
    )

    assert result["status"] == "completed"
    manifest_path = Path(session.workspace_root, "figures", "generated", "FIGURE_DELIVERY.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "code"
    assert manifest["renderer"] == "matplotlib"
    assert manifest["authoritative_source"].endswith(".py")
    FigureDeliveryManifest.model_validate(manifest)
    task = service.get_task(result["task_id"])
    assert task is not None
    artifact_paths = [artifact.relative_path for artifact in task.artifacts]
    assert artifact_paths.count("figures/generated/FIGURE_QA.json") == 1
    assert artifact_paths.count("figures/generated/FIGURE_DELIVERY.json") == 1
    assert Path(session.workspace_root, "figures", "generated", "FIGURE_01.png").exists()
    assert Path(session.workspace_root, "figures", "generated", "FIGURE_01.pdf").exists()


def test_figure_workflow_defaults_to_editable_academic_svg_without_data(service, monkeypatch):
    async def unexpected_generate_image(**kwargs):
        raise AssertionError("A framework default must not flatten the figure through Image-2")

    monkeypatch.setattr(agent_module, "generate_image", unexpected_generate_image)
    result = asyncio.run(service.chat(None, "/fig ???????????"))

    assert result["status"] == "completed"
    task = service.get_task(result["task_id"])
    assert task is not None
    manifest = json.loads(
        Path(task.artifact_root, "figures", "generated", "FIGURE_DELIVERY.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["mode"] == "academic_svg"
    assert manifest["renderer"] == "academic_svg"
    assert manifest["authoritative_source"].endswith(".svg")
    assert manifest["editability"] == "publication_vector"


def test_csv_does_not_override_explicit_mechanism_route(service):
    session = service.store.create_session()
    source = Path(session.workspace_root, "Content", "uploads", "results.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("method,score\nA,0.7\nB,0.9\n", encoding="utf-8")
    session.upload_batches.append(UploadBatchRecord(relative_paths=["Content/uploads/results.csv"]))
    service.store.save_session(session)

    result = asyncio.run(service.chat(session.session_id, "/fig 画攻击、防御到结果的机制图，要有生动图标"))

    assert result["status"] == "completed"
    manifest = json.loads(
        Path(session.workspace_root, "figures", "generated", "FIGURE_DELIVERY.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["figure_kind"] == "mechanism_diagram"
    assert manifest["renderer"] == "academic_svg"
    assert manifest["authoritative_source"].endswith(".svg")


def test_mechanism_default_does_not_call_imagegen(service, monkeypatch):
    async def unexpected_imagegen(**kwargs):
        raise AssertionError("ImageGen must be opt-in for a custom illustration asset")

    monkeypatch.setattr(agent_module, "generate_image", unexpected_imagegen)
    result = asyncio.run(service.chat(None, "/fig 画攻击、过滤、防御的机制图，要有生动图标"))
    task = service.get_task(result["task_id"])
    assert task is not None
    manifest = json.loads(
        Path(task.artifact_root, "figures", "generated", "FIGURE_DELIVERY.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "completed"
    assert manifest["renderer"] == "academic_svg"
    assert manifest["assist_assets"] == []
    assert not any("ImageGen" in warning for warning in manifest["warnings"])
    assert Path(task.artifact_root, manifest["authoritative_source"]).is_file()


def test_reference_image_and_explicit_pipeline_take_distinct_routes(tmp_path: Path):
    reference = tmp_path / "figures" / "uploads" / "reference.png"
    reference.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(reference)
    reference_sources = resolve_figure_source_config(
        "复刻 figures/uploads/reference.png 的布局",
        tmp_path,
    )

    reproduction = build_figure_contract("复刻参考图布局", reference_sources)
    pipeline = build_figure_contract(
        "画 Input -> Encoder -> Output 的严格 pipeline",
        FigureSourceConfig(resolved_scope="workspace"),
    )

    assert reproduction.kind == "reference_reproduction"
    assert reproduction.renderer == "drawio"
    assert pipeline.kind == "framework_diagram"
    assert pipeline.renderer == "academic_svg"


def test_explicit_data_source_precedes_latest_upload(tmp_path: Path):
    latest = tmp_path / "Content" / "uploads" / "latest.csv"
    explicit = tmp_path / "figures" / "uploads" / "chosen.csv"
    latest.parent.mkdir(parents=True)
    explicit.parent.mkdir(parents=True)
    latest.write_text("name,value\nold,1\nnew,2\n", encoding="utf-8")
    explicit.write_text("name,value\nA,3\nB,4\n", encoding="utf-8")
    batches = [UploadBatchRecord(relative_paths=["Content/uploads/latest.csv"])]

    config = resolve_figure_source_config(
        "根据 figures/uploads/chosen.csv 数据画柱状图",
        tmp_path,
        batches,
    )
    result = render_code_figure(tmp_path, objective="柱状图", explicit_source_refs=config.data_refs)

    assert config.resolved_scope == "selected"
    assert result.source_ref == "figures/uploads/chosen.csv"
    contract = build_figure_contract("figures/uploads/chosen.csv", config)
    assert contract.renderer == "matplotlib"


def test_drawio_contains_native_labels_and_directed_edges():
    sources = FigureSourceConfig(resolved_scope="workspace")
    contract = build_figure_contract(
        "画输入、智能体、输出的机制图，需要拖拽改结构并让箭头跟随，使用 Draw.io",
        sources,
    )
    style = build_visual_style_spec(contract)

    rendered = render_diagram(contract, style)
    root = ET.fromstring(rendered.editable_bytes)
    cells = root.findall(".//mxCell")

    assert rendered.editable_extension == "drawio"
    assert rendered.qa["hard_status"] == "pass"
    assert rendered.qa["checks"]["canvas_bounds"] is True
    assert all("图标" not in entity.label for entity in contract.entities)
    assert any(cell.get("id", "").endswith("_label") for cell in cells)
    assert any(cell.get("edge") == "1" and cell.get("source") and cell.get("target") for cell in cells)
    first_node = contract.entities[0].entity_id
    assert root.find(f".//mxCell[@id='{first_node}_label']").get("parent") == first_node
    assert root.find(f".//mxCell[@id='{first_node}_icon']").get("parent") == first_node
    assert "shape=image" in root.find(f".//mxCell[@id='{first_node}_icon']").get("style", "")
    assert rendered.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered.pdf_bytes.startswith(b"%PDF")


def test_framework_uses_drawio_only_for_explicit_structural_editing():
    contract = build_figure_contract(
        "画 Input -> Method -> Output 框架图，需要拖拽移动节点且箭头跟随",
        FigureSourceConfig(resolved_scope="workspace"),
    )

    assert contract.kind == "framework_diagram"
    assert contract.renderer == "drawio"
    assert contract.editability == "structural"


def test_academic_svg_has_stable_editable_object_groups():
    contract = build_figure_contract(
        "画 Input -> Method -> Output 的投稿框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    )
    rendered = render_diagram(contract, build_visual_style_spec(contract))
    svg = ET.fromstring(rendered.svg_bytes)
    ids = {element.get("id") for element in svg.iter() if element.get("id")}

    assert rendered.renderer == "academic_svg"
    assert rendered.editable_extension == "svg"
    assert {entity.entity_id for entity in contract.entities}.issubset(ids)
    assert {f"edge_{index + 1}" for index, _ in enumerate(contract.relations)}.issubset(ids)
    assert any(element.tag.endswith("tspan") for element in svg.iter())


def test_long_chinese_text_is_measured_and_wrapped_before_layout():
    contract = build_figure_contract(
        "画 原始多模态科研证据输入 -> 经过人工复核与一致性校验的核心分析模块 -> 可追溯且允许继续编辑的研究结论输出 的框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    )
    spec = build_diagram_render_spec(contract, build_visual_style_spec(contract))

    assert any(len(node["text_lines"]) > 1 for node in spec["nodes"])
    assert all(node["text_width"] <= node["width"] - 100 for node in spec["nodes"])
    assert validate_diagram(
        contract,
        build_visual_style_spec(contract),
        spec,
        render_diagram(contract, build_visual_style_spec(contract)).svg_bytes,
        render_diagram(contract, build_visual_style_spec(contract)).editable_bytes,
    )["checks"]["text_overflow"] is True


def test_elk_unavailable_uses_deterministic_fallback_and_warning(monkeypatch):
    contract = build_figure_contract(
        "画 Input -> Method -> Output 的框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    )
    original_which = diagram_module.shutil.which
    monkeypatch.setattr(
        diagram_module.shutil,
        "which",
        lambda name: None if name == "node" else original_which(name),
    )

    first = build_diagram_render_spec(contract, build_visual_style_spec(contract))
    second = build_diagram_render_spec(contract, build_visual_style_spec(contract))

    assert first["layout_engine"] == "fallback_grid"
    assert first["nodes"] == second["nodes"]
    assert any("fallback grid" in warning for warning in first["layout_warnings"])


def test_iconify_assets_are_cached_locally_with_license_metadata():
    contract = build_figure_contract(
        "画 Input -> Method -> Output 的框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    )
    spec = build_diagram_render_spec(contract, build_visual_style_spec(contract))

    assert all(node.get("icon_asset") for node in spec["nodes"])
    assert all(source["license"] == "MIT" for source in spec["asset_sources"])


def test_missing_cjk_font_is_a_hard_qa_failure(monkeypatch):
    contract = build_figure_contract(
        "画 输入 -> 核心方法 -> 输出 的框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    )
    monkeypatch.setattr(diagram_module, "_resolve_font_path", lambda *args, **kwargs: None)
    rendered = render_diagram(contract, build_visual_style_spec(contract))

    assert rendered.qa["hard_status"] == "fail"
    assert rendered.qa["checks"]["font_resolved"] is False


def test_legacy_figurespec_contract_maps_to_academic_svg_with_warning():
    contract = build_figure_contract(
        "画 Input -> Method -> Output 的框架图",
        FigureSourceConfig(resolved_scope="workspace"),
    ).model_copy(update={"renderer": "figurespec"})
    rendered = render_diagram(contract, build_visual_style_spec(contract))

    assert rendered.renderer == "academic_svg"
    assert rendered.editable_extension == "svg"
    assert any("deprecated" in warning for warning in rendered.qa["warnings"])


def test_edit_banana_draft_marks_raster_unverified_edges_and_ocr_labels():
    xml = b'''<mxfile><diagram><mxGraphModel><root>
      <mxCell id="0"/><mxCell id="1" parent="0"/>
      <mxCell id="image" value="Wrong OCR" style="shape=image;image=data:image/png;base64,AA==" vertex="1" parent="1"/>
      <mxCell id="edge" edge="1" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    </root></mxGraphModel></diagram></mxfile>'''

    result = inspect_edit_banana_drawio(xml, label_allowlist=["Expected"])

    assert result.canonical is False
    assert result.topology_verified is False
    assert result.raster_inside_drawio is True
    assert any("OCR labels" in warning for warning in result.warnings)
    assert any("connector endpoints" in warning for warning in result.warnings)


def test_edit_banana_unconfigured_enters_hitl_without_network(service, monkeypatch):
    session = service.store.create_session()
    reference = Path(session.workspace_root, "figures", "uploads", "reference.png")
    reference.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="white").save(reference)
    session.upload_batches.append(
        UploadBatchRecord(relative_paths=["figures/uploads/reference.png"])
    )
    service.store.save_session(session)

    async def unexpected_convert(*args, **kwargs):
        raise AssertionError("unconfigured Edit Banana must not make a network request")

    monkeypatch.setattr(agent_module, "convert_reference_to_drawio", unexpected_convert)
    result = asyncio.run(service.chat(session.session_id, "/fig 复刻参考图并拆解成可编辑对象"))

    assert result["status"] == "waiting_human"
    contract = json.loads(
        Path(session.workspace_root, "figures", "FIGURE_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    assert "EDIT_BANANA_BASE_URL" in contract["decision_question"]


def test_drawio_multi_panel_layout_keeps_native_objects_inside_assigned_panels():
    contract = FigureContract.model_validate_json(
        (FIGURE_FIXTURES / "mechanism_contract.json").read_text(encoding="utf-8")
    )
    style = build_visual_style_spec(contract)

    rendered = render_diagram(contract, style)
    root = ET.fromstring(rendered.editable_bytes)

    assert len(contract.panels) == 2
    assert {node["panel_id"] for node in rendered.render_spec["nodes"]} == {"a", "b"}
    assert {edge["style"] for edge in rendered.render_spec["edges"]} == {
        "solid",
        "dashed",
        "dotted",
    }
    assert len(root.findall(".//mxCell[@id='panel_a']")) == 1
    assert len(root.findall(".//mxCell[@id='panel_b']")) == 1
    assert rendered.qa["hard_status"] == "pass"


def test_missing_required_data_source_requests_human_decision():
    contract = build_figure_contract(
        "画准确率分组柱状图",
        FigureSourceConfig(resolved_scope="workspace"),
    )

    assert contract.kind == "data_plot"
    assert contract.renderer == "matplotlib"
    assert contract.decision_required is True
    assert "数据源" in contract.decision_question


def test_diagram_qa_catches_unapproved_label_and_reversed_edge():
    sources = FigureSourceConfig(resolved_scope="workspace")
    contract = build_figure_contract("画 Input -> Core Method -> Output 的框架图", sources)
    style = build_visual_style_spec(contract)
    render_spec = build_diagram_render_spec(contract, style)
    render_spec["nodes"][0]["label"] = "Hallucinated Label"
    render_spec["edges"][0]["from"], render_spec["edges"][0]["to"] = (
        render_spec["edges"][0]["to"],
        render_spec["edges"][0]["from"],
    )

    qa = validate_diagram(
        contract,
        style,
        render_spec,
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"{}",
    )

    assert qa["hard_status"] == "fail"
    assert any("allowlist" in error for error in qa["errors"])
    assert any("arrow directions" in error for error in qa["errors"])


def test_diagram_qa_catches_missing_entity_overlap_bounds_and_low_contrast():
    sources = FigureSourceConfig(resolved_scope="workspace")
    contract = build_figure_contract("画 Input -> Method -> Output 的框架图", sources)
    style = build_visual_style_spec(contract)
    style.palette[0] = "#FFFFFF"
    render_spec = build_diagram_render_spec(contract, style)
    render_spec["nodes"].pop()
    render_spec["nodes"][0]["x"] = -10
    render_spec["nodes"][1]["x"] = -10
    render_spec["nodes"][1]["y"] = render_spec["nodes"][0]["y"]

    qa = validate_diagram(
        contract,
        style,
        render_spec,
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"{}",
    )

    assert qa["hard_status"] == "fail"
    assert any("entities" in error for error in qa["errors"])
    assert any("overlap" in error for error in qa["errors"])
    assert any("bounds" in error for error in qa["errors"])
    assert any("low contrast" in warning for warning in qa["warnings"])


def test_wps_request_only_writes_manual_handoff(service, monkeypatch):
    async def unexpected_generate_image(**kwargs):
        raise AssertionError("WPS handoff must not trigger image generation for a framework route")

    monkeypatch.setattr(agent_module, "generate_image", unexpected_generate_image)
    result = asyncio.run(service.chat(None, "/fig 画一个 pipeline，并准备 WPS 图片转PPT交接"))
    task = service.get_task(result["task_id"])
    assert task is not None
    handoff = json.loads(
        Path(task.artifact_root, "figures", "generated", "WPS_HANDOFF.json").read_text(
            encoding="utf-8"
        )
    )

    assert handoff["external_upload_performed"] is False
    assert handoff["canonical_source"] is False
