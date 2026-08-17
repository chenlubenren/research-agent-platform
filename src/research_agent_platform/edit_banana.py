from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx


@dataclass
class EditBananaResult:
    drawio_xml: bytes
    raster_inside_drawio: bool
    topology_verified: bool = False
    canonical: bool = False
    vlm_called: bool = False
    warnings: list[str] = field(default_factory=list)


async def convert_reference_to_drawio(
    image_path: Path,
    *,
    base_url: str,
    timeout_seconds: float = 300,
    label_allowlist: list[str] | None = None,
) -> EditBananaResult:
    if not base_url:
        raise RuntimeError("Edit Banana is not configured (EDIT_BANANA_BASE_URL is empty)")
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/convert",
            files={"file": (image_path.name, image_path.read_bytes(), _mime_type(image_path))},
        )
        response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        payload = response.json()
        raw = payload.get("drawio_xml") or payload.get("xml") or ""
        if payload.get("drawio_base64"):
            xml_bytes = base64.b64decode(payload["drawio_base64"])
        else:
            xml_bytes = str(raw).encode("utf-8")
        vlm_called = bool(payload.get("vlm_called", False))
    else:
        xml_bytes = response.content
        vlm_called = False
    return inspect_edit_banana_drawio(
        xml_bytes,
        label_allowlist=label_allowlist or [],
        vlm_called=vlm_called,
    )


def inspect_edit_banana_drawio(
    xml_bytes: bytes,
    *,
    label_allowlist: list[str],
    vlm_called: bool = False,
) -> EditBananaResult:
    root = ET.fromstring(xml_bytes)
    if root.tag != "mxfile" or root.find(".//mxGraphModel") is None:
        raise ValueError("Edit Banana response is not an mxGraph Draw.io document")
    warnings: list[str] = []
    cells = root.findall(".//mxCell")
    raster = any(
        "image=data:image/" in (cell.get("style") or "").lower()
        and "image/svg+xml" not in (cell.get("style") or "").lower()
        for cell in cells
    )
    unconnected_edges = [
        cell.get("id", "")
        for cell in cells
        if cell.get("edge") == "1" and not (cell.get("source") and cell.get("target"))
    ]
    if unconnected_edges:
        warnings.append(f"unverified connector endpoints: {unconnected_edges}")
    allowed = set(label_allowlist)
    unexpected = sorted(
        {
            value
            for cell in cells
            if (value := (cell.get("value") or "").strip())
            and value not in allowed
            and not value.startswith("<")
        }
    )
    if unexpected:
        warnings.append(f"OCR labels outside FigureContract allowlist: {unexpected}")
    if raster:
        warnings.append("Draw.io draft contains movable raster image cells")
    return EditBananaResult(
        drawio_xml=xml_bytes,
        raster_inside_drawio=raster,
        vlm_called=vlm_called,
        warnings=warnings,
    )


def _mime_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }.get(path.suffix.lower(), "application/octet-stream")
