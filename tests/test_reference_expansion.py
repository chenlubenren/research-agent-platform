from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from reportlab.pdfgen import canvas

from research_agent_platform import reference_expansion
from research_agent_platform.reference_expansion import (
    expand_pdf_references,
    extract_reference_records,
)


def _pdf_bytes(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer)
    y = 760
    for line in lines:
        document.drawString(72, y, line)
        y -= 18
    document.save()
    return buffer.getvalue()


def _write_reference_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        _pdf_bytes(
            [
                "Main Paper Title",
                "References",
                "Alpha, A. Reliable graph baselines. In ICLR, 2020.",
                "Beta, B. Evidence grounded experiments. arXiv preprint arXiv:2101.01234, 2021.",
                "Appendix",
                "This text is not a reference.",
            ]
        )
    )


def test_extract_reference_records_stops_before_appendix(tmp_path: Path):
    source = tmp_path / "main.pdf"
    _write_reference_pdf(source)

    records = extract_reference_records(source)

    assert len(records) == 2
    assert records[0].reference_id == "R001"
    assert records[0].title == "Reliable graph baselines"
    assert records[0].year == 2020
    assert records[1].title == "Evidence grounded experiments"
    assert records[1].arxiv_id == "2101.01234"
    assert "Appendix" not in records[1].raw_text


def test_expand_references_downloads_public_pdfs_and_records_failures(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    source = workspace / "paper" / "uploads" / "main.pdf"
    _write_reference_pdf(source)

    async def fake_resolve(record, *, client, semaphore):
        if record.reference_id == "R001":
            return {
                "title": "Reliable graph baselines",
                "year": 2020,
                "authors": ["Alpha"],
                "openalex_id": "https://openalex.org/W1",
                "candidate_urls": ["https://example.org/alpha.pdf"],
            }
        return {}

    async def fake_download(client, urls, *, timeout_seconds):
        if "alpha.pdf" in urls[0]:
            return _pdf_bytes(["Reliable graph baselines", "Discussion and experiments"]), ""
        return b"", "not available"

    monkeypatch.setattr(reference_expansion, "_resolve_openalex_reference", fake_resolve)
    monkeypatch.setattr(reference_expansion, "_download_first_pdf", fake_download)

    result = asyncio.run(
        expand_pdf_references(
            "paper/uploads/main.pdf",
            workspace,
            primary_paper_id="P-MAIN00000001",
            cache=False,
        )
    )

    assert result.extracted_count == 2
    assert result.records[0].status == "downloaded"
    assert result.records[0].source_relative_path.startswith("bib/references/R001_")
    assert Path(workspace, result.records[0].source_relative_path).exists()
    assert result.records[1].status == "unavailable"
    manifest = json.loads(Path(workspace, result.manifest_relative_path).read_text(encoding="utf-8"))
    assert [record["status"] for record in manifest["records"]] == ["downloaded", "unavailable"]
