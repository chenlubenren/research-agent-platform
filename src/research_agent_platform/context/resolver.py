from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..paper_pipeline import collect_paper_evidence, paper_evidence_prompt


@dataclass(frozen=True)
class ResearchContextPackage:
    objective: str
    source_refs: list[str] = field(default_factory=list)
    evidence_records: list[dict] = field(default_factory=list)
    context_text: str = ""
    evidence_source: str = "none"
    limitations: list[str] = field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        return [
            str(record.get("evidence_id", ""))
            for record in self.evidence_records
            if str(record.get("evidence_id", ""))
        ]


def discover_pdf_sources(workspace_root: str | Path) -> list[str]:
    root = Path(workspace_root)
    candidates = list(root.glob("*/uploads/*.pdf"))
    candidates.extend((root / "bib" / "papers").glob("*.pdf"))
    unique: dict[str, Path] = {}
    for path in candidates:
        if not path.is_file():
            continue
        unique[str(path.resolve())] = path
    ordered = sorted(unique.values(), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return [path.relative_to(root).as_posix() for path in ordered]


def resolve_local_research_context(
    workspace_root: str | Path,
    objective: str,
    *,
    total_limit: int = 18000,
) -> ResearchContextPackage:
    root = Path(workspace_root)
    for relative_path in ("bib/EVIDENCE_MAP.md", "bib/LITERATURE_REVIEW.md"):
        path = root / relative_path
        if path.exists():
            return ResearchContextPackage(
                objective=objective,
                source_refs=[relative_path],
                context_text=path.read_text(encoding="utf-8", errors="ignore")[:total_limit],
                evidence_source="review",
            )

    source_refs = discover_pdf_sources(root)
    if not source_refs:
        return ResearchContextPackage(
            objective=objective,
            limitations=["No local PDF or review evidence was found."],
        )

    evidence_records = collect_paper_evidence(
        root,
        source_refs,
        total_limit=total_limit,
        query=objective,
        prioritize_research_sections=True,
    )
    limitations: list[str] = []
    if not evidence_records:
        limitations.append("Local PDFs were found, but no extractable text evidence was available.")
    return ResearchContextPackage(
        objective=objective,
        source_refs=source_refs,
        evidence_records=evidence_records,
        context_text=paper_evidence_prompt(evidence_records, limit=total_limit),
        evidence_source="local_pdf",
        limitations=limitations,
    )
