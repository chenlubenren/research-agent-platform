from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter


def extract_citations(manuscript: str) -> set[str]:
    keys = set(re.findall(r"(?<![A-Za-z0-9_@])@([A-Za-z0-9_:-]+)(?![A-Za-z0-9_:-]*\s*\{)", manuscript))
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def find_author_year_citations(manuscript: str) -> list[str]:
    citations: set[str] = set()
    for group in re.findall(r"\(([^()\n]+)\)", manuscript):
        for candidate in group.split(";"):
            match = re.fullmatch(
                r"\s*((?:[a-z][A-Za-z'’-]+\s+)*[A-Z][A-Za-z'’-]+)(?:(?:\s+et al\.)|(?:\s+(?:and|&)\s+[A-Z][A-Za-z'’-]+))?,?\s+(\d{4}[a-z]?)\s*",
                candidate,
            )
            if match:
                citations.add(f"{match.group(1)}, {match.group(2)}")
    return sorted(citations)


def extract_bibliography_author_years(bibliography: str) -> set[tuple[str, str]]:
    signatures: set[tuple[str, str]] = set()
    entries = re.split(r"(?=@[A-Za-z]+\s*\{)", bibliography)
    for entry in entries:
        author_match = re.search(r"\bauthor\s*=\s*[\{\"]([^}\"]+)", entry, re.I)
        year_match = re.search(r"\byear\s*=\s*[\{\"]?(\d{4}[a-z]?)", entry, re.I)
        if not author_match or not year_match:
            continue
        first_author = re.split(r"\s+and\s+", author_match.group(1), maxsplit=1, flags=re.I)[0]
        surname = first_author.split(",", 1)[0] if "," in first_author else first_author.split()[-1]
        normalized_surname = re.sub(r"[^A-Za-z]", "", surname).lower()
        if normalized_surname:
            signatures.add((normalized_surname, year_match.group(1).lower()))
    return signatures


def _author_year_signature(citation: str) -> tuple[str, str]:
    author, year = citation.split(", ", 1)
    return re.sub(r"[^A-Za-z]", "", author).lower(), year.lower()


def _has_nearby_evidence(
    lines: list[str],
    line_index: int,
    citation_keys: set[str] | None = None,
    evidence_ids: set[str] | None = None,
) -> bool:
    support_pattern = re.compile(
        r"@[A-Za-z0-9_:-]+|\[PE-[A-Z0-9]+\]|\\cite\w*\{|\([A-Z][A-Za-z'’-]+(?:\s+et al\.)?,?\s+\d{4}[a-z]?\)|\b(?:Figure|Fig\.|Table)\s+\d+\b",
        re.I,
    )
    start = max(0, line_index - 1)
    end = min(len(lines), line_index + 2)
    for line in lines[start:end]:
        if citation_keys is None and re.search(r"@[A-Za-z0-9_:-]+", line):
            return True
        if citation_keys is not None and any(key in citation_keys for key in re.findall(r"@([A-Za-z0-9_:-]+)", line)):
            return True
        if evidence_ids is None and re.search(r"\bPE-[A-Z0-9-]+\b", line, re.I):
            return True
        if evidence_ids is not None and any(
            evidence_id.upper() in evidence_ids
            for evidence_id in re.findall(r"\bPE-[A-Z0-9-]+\b", line, re.I)
        ):
            return True
        if support_pattern.search(line) and not re.search(r"@[A-Za-z0-9_:-]+|\bPE-[A-Z0-9-]+\b", line, re.I):
            return True
    return False


def _section_present(manuscript: str, section: str) -> bool:
    escaped = re.escape(section)
    return bool(
        re.search(rf"^#+\s+.*{escaped}", manuscript, re.I | re.M)
        or re.search(rf"\\(?:sub)*section\*?\{{[^}}]*{escaped}[^}}]*\}}", manuscript, re.I)
    )


def find_manuscript(workspace_root: Path) -> tuple[str, str]:
    allowed_directories = ("paper", "Content")
    candidates = sorted(
        (
            path
            for directory in allowed_directories
            for path in (workspace_root / directory).rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".md", ".txt", ".tex"}
            and path.name not in {"MANIFEST.md"}
        ),
        key=lambda path: (
            path.name.casefold() != "paper_revised.md",
            "uploads" in path.relative_to(workspace_root).parts,
            path.name.casefold(),
        ),
    )
    if not candidates:
        raise ValueError("缺少可审阅的稿件；请上传 Markdown、TXT 或 LaTeX 稿件，或先完成 /write。")
    path = candidates[0]
    return path.relative_to(workspace_root).as_posix(), path.read_text(encoding="utf-8", errors="ignore")


def build_review_package(
    manuscript_path: str,
    manuscript: str,
    workspace_root: Path | None = None,
    *,
    document_type: str = "journal_article",
) -> dict:
    findings: list[dict] = []
    required_sections = () if document_type == "degree_thesis_section" else ("abstract", "introduction", "method", "conclusion")
    for section in required_sections:
        if not _section_present(manuscript, section):
            findings.append(_finding("structure", "major", "whole manuscript", f"Missing required section: {section.title()}.", "Add a bounded section supported by the frozen evidence."))
    manuscript_lines = manuscript.splitlines()
    for line_no, line in enumerate(manuscript_lines, start=1):
        if re.search(r"AUTHOR INPUT NEEDED|TODO|PLACEHOLDER|CITATION NEEDED", line, re.I):
            findings.append(_finding("evidence", "major", f"line {line_no}", "Unresolved author input or placeholder remains.", "Resolve it with traceable evidence or retain it as an explicit limitation before submission."))
    if not extract_citations(manuscript) and not find_author_year_citations(manuscript):
        findings.append(_finding("citations", "minor", "whole manuscript", "No machine-readable citation keys were found.", "Verify that references are complete and use the target venue citation convention."))
    citation_keys: set[str] | None = None
    evidence_ids: set[str] | None = None
    if workspace_root is not None:
        citation_keys = set()
        evidence_ids = set()
        for path in (workspace_root / "bib").rglob("*.bib"):
            citation_keys.update(
                re.findall(r"@\w+\s*\{\s*([^,\s]+)", path.read_text(encoding="utf-8", errors="ignore"))
            )
        evidence_map_path = workspace_root / "paper" / "PAPER_EVIDENCE_MAP.json"
        if evidence_map_path.exists():
            try:
                records = json.loads(evidence_map_path.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    evidence_ids.update(
                        str(record.get("evidence_id", "")).upper()
                        for record in records
                        if isinstance(record, dict) and record.get("evidence_id")
                    )
            except json.JSONDecodeError:
                pass
    for line_no, line in enumerate(manuscript_lines, start=1):
        if re.search(r"\b(?:improve(?:s|d)?|outperform(?:s|ed)?|state-of-the-art|significant(?:ly)?|universal|always|never|prove(?:s|d)?)\b", line, re.I) and not _has_nearby_evidence(manuscript_lines, line_no - 1, citation_keys, evidence_ids):
            findings.append(_finding("claims", "major", f"line {line_no}", "Comparative or strong claim lacks a local citation or evidence identifier.", "Attach a source citation or frozen evidence ID, or narrow the claim."))
        if re.search(r"(?:\b\d+(?:\.\d+)?\s*%|\bp\s*[<=>]\s*0?\.\d+|\b(?:confidence interval|CI)\b)", line, re.I) and not _has_nearby_evidence(manuscript_lines, line_no - 1, citation_keys, evidence_ids):
            findings.append(_finding("claims", "major", f"line {line_no}", "Quantitative claim lacks a nearby citation, evidence identifier, or figure/table reference.", "Attach traceable evidence near the value or mark it as unresolved author input."))
        if len(line) > 280:
            findings.append(_finding("language-format", "minor", f"line {line_no}", "Sentence or paragraph line is unusually long.", "Split it into shorter claim-evidence-reasoning units."))
    if not _section_present(manuscript, "limitations") and not re.search(r"(?im)^#+\s+讨论", manuscript):
        findings.append(_finding("claims", "minor", "whole manuscript", "No explicit limitations section was found.", "State scope, uncertainty, and unsupported future work explicitly."))
    reviewers = ("editor", "domain", "methods", "evidence", "adversarial", "language-format")
    for index, finding in enumerate(findings, start=1):
        finding["finding_id"] = f"F{index:03d}"
        finding["reviewers"] = [reviewers[(index - 1) % len(reviewers)]]
    decision = "REVISE" if any(item["severity"] == "major" for item in findings) else "PASS"
    return {"review_mode": "simulated-peer-review", "manuscript_path": manuscript_path, "decision": decision, "findings": findings, "limitations": ["This is an automated simulated review, not a journal decision."]}


def merge_role_findings(role_findings: dict[str, list[dict]]) -> dict:
    """Merge independently produced role findings without inventing a consensus."""
    severity_rank = {"minor": 1, "major": 2}
    merged: dict[tuple[str, str, str], dict] = {}
    for role, findings in role_findings.items():
        for item in findings:
            location = str(item.get("location") or "whole manuscript").strip()
            category = str(item.get("category") or "review").strip().lower()
            problem = str(item.get("problem") or "Issue requires author review.").strip()
            key = (category, _review_key(location), _review_key(problem))
            severity = str(item.get("severity") or "minor").lower()
            severity = severity if severity in severity_rank else "minor"
            action = str(item.get("recommended_action") or "Review the manuscript against the frozen evidence.").strip()
            evidence_ids = sorted({str(value) for value in item.get("evidence_ids", []) if str(value)})
            if key not in merged:
                merged[key] = {
                    "finding_id": "",
                    "category": category,
                    "severity": severity,
                    "location": location,
                    "problem": problem,
                    "impact": str(item.get("impact") or "May weaken the manuscript's traceability or submission readiness."),
                    "recommended_action": action,
                    "evidence_ids": evidence_ids,
                    "reviewers": [role],
                    "conflict_status": "none",
                    "alternative_actions": [action],
                }
                continue
            target = merged[key]
            if severity_rank[severity] > severity_rank[target["severity"]]:
                target["severity"] = severity
            target["reviewers"] = sorted(set(target["reviewers"]) | {role})
            target["evidence_ids"] = sorted(set(target["evidence_ids"]) | set(evidence_ids))
            target["alternative_actions"] = sorted(set(target["alternative_actions"]) | {action})
            if len(target["alternative_actions"]) > 1:
                target["conflict_status"] = "conflicting_recommendations"

    ordered = sorted(
        merged.values(),
        key=lambda item: (-severity_rank[item["severity"]], item["location"], item["category"], item["problem"]),
    )
    for index, item in enumerate(ordered, start=1):
        item["finding_id"] = f"RF{index:03d}"
    return {
        "schema_version": "role-review-merge/v1",
        "findings": ordered,
        "merge_policy": "Exact normalized category, location, and problem matches are deduplicated; highest severity is retained; distinct recommendations are kept and flagged.",
    }


def parse_role_findings(role: str, content: str) -> list[dict]:
    """Accept only the explicit JSON finding contract from an isolated reviewer."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    candidates = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        return []
    normalized: list[dict] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        location = str(item.get("location") or "").strip()
        category = str(item.get("category") or "").strip().lower()
        severity = str(item.get("severity") or "").strip().lower()
        problem = str(item.get("problem") or "").strip()
        action = str(item.get("recommended_action") or "").strip()
        if not all((location, category, problem, action)) or severity not in {"major", "minor"}:
            continue
        evidence_ids = sorted({str(value).strip() for value in item.get("evidence_ids", []) if str(value).strip()})
        normalized.append(
            {
                "location": location,
                "category": category,
                "severity": severity,
                "problem": problem,
                "impact": str(item.get("impact") or "May weaken the manuscript's traceability or submission readiness."),
                "recommended_action": action,
                "evidence_ids": evidence_ids,
                "reviewers": [role],
            }
        )
    return normalized


def _review_key(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def build_final_gate_report(
    manuscript_path: str,
    manuscript: str,
    workspace_root: Path,
    *,
    document_type: str = "journal_article",
) -> dict:
    bibliography_keys: set[str] = set()
    bibliography_author_years: set[tuple[str, str]] = set()
    for path in (workspace_root / "bib").rglob("*.bib"):
        bibliography = path.read_text(encoding="utf-8", errors="ignore")
        bibliography_keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
        bibliography_author_years.update(extract_bibliography_author_years(bibliography))
    cited = extract_citations(manuscript)
    unknown = sorted(cited - bibliography_keys)
    author_year_citations = find_author_year_citations(manuscript)
    unknown_author_years = sorted(
        citation
        for citation in author_year_citations
        if _author_year_signature(citation) not in bibliography_author_years
    )
    known_evidence_ids: set[str] = set()
    evidence_map_path = workspace_root / "paper" / "PAPER_EVIDENCE_MAP.json"
    if evidence_map_path.exists():
        try:
            evidence_records = json.loads(evidence_map_path.read_text(encoding="utf-8"))
            if isinstance(evidence_records, list):
                known_evidence_ids.update(
                    str(record.get("evidence_id", "")).upper()
                    for record in evidence_records
                    if isinstance(record, dict) and record.get("evidence_id")
                )
        except json.JSONDecodeError:
            pass
    referenced_evidence_ids = {value.upper() for value in re.findall(r"\bPE-[A-Z0-9-]+\b", manuscript, re.I)}
    unknown_evidence_ids = sorted(referenced_evidence_ids - known_evidence_ids)
    placeholders = sorted({match.group(0) for match in re.finditer(r"AUTHOR INPUT NEEDED|TODO|PLACEHOLDER|CITATION NEEDED", manuscript, re.I)})
    labels = set(re.findall(r"\\label\{([^}]+)\}", manuscript))
    references = set(re.findall(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}", manuscript))
    figure_mentions = [int(value) for value in re.findall(r"(?i)\b(?:figure|fig\.)\s+(\d+)\b", manuscript)]
    table_mentions = [int(value) for value in re.findall(r"(?i)\btable\s+(\d+)\b", manuscript)]
    figure_definitions = [int(value) for value in re.findall(r"(?im)^\s*(?:figure|fig\.)\s+(\d+)\s*[:.]", manuscript)]
    table_definitions = [int(value) for value in re.findall(r"(?im)^\s*table\s+(\d+)\s*[:.]", manuscript)]
    def numbering_issues(values: list[int], definitions: list[int]) -> dict:
        counts = Counter(values)
        expected = set(range(1, max(values) + 1)) if values else set()
        definition_counts = Counter(definitions)
        return {
            "missing": sorted(expected - set(values)),
            "duplicate": sorted(value for value, count in definition_counts.items() if count > 1),
        }
    figure_issues = numbering_issues(figure_mentions, figure_definitions)
    table_issues = numbering_issues(table_mentions, table_definitions)
    required_sections = () if document_type == "degree_thesis_section" else ("abstract", "introduction", "method", "conclusion")
    missing_sections = [
        section
        for section in required_sections
        if not _section_present(manuscript, section)
    ]
    checks = [
        {"check": "manuscript_present", "status": "pass" if manuscript.strip() else "violated", "severity": "hard"},
        {"check": "citation_keys_resolve", "status": "violated" if unknown else "pass", "severity": "hard", "detail": unknown},
        {"check": "author_year_citations_resolve", "status": "violated" if unknown_author_years else "pass", "severity": "hard", "detail": unknown_author_years},
        {"check": "evidence_ids_resolve", "status": "violated" if unknown_evidence_ids else "pass", "severity": "hard", "detail": unknown_evidence_ids},
        {"check": "placeholders_resolved", "status": "violated" if placeholders else "pass", "severity": "soft", "detail": placeholders},
        {"check": "latex_cross_references_resolve", "status": "violated" if references - labels else "pass", "severity": "hard", "detail": sorted(references - labels)},
        {"check": "required_sections_present", "status": "violated" if missing_sections else "pass", "severity": "hard", "detail": missing_sections},
        {"check": "markdown_figure_numbering", "status": "violated" if figure_issues["missing"] or figure_issues["duplicate"] else "pass", "severity": "soft", "detail": figure_issues},
        {"check": "markdown_table_numbering", "status": "violated" if table_issues["missing"] or table_issues["duplicate"] else "pass", "severity": "soft", "detail": table_issues},
    ]
    hard = any(check["status"] == "violated" and check["severity"] == "hard" for check in checks)
    soft = any(check["status"] == "violated" and check["severity"] == "soft" for check in checks)
    return {
        "manuscript_path": manuscript_path,
        "document_type": document_type,
        "decision": "BLOCK" if hard else "REVISE" if soft else "PASS",
        "checks": checks,
    }


def review_report_markdown(payload: dict) -> str:
    """Render the author-facing simulated review without exposing internal gates."""
    findings = payload.get("findings") or []
    decision = str(payload.get("decision") or "REVISE")
    status = str(payload.get("review_status") or "available")
    lines = [
        "# Simulated Peer Review",
        "",
        f"- Manuscript: `{payload.get('manuscript_path', 'unknown')}`",
        f"- Status: {status}",
        f"- Recommendation: {decision}",
        "- Scope: This is a simulated scholarly review, not an editorial decision or a submission-readiness certification.",
        "",
        "## Editorial Summary",
        "",
    ]
    if status == "needs_attention":
        unavailable_roles = ", ".join(str(role) for role in payload.get("unavailable_roles") or [])
        detail = f" Unavailable reviewer roles: {unavailable_roles}." if unavailable_roles else ""
        lines.append(
            "The simulated review is incomplete and must not be treated as a clean review result."
            + detail
        )
    elif not findings:
        lines.append("No material issue was identified by the automated simulated review. The author should still verify evidence, citations, and venue requirements before submission.")
    else:
        major_count = sum(item.get("severity") == "major" for item in findings)
        lines.append(f"The review identified {len(findings)} issue(s), including {major_count} major issue(s). Address major issues before treating the manuscript as ready for external review.")
    for heading, severity in (("Major Comments", "major"), ("Minor Comments", "minor")):
        lines.extend(["", f"## {heading}", ""])
        matching = [item for item in findings if item.get("severity") == severity]
        if not matching:
            lines.append("None.")
            continue
        for item in matching:
            finding_id = item.get("finding_id", "Finding")
            location = item.get("location", "manuscript")
            problem = item.get("problem", "Issue requires author review.")
            action = item.get("recommended_action", "Revise with traceable evidence.")
            lines.extend(
                [
                    f"### {finding_id}: {item.get('category', 'review').replace('-', ' ').title()}",
                    "",
                    f"- Location: {location}",
                    f"- Comment: {problem}",
                    f"- Suggested action: {action}",
                ]
            )
    lines.extend(["", "## Review Boundary", ""])
    lines.extend(f"- {item}" for item in payload.get("limitations") or ["This review is advisory only."])
    return "\n".join(lines) + "\n"


def to_markdown(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def _infer_section_role(text: str) -> dict:
    candidates = (
        ("abstract", ("摘要", "abstract"), "Summarize purpose, approach, bounded material and intended contribution; do not add unsupported results."),
        ("introduction", ("引言", "introduction"), "Establish problem, gap, contribution and scope; do not report results or repeat a literature list."),
        ("literature_review", ("文献综述", "related work", "literature review"), "Synthesize sources by theme or contrast; do not treat retrieval metadata as evidence."),
        ("methods", ("研究方法", "方法", "methods", "methodology"), "Report design, materials, procedures and analysis exactly as supplied; do not invent protocol details."),
        ("results", ("研究结果", "结果", "experiments", "results"), "Report observed values, units and uncertainty before interpretation; do not introduce causal claims beyond evidence."),
        ("discussion", ("讨论", "discussion"), "Interpret supported findings against conditions and limits; keep mechanisms and implications calibrated."),
        ("conclusion", ("结论", "conclusion"), "State only findings and implications already supported in the manuscript; do not introduce new evidence."),
    )
    lowered = text.casefold()
    matches = [item for item in candidates if any(term.casefold() in lowered for term in item[1])]
    if len(matches) != 1:
        return {"role": "unknown", "confidence": "low", "guidance": "Ask the author to confirm the intended section before altering section-specific structure."}
    role, _, guidance = matches[0]
    return {"role": role, "confidence": "medium", "guidance": guidance}


def build_writing_context(task_objective: str, source_refs: list[str], venue_profile: dict) -> dict:
    lowered = task_objective.lower()
    language = "zh" if re.search(r"中文|汉语|chinese", lowered) else "en"
    article_type = "review" if re.search(r"综述|survey|review article", lowered) else "research"
    document_type = (
        "degree_thesis_section"
        if re.search(r"学位论文|博士论文|硕士论文|博士学位|硕士学位|thesis|dissertation|章节润色|章节修改|论文框架", lowered)
        else "journal_article"
    )
    disciplines = {
        "environmental_science_engineering": (
            "environmental science",
            "environmental engineering",
            "hydrology",
            "环境科学",
            "环境工程",
            "水文",
        ),
        "ai_computer_science": ("computer science", "machine learning", "机器学习", "人工智能", " ai "),
        "medicine_clinical": ("medicine", "clinical", "医学", "临床"),
        "chemistry_materials": ("chemistry", "materials science", "化学", "材料科学"),
        "social_science": ("social science", "sociology", "社会科学", "社会学"),
        "law_legal": ("legal", "jurisprudence", "法律", "法学"),
        "arts_humanities": ("humanities", "literary studies", "人文", "文学", "历史学", "艺术史"),
        "mathematics_theory": ("mathematics", "mathematical", "theorem", "数学", "定理"),
        "life_science_biology": ("life science", "biology", "生命科学", "生物学"),
    }
    discipline = "general"
    for name, terms in disciplines.items():
        if any(term in lowered for term in terms):
            discipline = name
            break
    return {
        "schema_version": "writing-context/v1",
        "language": language,
        "article_type": article_type,
        "document_type": document_type,
        "discipline": discipline,
        "section_role": _infer_section_role(task_objective),
        "venue_profile": venue_profile,
        "source_boundary": "frozen_at_task_start",
        "source_refs": source_refs,
        "rules": [
            "Evidence precedes prose; unsupported claims remain AUTHOR INPUT NEEDED.",
            "Preserve supplied numbers, equations, citation keys, figures, and tables.",
            "Keep demonstrated results separate from intended contributions and future work.",
        ],
    }


def build_revision_audit(original: str, revised: str) -> dict:
    original_lines = original.splitlines()
    revised_lines = revised.splitlines()
    changed = sum(1 for before, after in zip(original_lines, revised_lines) if before != after)
    changed += abs(len(original_lines) - len(revised_lines))
    original_numbers = _audit_number_tokens(original)
    revised_numbers = _audit_number_tokens(revised)
    numbers_preserved, derived_numbers = _audit_numeric_changes(original_numbers, revised_numbers)
    return {
        "schema_version": "revision-audit/v1",
        "changed_line_count": changed,
        "original_line_count": len(original_lines),
        "revised_line_count": len(revised_lines),
        "numbers_preserved": numbers_preserved,
        "derived_numbers": derived_numbers,
        "citation_keys_preserved": sorted(re.findall(r"@([A-Za-z0-9_:-]+)", original)) == sorted(re.findall(r"@([A-Za-z0-9_:-]+)", revised)),
        "unresolved_author_inputs": sorted(set(re.findall(r"AUTHOR INPUT NEEDED[^\n]*", revised, re.I))),
    }


def build_revision_rationale(original: str, revised: str, review: str = "") -> str:
    """Create an author-facing rationale without pretending that a diff proves scientific correctness."""
    original_lines = original.splitlines()
    revised_lines = revised.splitlines()
    changed = [
        (index + 1, before, after)
        for index, (before, after) in enumerate(zip(original_lines, revised_lines))
        if before.strip() != after.strip()
    ]
    lines = [
        "# Revision Rationale",
        "",
        "This rationale records why the platform changed wording or structure. It is an author-verification aid, not proof that the underlying science is correct.",
        "",
        "## Basis",
        "",
        "- Evidence basis: supplied manuscript, frozen source records, and the internal reviewer findings.",
        "- Preservation rule: numbers, citations, equations, and uncertainty are not changed unless the reviewer finding explicitly requires author verification.",
        "",
        "## Changes",
        "",
    ]
    if not changed:
        lines.append("No line-level change was detected in the final manuscript.")
    else:
        for line_no, before, after in changed[:80]:
            lines.extend([
                f"### Change near source line {line_no}",
                "",
                f"- Before: {before.strip() or '[blank line]'}",
                f"- After: {after.strip() or '[blank line]'}",
                "- Reason: improve claim precision, section coherence, language, or reviewer-request traceability; confirm against the cited source before submission.",
                "",
            ])
    if review.strip():
        findings = [line.strip() for line in review.splitlines() if re.match(r"[-*]\s+", line)]
        if findings:
            lines.extend(["## Reviewer Basis", "", *[f"- {item.lstrip('-* ').strip()}" for item in findings[:24]], ""])
    return "\n".join(lines).rstrip() + "\n"


def _audit_numeric_changes(original_numbers: list[str], revised_numbers: list[str]) -> tuple[bool, list[float]]:
    original_values = [float(item.rstrip("%")) for item in original_numbers]
    revised_values = [float(item.rstrip("%")) for item in revised_numbers]
    derivable_original = {
        value
        for value in original_values
        if any(
            abs(abs(left - right) - value) < 1e-9
            for index, left in enumerate(original_values)
            for right in original_values[index + 1 :]
            if left != right
        )
    }
    required_original = set(original_values) - derivable_original
    if not required_original.issubset(set(revised_values)):
        return False, []
    derived_values = {
        round(abs(left - right), 10)
        for index, left in enumerate(original_values)
        for right in original_values[index + 1 :]
        if left != right
    }
    derived_numbers: list[float] = []
    for value in set(revised_values) - set(original_values):
        if value not in derived_values:
            return False, []
        derived_numbers.append(value)
    return True, sorted(derived_numbers)


def _audit_number_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    pattern = re.compile(r"(?<![A-Za-z0-9_-])\d+(?:\.\d+)?%?(?![A-Za-z0-9_-])")
    for match in pattern.finditer(text):
        token = match.group(0)
        line_start = text.rfind("\n", 0, match.start()) + 1
        line = text[line_start : text.find("\n", match.end()) if "\n" in text[match.end() :] else len(text)]
        if re.match(r"\s*\d+\.\s", line) and "." not in token:
            continue
        tokens.append(token)
    return tokens


def _finding(category: str, severity: str, location: str, problem: str, action: str) -> dict:
    return {"finding_id": "", "category": category, "severity": severity, "location": location, "problem": problem, "impact": "May weaken the manuscript's traceability or submission readiness.", "recommended_action": action, "reviewers": []}
