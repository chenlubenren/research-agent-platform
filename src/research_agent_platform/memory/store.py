from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WikiPaper:
    paper_id: str
    source_relative_path: str
    wiki_pdf_relative_path: str
    summary_relative_path: str
    title: str


class ResearchWikiStore:
    DEFAULT_IDEA_MINIMUM_PAPERS = 5
    SUMMARY_SECTIONS = (
        "研究问题",
        "核心方法",
        "数据集与实验设置",
        "主要结果",
        "作者讨论与局限",
        "作者提出的未来工作",
        "作者明确指出的 Gap",
        "基于证据推断的 Gap",
        "可复用证据",
        "与其他论文的关系",
        "对 Idea 生成的提示",
        "证据边界与未确认内容",
    )

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.wiki_root = self.workspace_root / "wiki"
        self.papers_root = self.wiki_root / "papers"
        self.ideas_root = self.wiki_root / "ideas"
        self.papers_root.mkdir(parents=True, exist_ok=True)
        self.ideas_root.mkdir(parents=True, exist_ok=True)

    def paper_for_source(self, source_relative_path: str) -> WikiPaper:
        source = self._workspace_path(source_relative_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12].upper()
        paper_id = f"P-{digest}"
        paper_root = self.papers_root / paper_id
        return WikiPaper(
            paper_id=paper_id,
            source_relative_path=source_relative_path,
            wiki_pdf_relative_path=f"wiki/papers/{paper_id}/source.pdf",
            summary_relative_path=f"wiki/papers/{paper_id}/summary.md",
            title=source.stem,
        )

    def ensure_pdf_copy(self, paper: WikiPaper) -> bool:
        source = self._workspace_path(paper.source_relative_path)
        target = self._workspace_path(paper.wiki_pdf_relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and self._sha256(target) == self._sha256(source):
            return False
        shutil.copy2(source, target)
        return True

    def summary_exists(self, paper: WikiPaper) -> bool:
        return self._workspace_path(paper.summary_relative_path).exists()

    def write_summary(self, paper: WikiPaper, content: str) -> None:
        target = self._workspace_path(paper.summary_relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")

    def paper_summary_markdown(self, paper: WikiPaper, generated_summary: str) -> str:
        normalized_summary = generated_summary.strip()
        if normalized_summary.startswith("# "):
            normalized_summary = "\n".join(normalized_summary.splitlines()[1:]).lstrip()
        existing_sections = set(re.findall(r"^##\s+(.+?)\s*$", normalized_summary, re.M))
        missing_sections = [
            section for section in self.SUMMARY_SECTIONS if section not in existing_sections
        ]
        if missing_sections:
            normalized_summary = normalized_summary.rstrip() + "\n\n" + "\n\n".join(
                f"## {section}\n\n- 未确认。" for section in missing_sections
            )
        return (
            f"# {paper.title}\n\n"
            "## 基本信息\n"
            f"- Paper ID: `{paper.paper_id}`\n"
            f"- 原始文件: `{paper.source_relative_path}`\n"
            f"- Wiki PDF: `{paper.wiki_pdf_relative_path}`\n\n"
            f"{normalized_summary}\n"
        )

    def fallback_summary(self, paper: WikiPaper, evidence_records: Iterable[dict]) -> str:
        evidence_lines: list[str] = []
        for record in list(evidence_records)[:12]:
            evidence_id = str(record.get("evidence_id", ""))
            page = record.get("page") or "unknown"
            excerpt = str(record.get("excerpt", "")).strip().replace("\n", " ")[:500]
            evidence_lines.append(f"- `{evidence_id}` | 第 {page} 页 | {excerpt or '未提取到文本'}")
        evidence_block = "\n".join(evidence_lines) or "- 暂无可提取证据。"
        return (
            "## 研究问题\n\n- 待根据论文证据补充。\n\n"
            "## 核心方法\n\n- 待根据论文证据补充。\n\n"
            "## 数据集与实验设置\n\n- 待根据论文证据补充。\n\n"
            "## 主要结果\n\n- 待根据论文证据补充。\n\n"
            "## 作者讨论与局限\n\n- 未确认。\n\n"
            "## 作者提出的未来工作\n\n- 未确认。\n\n"
            "## 作者明确指出的 Gap\n\n- 未确认。\n\n"
            "## 基于证据推断的 Gap\n\n- 未确认。\n\n"
            f"## 可复用证据\n\n{evidence_block}\n\n"
            "## 与其他论文的关系\n\n- 暂无已验证关系。\n\n"
            "## 对 Idea 生成的提示\n\n- 仅使用上方可定位证据，证据不足处保持不确定。\n\n"
            "## 证据边界与未确认内容\n\n- 未确认。"
        )

    def rebuild_index(self) -> str:
        papers = self.list_papers()
        lines = ["# Research Wiki", "", f"- 论文数量: {len(papers)}", "", "## 论文目录", ""]
        if not papers:
            lines.append("- 暂无论文。")
        for paper in papers:
            lines.append(
                f"- [{paper.title}](papers/{paper.paper_id}/summary.md) "
                f"(`{paper.paper_id}`, [PDF](papers/{paper.paper_id}/source.pdf))"
            )
        return "\n".join(lines) + "\n"

    def query_pack(self, query: str, *, limit: int = 5, character_limit: int = 8000) -> str:
        query_terms = self._terms(query)
        ranked: list[tuple[int, WikiPaper, str]] = []
        for paper in self.list_papers():
            summary_path = self._workspace_path(paper.summary_relative_path)
            summary = summary_path.read_text(encoding="utf-8", errors="ignore")
            normalized = summary.casefold()
            score = sum(normalized.count(term.casefold()) for term in query_terms)
            ranked.append((score, paper, summary))
        ranked.sort(key=lambda item: (-item[0], item[1].paper_id))
        selected = ranked[:limit]
        per_paper_limit = max(
            450,
            min(5000, (character_limit - 900) // max(1, len(selected))),
        )
        lines = [
            "# Wiki Query Pack",
            "",
            f"- Query: {query}",
            f"- Selected papers: {len(selected)}",
            "",
        ]
        coverage = self.coverage_report()
        lines.extend(self._coverage_lines(coverage))
        lines.extend(self._evidence_matrix_lines())
        if not selected:
            lines.extend(["## Evidence Limitations", "", "- Wiki 中暂无可用论文。"])
            return "\n".join(lines) + "\n"
        for score, paper, summary in selected:
            excerpt = self._summary_excerpt(summary, query_terms, per_paper_limit)
            paper_block = [
                f"## {paper.paper_id}: {paper.title}",
                "",
                f"- Relevance score: {score}",
                f"- Summary: `{paper.summary_relative_path}`",
                f"- PDF: `{paper.wiki_pdf_relative_path}`",
                "",
                excerpt,
                "",
            ]
            candidate = "\n".join(lines + paper_block)
            if len(candidate) > character_limit - 300 and any(line.startswith("## P-") for line in lines):
                break
            lines.extend(paper_block)
        lines.extend(
            [
                *self._reference_inventory_lines(),
                "## Evidence Limitations",
                "",
                "- 只使用以上论文页面中已记录的 Evidence ID。",
                "- `unavailable`、`unresolved` 和 `skipped_limit` 参考文献只有引用元数据，不属于已阅读证据。",
                "- 模型推断不能替代论文作者明确陈述。",
            ]
        )
        return ("\n".join(lines) + "\n")[:character_limit]

    def _reference_inventory_lines(self) -> list[str]:
        catalog_path = self.wiki_root / "reference_catalog.json"
        if not catalog_path.exists():
            return []
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list) or not records:
            return []
        lines = [
                "## Metadata-only Reference Inventory",
            "",
            "以下为输入论文的直接参考文献目录。只有绑定 Paper ID 的条目已进入 Wiki 证据库。",
            "",
        ]
        for item in records:
            if not isinstance(item, dict):
                continue
            reference_id = str(item.get("reference_id") or "")
            title = str(item.get("resolved_title") or item.get("title") or item.get("raw_text") or "")
            status = str(item.get("status") or "unknown")
            paper_id = str(item.get("paper_id") or "")
            suffix = f" -> `{paper_id}`" if paper_id else " -> 仅元数据"
            lines.append(f"- `{reference_id}` | `{status}` | {title[:180]}{suffix}")
        lines.append("")
        return lines

    def coverage_report(self, *, minimum_papers: int | None = None) -> dict:
        minimum = minimum_papers or self.DEFAULT_IDEA_MINIMUM_PAPERS
        papers = self.list_papers()
        full_text_papers = [paper for paper in papers if self._is_complete_evidence_paper(paper)]
        catalog_records = self._reference_catalog_records()
        metadata_only = sum(
            1
            for record in catalog_records
            if str(record.get("status") or "") not in {"downloaded", "duplicate"}
        )
        return {
            "minimum_papers": minimum,
            "full_text_papers": len(full_text_papers),
            "full_text_paper_ids": [paper.paper_id for paper in full_text_papers],
            "incomplete_papers": len(papers) - len(full_text_papers),
            "downloaded_reference_papers": sum(
                1
                for record in catalog_records
                if str(record.get("status") or "") in {"downloaded", "duplicate"}
            ),
            "metadata_only_references": metadata_only,
            "reference_records": len(catalog_records),
            "ready_for_top_journal_candidate": len(full_text_papers) >= minimum,
        }

    def _is_complete_evidence_paper(self, paper: WikiPaper) -> bool:
        pdf_path = self._workspace_path(paper.wiki_pdf_relative_path)
        summary_path = self._workspace_path(paper.summary_relative_path)
        if not pdf_path.is_file() or not summary_path.is_file():
            return False
        try:
            summary = summary_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return bool(re.search(r"\bPE-[A-F0-9]{10}\b", summary, re.I))

    def _coverage_lines(self, coverage: dict) -> list[str]:
        ready = "是" if coverage["ready_for_top_journal_candidate"] else "否"
        lines = [
            "## Evidence Coverage",
            "",
            f"- 完整可读论文数: {coverage['full_text_papers']}",
            f"- Idea 定稿最低要求: {coverage['minimum_papers']}",
            f"- 仅元数据参考文献数: {coverage['metadata_only_references']}",
            f"- 缺少 PDF、总结或 Evidence ID 的论文数: {coverage['incomplete_papers']}",
            f"- 参考文献记录总数: {coverage['reference_records']}",
            f"- 是否达到顶刊候选门槛: {ready}",
            f"- 可作为证据的 Paper ID: {', '.join(f'`{item}`' for item in coverage['full_text_paper_ids']) or '暂无'}",
            "- 仅元数据或下载失败的参考文献不能作为已读证据。",
            "",
        ]
        return lines

    def _evidence_matrix_lines(self) -> list[str]:
        lines = [
            "## Cross-Paper Evidence Matrix",
            "",
            "| 论文 | 已解决问题 | 方法机制 | 已知局限 | 作者 Future Work | 可支持方向 | Evidence ID |",
            "|---|---|---|---|---|---|---|",
        ]
        for paper in self.list_papers():
            summary_path = self._workspace_path(paper.summary_relative_path)
            try:
                summary = summary_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            values = {
                "已解决问题": self._summary_section_excerpt(summary, "研究问题"),
                "方法机制": self._summary_section_excerpt(summary, "核心方法"),
                "已知局限": self._summary_section_excerpt(summary, "作者讨论与局限"),
                "作者 Future Work": self._summary_section_excerpt(summary, "作者提出的未来工作"),
                "可支持方向": self._summary_section_excerpt(summary, "对 Idea 生成的提示"),
            }
            evidence_ids = sorted(set(re.findall(r"\bPE-[A-F0-9]{10}\b", summary, re.I)))
            cells = [
                f"`{paper.paper_id}` {paper.title[:80]}",
                *[self._matrix_cell(values[key]) for key in ("已解决问题", "方法机制", "已知局限", "作者 Future Work", "可支持方向")],
                self._matrix_cell(", ".join(evidence_ids[:8]) or "未记录"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    def _reference_catalog_records(self) -> list[dict]:
        catalog_path = self.wiki_root / "reference_catalog.json"
        if not catalog_path.exists():
            return []
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def _summary_section_excerpt(summary: str, title: str, limit: int = 220) -> str:
        match = re.search(
            rf"(?ms)^##\s*{re.escape(title)}\s*\n(.*?)(?=^##\s|\Z)",
            summary,
        )
        if not match:
            return "未记录"
        excerpt = re.sub(r"\s+", " ", match.group(1)).strip()
        return excerpt[:limit] or "未确认"

    @staticmethod
    def _matrix_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")[:240]

    @staticmethod
    def _summary_excerpt(summary: str, query_terms: list[str], limit: int) -> str:
        matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", summary))
        if not matches:
            return summary[:limit].rstrip()
        priorities = {
            "作者讨论与局限": 140,
            "作者提出的未来工作": 140,
            "对 idea 生成的提示": 130,
            "可复用证据": 120,
            "研究问题": 110,
            "核心方法": 100,
            "主要结果": 80,
            "数据集与实验设置": 60,
            "基本信息": 20,
        }
        sections: list[tuple[int, int, str]] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(summary)
            block = summary[match.start() : end].strip()
            title = match.group(1).strip()
            normalized = block.casefold()
            score = priorities.get(title.casefold(), priorities.get(title, 30))
            score += sum(5 * normalized.count(term.casefold()) for term in query_terms)
            sections.append((score, -index, block))
        selected: list[str] = []
        used = 0
        for _, _, block in sorted(sections, reverse=True):
            if selected and used + len(block) + 2 > limit:
                continue
            selected.append(block)
            used += len(block) + 2
            if used >= limit:
                break
        return "\n\n".join(selected)[:limit].rstrip()

    def list_papers(self) -> list[WikiPaper]:
        papers: list[WikiPaper] = []
        for summary_path in sorted(self.papers_root.glob("*/summary.md")):
            content = summary_path.read_text(encoding="utf-8", errors="ignore")
            paper_id = summary_path.parent.name
            title_match = re.search(r"^#\s+(.+)$", content, re.M)
            source_match = re.search(r"^- 原始文件:\s*`([^`]+)`", content, re.M)
            papers.append(
                WikiPaper(
                    paper_id=paper_id,
                    source_relative_path=source_match.group(1) if source_match else "",
                    wiki_pdf_relative_path=f"wiki/papers/{paper_id}/source.pdf",
                    summary_relative_path=f"wiki/papers/{paper_id}/summary.md",
                    title=title_match.group(1).strip() if title_match else paper_id,
                )
            )
        return papers

    def write_idea_page(
        self,
        idea_id: str,
        *,
        final_idea: str,
        verification: str,
        source_paper_ids: Iterable[str],
    ) -> str:
        paper_ids = list(dict.fromkeys(source_paper_ids))
        content = (
            f"# Idea {idea_id}\n\n"
            "## 来源论文\n\n"
            + ("\n".join(f"- `{paper_id}`" for paper_id in paper_ids) or "- 暂无已绑定论文。")
            + "\n\n## 最终 Idea\n\n"
            + final_idea.strip()
            + "\n\n## 批评与核验\n\n"
            + verification.strip()
            + "\n"
        )
        relative_path = f"wiki/ideas/{idea_id}.md"
        target = self._workspace_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return relative_path

    def append_idea_relations(self, idea_id: str, paper_ids: Iterable[str]) -> str:
        relations_path = self.wiki_root / "relations.jsonl"
        existing = {
            line.strip()
            for line in relations_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        } if relations_path.exists() else set()
        for paper_id in dict.fromkeys(paper_ids):
            relation = json.dumps(
                {
                    "source": f"idea:{idea_id}",
                    "target": f"paper:{paper_id}",
                    "relation": "idea_based_on",
                    "evidence": "wiki query pack",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            existing.add(relation)
        relations_path.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""), encoding="utf-8")
        return "wiki/relations.jsonl"

    def append_citation_relations(
        self,
        source_paper_id: str,
        references: Iterable[dict],
    ) -> str:
        relations_path = self.wiki_root / "relations.jsonl"
        existing = {
            line.strip()
            for line in relations_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        } if relations_path.exists() else set()
        for reference in references:
            target_paper_id = str(reference.get("paper_id") or "")
            if not target_paper_id:
                continue
            relation = json.dumps(
                {
                    "source": f"paper:{source_paper_id}",
                    "target": f"paper:{target_paper_id}",
                    "relation": "cites",
                    "evidence": str(reference.get("reference_id") or "reference list"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            existing.add(relation)
        relations_path.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""), encoding="utf-8")
        return "wiki/relations.jsonl"

    def paper_ids_from_query_pack(self, query_pack: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"\bP-[A-F0-9]{12}\b", query_pack)))

    def _workspace_path(self, relative_path: str) -> Path:
        target = (self.workspace_root / relative_path).resolve()
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Wiki path escapes workspace: {relative_path}") from exc
        return target

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _terms(query: str) -> list[str]:
        english = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query)
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        chinese_bigrams = [
            chunk[index : index + 2]
            for chunk in chinese_chunks
            for index in range(max(0, len(chunk) - 1))
        ]
        return list(dict.fromkeys(english + chinese_chunks + chinese_bigrams))
