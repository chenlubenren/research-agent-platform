"""Post-retrieval research-direction discovery for the two-round /review flow.

The first (discovery) retrieval round returns papers that span several sub-branches.
This module clusters the *actually retrieved* papers into 3-5 research directions so
the user can pick one to deep-dive. Directions are always grounded in real paper ids;
if the LLM clustering fails we fall back to grouping by the discovery query facets so
the workflow never breaks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

from .connectors.scholar import PaperRecord
from .review_screening import parse_json_block

GenerateText = Callable[..., Awaitable[str]]
OnEvent = Callable[[str], None]

UNCLASSIFIED_TITLE = "未归类 / 需人工判断"
_UNCLASSIFIED_MARKERS = {"unclassifiable", "unclassified", "other", "misc", "n/a", "none", "未归类", "其他", "无法归类"}


@dataclass
class ReviewDirection:
    id: str
    title: str
    description: str = ""
    research_questions: list[str] = field(default_factory=list)
    representative_paper_ids: list[str] = field(default_factory=list)
    candidate_paper_ids: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    evidence_density: str = "moderate"
    why_distinct: str = ""


def directions_to_json(directions: list[ReviewDirection]) -> list[dict]:
    return [asdict(direction) for direction in directions]


def directions_from_json(payload: object) -> list[ReviewDirection]:
    directions: list[ReviewDirection] = []
    if not isinstance(payload, list):
        return directions
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("id", "")).strip():
            continue
        directions.append(
            ReviewDirection(
                id=str(item.get("id")).strip(),
                title=str(item.get("title", "")).strip() or str(item.get("id")).strip(),
                description=str(item.get("description", "")).strip(),
                research_questions=[str(q).strip() for q in item.get("research_questions", []) if str(q).strip()],
                representative_paper_ids=[str(p).strip() for p in item.get("representative_paper_ids", []) if str(p).strip()],
                candidate_paper_ids=[str(p).strip() for p in item.get("candidate_paper_ids", []) if str(p).strip()],
                query_terms=[str(t).strip() for t in item.get("query_terms", []) if str(t).strip()],
                evidence_density=str(item.get("evidence_density", "moderate")).strip() or "moderate",
                why_distinct=str(item.get("why_distinct", "")).strip(),
            )
        )
    return directions


def _candidate_block(papers: list[PaperRecord]) -> str:
    """Corpus-level view: title + abstract only (no matched_queries).

    Feeding the discovery queries back to the clustering model made it regroup papers by
    the query that fetched them ("query self-reinforcement"). Clustering on the actual
    title+abstract content instead yields genuine research themes.
    """

    lines: list[str] = []
    for paper in papers:
        abstract = (paper.abstract or "").strip().replace("\n", " ")
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(
            f"- id: {paper.paper_id}; title: {paper.title or 'Untitled'}; abstract: {abstract or 'n/a'}"
        )
    return "\n".join(lines)


def _validate_directions(
    directions: list[ReviewDirection],
    known_ids: set[str],
    *,
    max_directions: int,
    min_papers: int,
) -> list[ReviewDirection]:
    """Drop hallucinated ids/directions, merge heavy overlaps, mark sparse ones."""

    cleaned: list[ReviewDirection] = []
    for direction in directions:
        reps = [pid for pid in direction.representative_paper_ids if pid in known_ids]
        cands = [pid for pid in direction.candidate_paper_ids if pid in known_ids]
        # Every direction must be backed by real papers.
        cands = list(dict.fromkeys(cands + reps))
        reps = list(dict.fromkeys(reps)) or cands[:2]
        if not reps:
            continue
        direction.representative_paper_ids = reps[:5]
        direction.candidate_paper_ids = cands
        direction.evidence_density = (
            "sparse" if len(cands) < max(1, min_papers) else direction.evidence_density or "moderate"
        )
        cleaned.append(direction)

    # Merge directions whose candidate sets overlap heavily (Jaccard >= 0.8).
    merged: list[ReviewDirection] = []
    for direction in cleaned:
        target = None
        dcands = set(direction.candidate_paper_ids)
        for existing in merged:
            ecands = set(existing.candidate_paper_ids)
            union = dcands | ecands
            if union and len(dcands & ecands) / len(union) >= 0.8:
                target = existing
                break
        if target is None:
            merged.append(direction)
        else:
            target.candidate_paper_ids = list(dict.fromkeys(target.candidate_paper_ids + direction.candidate_paper_ids))
            target.representative_paper_ids = list(
                dict.fromkeys(target.representative_paper_ids + direction.representative_paper_ids)
            )[:5]
            target.query_terms = list(dict.fromkeys(target.query_terms + direction.query_terms))

    # Renumber to stable D1..Dn and cap.
    final = merged[: max(1, max_directions)]
    for index, direction in enumerate(final, start=1):
        direction.id = f"D{index}"
    return final


def _annotate_direction_membership(papers: list[PaperRecord], directions: list[ReviewDirection]) -> None:
    by_id = {paper.paper_id: paper for paper in papers}
    for direction in directions:
        for pid in direction.candidate_paper_ids:
            paper = by_id.get(pid)
            if paper is not None and direction.id not in paper.direction_ids:
                paper.direction_ids.append(direction.id)


def _normalize_theme(theme: str) -> str:
    return re.sub(r"\s+", " ", (theme or "").strip().casefold()).strip(" .,:;-—")


def _is_unclassified(theme: str) -> bool:
    norm = _normalize_theme(theme)
    return not norm or norm in _UNCLASSIFIED_MARKERS


def _build_label_prompt(papers: list[PaperRecord], *, topic: str, scope: str) -> dict[str, str]:
    system = (
        "You are organizing retrieved papers into research themes for a literature review. "
        "Read each paper's title and abstract and assign it to a concise, HUMAN-READABLE research theme "
        "(a noun phrase describing the research object/method, e.g. 'Memory retrieval policies for LLM agents'). "
        "Never use a raw search query or a single keyword as a theme. If a paper does not clearly fit any coherent "
        "theme, set its theme to 'unclassifiable'. Return ONLY a JSON array; each element is an object with keys "
        '"id" (the given paper id), "theme" (the human-readable theme or "unclassifiable"), and "reason" '
        "(one short clause on why it belongs there). Only score the ids provided; never invent ids."
    )
    user = (
        f"Review topic: {topic}\n\nScope:\n{scope or '(topic only)'}\n\n"
        f"Papers:\n{_candidate_block(papers)}\n\nReturn the JSON array now."
    )
    return {"system": system, "user": user}


async def _label_papers(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    generate_text: GenerateText,
    model: str | None,
    batch_size: int,
    timeout: float | None,
    on_event: OnEvent | None,
) -> tuple[dict[str, tuple[str, str]], bool]:
    """Return ({paper_id: (theme, reason)}, parsed_any). Missing ids stay unlabeled."""

    labels: dict[str, tuple[str, str]] = {}
    known = {paper.paper_id for paper in papers}
    parsed_any = False
    batch = max(1, batch_size)
    groups = [papers[i : i + batch] for i in range(0, len(papers), batch)]
    for index, group in enumerate(groups, start=1):
        prompt = _build_label_prompt(group, topic=topic, scope=scope)
        try:
            raw = await generate_text(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                model=model,
                temperature=0.1,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surface, do not swallow.
            if on_event:
                on_event(f"方向标注批次 {index}/{len(groups)} 失败：{exc.__class__.__name__}")
            continue
        parsed = parse_json_block(raw)
        rows = parsed if isinstance(parsed, list) else []
        if not rows:
            if on_event:
                on_event(f"方向标注批次 {index}/{len(groups)} 无法解析 JSON")
            continue
        parsed_any = True
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id", "")).strip()
            if pid not in known:
                continue
            theme = str(row.get("theme", "")).strip()
            reason = str(row.get("reason", "")).strip()
            labels[pid] = (theme, reason)
    return labels, parsed_any


def _cluster_from_labels(
    candidates: list[PaperRecord],
    labels: dict[str, tuple[str, str]],
    *,
    max_directions: int,
    min_papers: int,
) -> tuple[list[ReviewDirection], list[str]]:
    themed: dict[str, dict] = {}
    unclassified: list[PaperRecord] = []
    for paper in candidates:
        theme, reason = labels.get(paper.paper_id, ("", ""))
        if _is_unclassified(theme):
            unclassified.append(paper)
            continue
        key = _normalize_theme(theme)
        slot = themed.setdefault(key, {"title": theme.strip(), "papers": [], "reasons": []})
        slot["papers"].append(paper)
        if reason:
            slot["reasons"].append(reason)

    ordered = sorted(themed.values(), key=lambda slot: len(slot["papers"]), reverse=True)
    ordered = ordered[: max(1, max_directions)]
    directions: list[ReviewDirection] = []
    for slot in ordered:
        members = sorted(slot["papers"], key=lambda p: p.relevance_score, reverse=True)
        directions.append(
            ReviewDirection(
                id="D0",  # renumbered below
                title=slot["title"],
                description=slot["reasons"][0] if slot["reasons"] else "",
                research_questions=[],
                representative_paper_ids=[p.paper_id for p in members[:3]],
                candidate_paper_ids=[p.paper_id for p in members],
                query_terms=[],
                evidence_density="sparse" if len(members) < max(1, min_papers) else "moderate",
                why_distinct=f"聚焦「{slot['title']}」，与其他主题按研究对象/方法区分。",
            )
        )
    if not directions:
        return [], []
    for index, direction in enumerate(directions, start=1):
        direction.id = f"D{index}"
    recommended = [directions[0].id]
    # Append an explicit unclassifiable bucket (never recommended) so borderline papers are visible.
    if unclassified:
        members = sorted(unclassified, key=lambda p: p.relevance_score, reverse=True)
        directions.append(
            ReviewDirection(
                id=f"D{len(directions) + 1}",
                title=UNCLASSIFIED_TITLE,
                description="模型无法把这些论文稳定归入任一主题，保留以待人工判断，不建议直接深挖。",
                representative_paper_ids=[p.paper_id for p in members[:3]],
                candidate_paper_ids=[p.paper_id for p in members],
                evidence_density="sparse",
                why_distinct="不属于任何清晰主题的边界/离群论文。",
            )
        )
    return directions, recommended


async def discover_review_directions(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    generate_text: GenerateText,
    model: str | None = None,
    max_candidates: int = 40,
    max_directions: int = 5,
    min_papers: int = 2,
    batch_size: int = 12,
    timeout: float | None = None,
    on_event: OnEvent | None = None,
) -> tuple[list[ReviewDirection], list[str]]:
    """Corpus-level theme discovery: batch-label title+abstract, then cluster by theme.

    Returns (directions, recommended_ids). Empty directions => caller should fall back to
    query-facet grouping. A trailing "未归类" bucket collects papers the model could not
    assign, so boundary papers stay visible instead of being force-fit into a theme.
    """

    candidates = [paper for paper in papers if paper.screen_decision != "drop"][: max(1, max_candidates)]
    if not candidates:
        return [], []
    labels, parsed_any = await _label_papers(
        candidates,
        topic=topic,
        scope=scope,
        generate_text=generate_text,
        model=model,
        batch_size=batch_size,
        timeout=timeout,
        on_event=on_event,
    )
    if not parsed_any:
        return [], []
    directions, recommended = _cluster_from_labels(
        candidates, labels, max_directions=max_directions, min_papers=min_papers
    )
    if not directions:
        return [], []
    known_ids = {paper.paper_id for paper in candidates}
    directions = _validate_directions(
        directions, known_ids, max_directions=max_directions + 1, min_papers=min_papers
    )
    if not directions:
        return [], []
    valid_ids = {direction.id for direction in directions}
    recommended = [rid for rid in recommended if rid in valid_ids] or [directions[0].id]
    # Never recommend the unclassifiable bucket.
    recommended = [rid for rid in recommended if next((d.title for d in directions if d.id == rid), "") != UNCLASSIFIED_TITLE]
    if not recommended:
        non_bucket = [d for d in directions if d.title != UNCLASSIFIED_TITLE]
        recommended = [non_bucket[0].id] if non_bucket else []
    _annotate_direction_membership(candidates, directions)
    return directions, recommended


def directions_from_facets(
    papers: list[PaperRecord], *, max_directions: int = 5, min_papers: int = 2
) -> tuple[list[ReviewDirection], list[str]]:
    """Deterministic fallback: group papers by their matched discovery query."""

    candidates = [paper for paper in papers if paper.screen_decision != "drop"]
    groups: dict[str, list[PaperRecord]] = {}
    for paper in candidates:
        facet = (paper.matched_queries[0] if paper.matched_queries else "").strip() or "general"
        groups.setdefault(facet, []).append(paper)
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[: max(1, max_directions)]
    directions: list[ReviewDirection] = []
    for index, (facet, members) in enumerate(ordered, start=1):
        members.sort(key=lambda p: p.relevance_score, reverse=True)
        directions.append(
            ReviewDirection(
                id=f"D{index}",
                title=facet,
                description="按第一轮查询分面自动归组（LLM 方向归纳不可用时的回退）。",
                research_questions=[],
                representative_paper_ids=[p.paper_id for p in members[:3]],
                candidate_paper_ids=[p.paper_id for p in members],
                query_terms=[facet],
                evidence_density="sparse" if len(members) < max(1, min_papers) else "moderate",
                why_distinct=f"主要由查询「{facet}」命中。",
            )
        )
    _annotate_direction_membership(candidates, directions)
    recommended = [directions[0].id] if directions else []
    return directions, recommended


def _selected_directions(directions: list[ReviewDirection], selected_ids: list[str]) -> list[ReviewDirection]:
    by_id = {direction.id: direction for direction in directions}
    picked = [by_id[sid] for sid in selected_ids if sid in by_id]
    return picked or (directions[:1] if directions else [])


def focused_scope_markdown(
    directions: list[ReviewDirection],
    *,
    topic: str,
    selected_ids: list[str],
    mode: str = "focused",
    custom_text: str = "",
) -> str:
    lines = ["## Focused Retrieval Scope", "", f"- Mode: {mode}", f"- Topic: {topic}"]
    if mode == "panorama":
        lines.append("- Selected directions: (panorama — 不聚焦，沿用第一轮全部保留)")
        lines.append("")
        lines.append("选定方向: 全景综述，覆盖第一轮所有保留方向。")
        return "\n".join(lines) + "\n"
    picked = _selected_directions(directions, selected_ids)
    lines.append(f"- Selected directions: {', '.join(d.id for d in picked) or 'D1'}")
    lines.append("")
    if custom_text:
        lines.append("选定方向:")
        lines.append(custom_text.strip())
        lines.append("")
    else:
        lines.append("选定方向:")
        lines.extend(f"- [{d.id}] {d.title}" for d in picked)
        lines.append("")
    questions = list(dict.fromkeys(q for d in picked for q in d.research_questions))
    if questions:
        lines.append("核心研究问题:")
        lines.extend(f"- {q}" for q in questions[:8])
        lines.append("")
    terms = list(dict.fromkeys(t for d in picked for t in d.query_terms))
    if terms:
        lines.append("重点方法/术语:")
        lines.append("- " + ", ".join(terms[:20]))
        lines.append("")
    reps = list(dict.fromkeys(p for d in picked for p in d.representative_paper_ids))
    if reps:
        lines.append(f"代表论文: {', '.join(reps[:10])}")
        lines.append("")
    lines.append("明确排除: 与所选方向无关的其它子方向、仅泛泛相关而不涉及所选机制的论文。")
    return "\n".join(lines) + "\n"


def interpret_selection(feedback: str, directions: list[ReviewDirection]) -> dict:
    """Parse the user's free-text reply into a selection descriptor."""

    text = (feedback or "").strip()
    lowered = text.lower()
    if not text:
        return {"mode": "focused", "selected_ids": [], "custom_text": ""}
    panorama_markers = ("全景", "不聚焦", "panorama", "all directions", "都要", "全部方向", "综述全部")
    if any(marker in lowered or marker in text for marker in panorama_markers):
        return {"mode": "panorama", "selected_ids": [], "custom_text": ""}
    import re

    valid_ids = {direction.id.lower(): direction.id for direction in directions}
    found = []
    for token in re.findall(r"\bD\s?(\d{1,2})\b", text, re.I):
        key = f"d{token}"
        if key in valid_ids and valid_ids[key] not in found:
            found.append(valid_ids[key])
    if found:
        return {"mode": "focused", "selected_ids": found, "custom_text": ""}
    # No recognizable direction id -> treat the whole reply as a custom new direction.
    return {"mode": "focused", "selected_ids": [], "custom_text": text}


def build_directions_markdown(
    directions: list[ReviewDirection],
    recommended_ids: list[str],
    *,
    topic: str,
    fallback_used: bool = False,
) -> str:
    """Pass-1 artifact: direction cards + blocking decision + default Focused Scope."""

    recommended = [rid for rid in recommended_ids if any(d.id == rid for d in directions)]
    if not recommended and directions:
        recommended = [directions[0].id]
    rec_id = recommended[0] if recommended else "D1"
    rec_title = next((d.title for d in directions if d.id == rec_id), rec_id)

    lines = [
        "# Research Directions",
        "",
        f"- Topic: {topic}",
        f"- Directions discovered: {len(directions)}",
    ]
    if fallback_used:
        lines.append("- Note: LLM direction clustering unavailable; grouped by discovery query facets.")
    lines.append("")
    for direction in directions:
        lines.extend(
            [
                f"## {direction.id}: {direction.title}",
                "",
                (direction.description or "").strip(),
                f"- 候选论文: {len(direction.candidate_paper_ids)} ({direction.evidence_density})",
                f"- 代表论文: {', '.join(direction.representative_paper_ids) or 'n/a'}",
                f"- 常见术语: {', '.join(direction.query_terms) or 'n/a'}",
            ]
        )
        if direction.research_questions:
            lines.append("- 可回答的问题:")
            lines.extend(f"  - {q}" for q in direction.research_questions[:5])
        if direction.why_distinct:
            lines.append(f"- 区分点: {direction.why_distinct}")
        lines.append("")

    lines.extend(
        [
            "## Decision Required",
            "",
            "- Blocking: Yes",
            f"- 决策原因: 第一轮检索发现了 {len(directions)} 个有文献支撑、彼此不同的研究方向，"
            "需你确认继续做全景综述还是聚焦其中某个（些）方向做第二轮深度检索。",
            f"- Recommended Default: 深挖 {rec_id}「{rec_title}」，与原始研究问题匹配度最高且候选文献较充足。",
            f"- Option A: 采用推荐方向 {rec_id}，进入第二轮深度检索。",
            "- Option B: 回复方向编号（可多选，如「D2 和 D3」）或直接描述一个全新的方向。",
            "- Option C: 不聚焦，继续做全景综述。",
            "",
        ]
    )
    lines.append(
        focused_scope_markdown(directions, topic=topic, selected_ids=recommended, mode="focused")
    )
    return "\n".join(lines) + "\n"


def focused_query_plan(
    scope_meta: dict,
    directions: list[ReviewDirection],
    *,
    topic: str,
    limit: int = 10,
) -> list[str]:
    """Deterministic deep-dive query list built from the selected directions."""

    queries: list[str] = []
    if scope_meta.get("custom_text"):
        queries.append(scope_meta["custom_text"].strip()[:160])
    picked = _selected_directions(directions, scope_meta.get("selected_ids", []))
    for direction in picked:
        if direction.title:
            queries.append(f"{topic} {direction.title}".strip())
        for term in direction.query_terms[:4]:
            queries.append(term)
        for question in direction.research_questions[:2]:
            queries.append(question)
    if not queries:
        queries.append(topic)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(str(query).split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
    return deduped[: max(1, limit)]
