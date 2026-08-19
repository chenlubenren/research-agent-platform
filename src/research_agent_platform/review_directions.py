"""Post-retrieval research-direction discovery for the two-round /review flow.

The first (discovery) retrieval round returns papers that span several sub-branches.
This module clusters the *actually retrieved* papers into 3-5 research directions so
the user can pick one to deep-dive. Directions are always grounded in real paper ids;
if the LLM clustering fails we fall back to grouping by the discovery query facets so
the workflow never breaks.

Round-2 queries are not a second copy of the discovery plan. Discovery uses Brief Qn
(topic / English / Chinese / named anchors / survey / foundational) for recall.
Focused retrieval searches *inside* the selected directions: named artifacts from
representative titles (PaperBench, MLE-bench), plus overlapping discovery anchor
queries. Direction theme titles are labels, not search strings — concatenating
``topic + English theme`` turns Crossref/arXiv into a bag-of-words on generic
tokens such as ``end-to-end`` / ``benchmark`` / ``systems``.
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


def _is_unclassified_direction(direction: ReviewDirection) -> bool:
    return direction.title == UNCLASSIFIED_TITLE or _is_unclassified(direction.title)


def _selectable_directions(directions: list[ReviewDirection]) -> list[ReviewDirection]:
    return [direction for direction in directions if not _is_unclassified_direction(direction)]


# Tokens that are useful as *theme labels* but poison keyword search (Crossref/arXiv
# treat the query as a bag of words). A focused query must keep at least one token
# that is not in this set after the original topic words are removed.
_GENERIC_SEARCH_TOKENS = {
    "a", "an", "and", "the", "for", "of", "on", "to", "in", "or", "vs", "via", "with",
    "from", "into", "using", "based", "towards", "toward", "its", "and", "new",
    "agent", "agents", "multi-agent", "multiagent", "llm", "llms", "ai",
    "language", "model", "models", "system", "systems", "framework", "frameworks",
    "benchmark", "benchmarks", "bench", "evaluation", "evaluating", "evaluate",
    "protocol", "metrics", "metric", "review", "survey", "analysis", "study",
    "scientific", "science", "research", "researcher", "researchers",
    "machine", "learning", "deep", "neural", "artificial", "intelligence",
    "autonomous", "end-to-end", "end", "open-ended", "fully", "automated",
    "experiment", "experimentation", "experimental", "paper", "papers",
    "replication", "reproducibility", "reproducible", "scientist", "scientists",
    "discovery", "engineering", "performance",
    "评测", "基准", "综述", "智能体", "科研", "系统", "多智能体", "大模型", "文献", "进展",
}
_GENERIC_ZH_FRAGMENTS = (
    "评测基准", "科研系统", "多智能体", "智能体", "大模型", "评测", "基准", "综述",
    "科研", "系统", "文献", "进展", "研究",
)


def _query_tokens(text: str) -> list[str]:
    english = re.findall(r"[a-z0-9][a-z0-9+-]{1,}", (text or "").casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
    return [*english, *chinese]


def _is_generic_token(token: str) -> bool:
    norm = (token or "").casefold().strip(".-")
    if not norm or norm in _GENERIC_SEARCH_TOKENS:
        return True
    if re.fullmatch(r"[\u4e00-\u9fff]+", token or ""):
        stripped = token
        for frag in sorted(_GENERIC_ZH_FRAGMENTS, key=len, reverse=True):
            stripped = stripped.replace(frag, "")
        stripped = re.sub(r"[的与和及]", "", stripped)
        return not stripped.strip()
    return False


def is_generic_search_query(query: str, *, topic: str = "") -> bool:
    """True when a query has no distinctive token beyond the original review topic."""

    if re.search(
        r"\b(?!benchmarks?\b)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*-?[Bb]ench(?:mark)?s?\b",
        query or "",
    ):
        return False
    if re.search(r"\b(?:The\s+)?(?:Jr\.?\s+)?AI\s+Scientist\b", query or "", re.I):
        return False
    topic_tokens = {tok for tok in _query_tokens(topic) if tok}
    distinctive = [
        tok
        for tok in _query_tokens(query)
        if not _is_generic_token(tok) and tok not in topic_tokens
    ]
    return len(distinctive) < 1


def _looks_like_named_seed(text: str) -> bool:
    cleaned = " ".join((text or "").split()).strip(" .,:;-—\"'`")
    if not cleaned or len(cleaned) > 70:
        return False
    words = cleaned.split()
    if not words or len(words) > 8:
        return False
    if re.search(r"\b(?:The\s+)?(?:Jr\.?\s+)?AI\s+Scientist\b", cleaned, re.I):
        return True
    if is_generic_search_query(cleaned):
        return False
    if re.search(r"[A-Za-z]+-?[Bb]ench", cleaned) and not re.search(r"\bbenchmarks?\b", cleaned, re.I):
        return True
    if re.search(r"\d", cleaned):
        return True
    if "-" in cleaned and not cleaned.casefold().startswith("end-to-end"):
        return True
    if re.search(r"\bThe\s+", cleaned):
        return True
    if re.search(r"[A-Z]{2,}|[a-z][A-Z]", cleaned):
        return True
    return len(words) == 1 and cleaned[0].isupper() and len(cleaned) >= 4


def extract_named_seeds(title: str) -> list[str]:
    """Pull searchable proper names (PaperBench, MLE-bench, The AI Scientist) from a title."""

    seeds: list[str] = []
    raw = (title or "").strip()
    if not raw:
        return seeds
    if ":" in raw:
        head = raw.split(":", 1)[0].strip()
        ai = re.search(r"\b(?:The\s+)?(?:Jr\.?\s+)?AI\s+Scientist\b", head, re.I)
        if ai:
            seeds.append(" ".join(ai.group(0).split()))
        elif _looks_like_named_seed(head):
            seeds.append(" ".join(head.split()))
    for match in re.finditer(
        r"\b(?!benchmarks?\b)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*-?[Bb]ench(?:mark)?s?\b",
        raw,
    ):
        seeds.append(match.group(0))
    for match in re.finditer(r"\b(?:The\s+)?(?:Jr\.?\s+)?AI\s+Scientist\b", raw, re.I):
        seeds.append(" ".join(match.group(0).split()))
    deduped: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        key = seed.casefold()
        if key not in seen:
            deduped.append(seed)
            seen.add(key)
    return deduped


def _compact_theme_query(*parts: str) -> str:
    distinctive: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in _query_tokens(part):
            if _is_generic_token(token) or token in seen:
                continue
            seen.add(token)
            distinctive.append(token)
            if len(distinctive) >= 6:
                break
        if len(distinctive) >= 6:
            break
    if len(distinctive) < 2:
        return ""
    return " ".join(distinctive)


def derive_direction_query_terms(papers: list[PaperRecord], *, extra_text: str = "") -> list[str]:
    """Deterministic focused queries from papers already clustered into a direction."""

    seeds: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen or is_generic_search_query(cleaned):
            return
        seen.add(key)
        seeds.append(cleaned)

    for paper in papers:
        for seed in extract_named_seeds(paper.title or ""):
            _add(seed)
    if extra_text:
        for seed in extract_named_seeds(extra_text):
            _add(seed)
        compact = _compact_theme_query(extra_text)
        if compact:
            _add(compact)
    if not seeds:
        compact = _compact_theme_query(*(paper.title or "" for paper in papers), extra_text)
        if compact:
            _add(compact)
    return seeds


def _sentence_like(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if "?" in stripped or "？" in stripped:
        return True
    return len(stripped.split()) > 12


def _discovery_query_reusable(query: str, *, topic: str, seeds: list[str]) -> bool:
    cleaned = " ".join((query or "").split()).strip()
    if not cleaned or cleaned.casefold() == (topic or "").strip().casefold():
        return False
    if is_generic_search_query(cleaned, topic=topic):
        return False
    suffixes = ("review", "systematic review", "综述", "研究进展")
    for suffix in suffixes:
        if cleaned.casefold() == f"{topic} {suffix}".casefold():
            return False
    blob = cleaned.casefold()
    named_hits = [seed for seed in seeds if len(seed) >= 4 and seed.casefold() in blob]
    if any(re.search(r"bench", seed, re.I) for seed in named_hits):
        return True
    return len(named_hits) >= 2


def _direction_cluster_pool(papers: list[PaperRecord], max_candidates: int) -> list[PaperRecord]:
    """Prefer LLM-judged core/transferable; fill with judged background. Never include unscreened."""

    limit = max(1, max_candidates)
    if not any(paper.screen_decision or paper.screen_tier for paper in papers):
        return papers[:limit]
    counted: list[PaperRecord] = []
    background: list[PaperRecord] = []
    for paper in papers:
        if paper.screen_decision == "drop" or paper.screen_tier in {"exclude", "unscreened"}:
            continue
        if paper.screen_decision != "keep":
            continue
        if paper.screen_tier in {"core", "transferable"}:
            counted.append(paper)
        elif paper.screen_tier == "background":
            background.append(paper)
    return (counted + background)[:limit]


def _recommend_direction_ids(directions: list[ReviewDirection], papers: list[PaperRecord]) -> list[str]:
    by_id = {paper.paper_id: paper for paper in papers}
    ranked: list[tuple[int, int, int, str]] = []
    for direction in _selectable_directions(directions):
        core = transferable = 0
        for pid in direction.candidate_paper_ids:
            paper = by_id.get(pid)
            if paper is None:
                continue
            if paper.screen_tier == "core":
                core += 1
            elif paper.screen_tier == "transferable":
                transferable += 1
        ranked.append((core, transferable, len(direction.candidate_paper_ids), direction.id))
    ranked.sort(reverse=True)
    return [ranked[0][3]] if ranked else []


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
                query_terms=derive_direction_query_terms(
                    members, extra_text=f"{slot['title']} {slot['reasons'][0] if slot['reasons'] else ''}"
                ),
                evidence_density="sparse" if len(members) < max(1, min_papers) else "moderate",
                why_distinct=f"聚焦「{slot['title']}」，与其他主题按研究对象/方法区分。",
            )
        )
    if not directions:
        return [], []
    for index, direction in enumerate(directions, start=1):
        direction.id = f"D{index}"
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
    recommended = _recommend_direction_ids(directions, candidates)
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

    candidates = _direction_cluster_pool(papers, max_candidates)
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
    directions, _clustered = _cluster_from_labels(
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
    recommended = _recommend_direction_ids(directions, candidates)
    _annotate_direction_membership(candidates, directions)
    return directions, recommended


def directions_from_facets(
    papers: list[PaperRecord], *, max_directions: int = 5, min_papers: int = 2
) -> tuple[list[ReviewDirection], list[str]]:
    """Deterministic fallback: group papers by their matched discovery query."""

    candidates = _direction_cluster_pool(papers, max_candidates=max(40, max_directions * 10))
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
    recommended = _recommend_direction_ids(directions, candidates)
    return directions, recommended


def _selected_directions(directions: list[ReviewDirection], selected_ids: list[str]) -> list[ReviewDirection]:
    selectable = _selectable_directions(directions)
    by_id = {direction.id: direction for direction in selectable}
    picked = [by_id[sid] for sid in selected_ids if sid in by_id]
    return picked or (selectable[:1] if selectable else [])


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

    selectable_ids = {direction.id.lower(): direction.id for direction in _selectable_directions(directions)}
    found = []
    saw_direction_token = False
    for token in re.findall(r"\bD\s?(\d{1,2})\b", text, re.I):
        saw_direction_token = True
        key = f"d{token}"
        if key in selectable_ids and selectable_ids[key] not in found:
            found.append(selectable_ids[key])
    if found:
        return {"mode": "focused", "selected_ids": found, "custom_text": ""}
    if saw_direction_token:
        # User only named the unclassifiable bucket (or unknown ids); fall back to recommended.
        return {"mode": "focused", "selected_ids": [], "custom_text": ""}
    # No recognizable direction id -> treat the whole reply as a custom new direction.
    return {"mode": "focused", "selected_ids": [], "custom_text": text}


def build_directions_markdown(
    directions: list[ReviewDirection],
    recommended_ids: list[str],
    *,
    topic: str,
    fallback_used: bool = False,
    core_count: int | None = None,
    coverage_state: str = "",
) -> str:
    """Pass-1 artifact: direction cards + blocking decision + default Focused Scope.

    ``core_count`` and ``coverage_state`` are surfaced in the decision block: directions
    clustered without any core evidence describe whatever the transferable tier happened
    to contain, which is exactly when the user needs to be told before approving.
    """

    recommended = [rid for rid in recommended_ids if any(d.id == rid for d in _selectable_directions(directions))]
    if not recommended:
        recommended = _recommend_direction_ids(directions, [])
        selectable = _selectable_directions(directions)
        if not recommended and selectable:
            recommended = [selectable[0].id]
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
    if core_count == 0:
        lines.append(
            "- ⚠ 核心层为空（core=0）：以下方向由 transferable/background 层聚出，"
            "描述的是「相邻或被评测对象」而不是范围卡定义的核心研究对象，可能偏离你的原始问题。"
        )
    if coverage_state == "degraded":
        lines.append(
            "- ⚠ 覆盖降级（coverage=degraded）：本轮有关键索引失效或被领域判定关闭，"
            "方向清单可能缺失整类文献。"
        )
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
        if _is_unclassified_direction(direction):
            lines.append("- 可选: 否（占位清单，不进入第二轮检索，请勿回复该编号）")
        lines.append("")

    reason = (
        f"- 决策原因: 第一轮检索发现了 {len(directions)} 个有文献支撑、彼此不同的研究方向，"
        "需你确认继续做全景综述还是聚焦其中某个（些）方向做第二轮深度检索。"
    )
    if core_count == 0:
        reason = (
            f"- 决策原因: 第一轮检索的核心层为空（core=0），下列 {len(directions)} 个方向只能由"
            "相邻文献聚出；直接进入第二轮会把综述问题替换成这些方向本身。请确认方向，"
            "或直接描述你要的核心对象让第二轮重新定向。"
        )
    lines.extend(
        [
            "## Decision Required",
            "",
            "- Blocking: Yes",
            reason,
            f"- Recommended Default: 深挖 {rec_id}「{rec_title}」，与原始研究问题匹配度最高且候选文献较充足。",
            f"- Option A: 采用推荐方向 {rec_id}，进入第二轮深度检索。",
            "- Option B: 回复可选方向编号（可多选，如「D2 和 D3」）或直接描述一个全新的方向。未归类编号不可选。",
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
    papers: list[PaperRecord] | None = None,
    discovery_queries: list[str] | None = None,
) -> list[str]:
    """Build the second-round (focused) query list.

    Discovery queries map the field. Focused queries hunt siblings of papers the user
    already accepted in a direction: named artifacts, then overlapping round-1 anchor
    queries. Never concatenate ``topic + direction.title`` — those titles are cluster
    labels and expand to generic tokens on scholarly APIs.
    """

    queries: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = " ".join(str(raw or "").split()).strip()[:160]
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen or key == (topic or "").strip().casefold():
            return
        if any(key != existing and key in existing for existing in seen if len(key) >= 8):
            return
        if is_generic_search_query(cleaned, topic=topic):
            return
        seen.add(key)
        queries.append(cleaned)

    custom = str(scope_meta.get("custom_text") or "").strip()
    if custom:
        _push(custom)
    selected_ids = [str(item).strip() for item in (scope_meta.get("selected_ids") or []) if str(item).strip()]
    # A free-text new direction should not silently also search the recommended D1 theme.
    picked = [] if (custom and not selected_ids) else _selected_directions(directions, selected_ids)
    papers_by_id = {paper.paper_id: paper for paper in (papers or []) if paper.paper_id}

    collected_seeds: list[str] = []
    for direction in picked:
        if _is_unclassified_direction(direction):
            continue
        terms = [term for term in direction.query_terms if str(term).strip()]
        if not terms:
            members = [
                papers_by_id[pid]
                for pid in list(dict.fromkeys(direction.representative_paper_ids + direction.candidate_paper_ids))
                if pid in papers_by_id
            ]
            terms = derive_direction_query_terms(
                members, extra_text=f"{direction.title} {direction.description}"
            )
        for term in terms:
            _push(term)
            collected_seeds.extend(extract_named_seeds(term) or ([term] if not is_generic_search_query(term, topic=topic) else []))
        for question in direction.research_questions[:2]:
            if not _sentence_like(question):
                _push(question)

    for query in discovery_queries or []:
        if _discovery_query_reusable(query, topic=topic, seeds=collected_seeds or queries):
            _push(query)

    unique_seeds = list(dict.fromkeys(s for s in collected_seeds if s))
    if len(unique_seeds) >= 2:
        _push(" ".join(unique_seeds[:6]))

    if not queries:
        for direction in picked:
            if _is_unclassified_direction(direction):
                continue
            _push(_compact_theme_query(direction.title, direction.description))
    return queries[: max(1, limit)]
