"""Layered keep/drop/uncertain screening for the /review pipeline.

The screener runs four explicit layers instead of one oversized LLM call:

  L1 hard_exclude      - deterministic exclusion terms + optional relevance floor.
  L2 semantic_rerank   - pluggable embedding similarity (falls back to lexical score).
  L3 llm_adjudicate    - small per-batch LLM JSON calls (never one 40-paper call).
  L4 uncertain queue   - boundary papers are marked ``uncertain`` rather than forced.

Every paper ends with a ``screen_tier`` in {core, transferable, background, exclude}
and a ``screen_decision`` in {keep, drop, uncertain}. There is NO silent fallback:
when a batch or the whole call fails, the reason is written to ``summary['screen_error']``
and pushed through the optional ``on_event`` callback so the caller can surface
"screening degraded: <reason>" in the retrieval quality report.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Awaitable, Callable, Sequence

from .config import config
from .connectors.scholar import PaperRecord

GenerateText = Callable[..., Awaitable[str]]
EmbedTexts = Callable[[list[str]], Awaitable[list[list[float]]]]
OnEvent = Callable[[str], None]

VALID_TIERS = {"core", "transferable", "background", "exclude", "unscreened"}
VALID_DECISIONS = {"keep", "drop", "uncertain"}
_COUNTED_TIERS = {"core", "transferable"}


def parse_json_block(text: str) -> object | None:
    """Best-effort extraction of a JSON object/array from an LLM response."""

    if not text:
        return None
    cleaned = text.strip()
    fence = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except (ValueError, TypeError):
                continue
    return None


def _coerce_rows(parsed: object) -> list[dict]:
    if isinstance(parsed, dict):
        for key in ("results", "papers", "screening", "decisions"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        return []
    return [row for row in parsed if isinstance(row, dict)]


def _as_float(value: object) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_keep(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"keep", "true", "yes", "y", "1", "include", "relevant"}:
            return True
        if token in {"drop", "false", "no", "n", "0", "exclude", "irrelevant"}:
            return False
    return None


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[;,]", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def parse_exclusion_terms(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [term.strip() for term in re.split(r"[;,\n]", raw) if term.strip()]
    return [str(term).strip() for term in raw if str(term).strip()]


# --- L1: hard exclusion ----------------------------------------------------


def hard_exclude(
    papers: list[PaperRecord],
    *,
    exclusion_terms: Sequence[str] = (),
    relevance_floor: float = 0.0,
) -> int:
    """Deterministically drop clearly out-of-scope papers before any LLM call."""

    terms = [term.casefold() for term in exclusion_terms if term]
    excluded = 0
    for paper in papers:
        text = f"{paper.title or ''} {paper.abstract or ''} {paper.venue or ''}".casefold()
        hits = [term for term in terms if term and term in text]
        if hits:
            paper.screen_tier = "exclude"
            paper.screen_decision = "drop"
            paper.violated_exclusion = list(dict.fromkeys(paper.violated_exclusion + hits))
            paper.screen_reason = paper.screen_reason or ("命中硬排除词：" + ", ".join(hits))
            excluded += 1
            continue
        if relevance_floor > 0 and paper.relevance_score < relevance_floor:
            paper.screen_tier = "exclude"
            paper.screen_decision = "drop"
            paper.screen_reason = paper.screen_reason or (
                f"词面相关度 {paper.relevance_score:.2f} 低于硬阈值 {relevance_floor:.2f}"
            )
            excluded += 1
    return excluded


# --- L2: semantic rerank ---------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def semantic_rerank(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    embed: EmbedTexts | None = None,
    on_event: OnEvent | None = None,
) -> tuple[list[PaperRecord], str]:
    """Order candidates by relevance. Returns (ordered_papers, mode).

    ``mode`` is 'embedding' when a pluggable embedder was used successfully, else
    'lexical' (a degradation to ``relevance_score`` ordering, never a hard failure).
    """

    candidates = [paper for paper in papers if paper.screen_tier != "exclude"]
    if not candidates:
        return [], "lexical"
    if embed is None:
        return sorted(candidates, key=lambda p: p.relevance_score, reverse=True), "lexical"
    query = f"{topic}\n{scope}".strip() or topic
    inputs = [query] + [
        f"{p.title or ''}. {(p.abstract or '')[:1500]}".strip() for p in candidates
    ]
    try:
        vectors = await embed(inputs)
    except Exception as exc:  # noqa: BLE001 - degrade to lexical, but surface it.
        if on_event:
            on_event(f"语义 rerank 不可用，退化为词面排序：{exc.__class__.__name__}: {exc}")
        return sorted(candidates, key=lambda p: p.relevance_score, reverse=True), "lexical"
    if not vectors or len(vectors) != len(inputs):
        if on_event:
            on_event("语义 rerank 返回向量数量不匹配，退化为词面排序")
        return sorted(candidates, key=lambda p: p.relevance_score, reverse=True), "lexical"
    query_vec = vectors[0]
    scored = [(_cosine(query_vec, vectors[index + 1]), paper) for index, paper in enumerate(candidates)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in scored], "embedding"


# --- L3: batched LLM adjudication ------------------------------------------


def _candidate_block(papers: list[PaperRecord]) -> str:
    lines: list[str] = []
    for paper in papers:
        abstract = (paper.abstract or "").strip().replace("\n", " ")
        if len(abstract) > 600:
            abstract = abstract[:600] + "..."
        year = paper.year if paper.year is not None else "unknown"
        lines.append(
            f"- id: {paper.paper_id}\n"
            f"  title: {paper.title or 'Untitled'}\n"
            f"  year: {year}\n"
            f"  venue: {paper.venue or 'unknown'}\n"
            f"  matched_queries: {', '.join(paper.matched_queries) or 'none'}\n"
            f"  abstract: {abstract or 'not available'}"
        )
    return "\n".join(lines)


def build_screening_prompt(papers: list[PaperRecord], *, topic: str, scope: str) -> dict[str, str]:
    system = (
        "You are a systematic-review screening assistant. For each candidate paper decide, strictly by the "
        "stated scope, whether to keep it and how central it is. Return ONLY a JSON array (no prose, no code "
        "fence). Each element must be an object with keys: "
        '"id" (the given paper id), '
        '"decision" (one of "keep", "drop", "uncertain"), '
        '"tier" (one of "core", "transferable", "background", "exclude"), '
        '"relevance" (0.0-1.0), '
        '"matched_inclusion" (array of the scope inclusion criteria this paper satisfies), '
        '"violated_exclusion" (array of exclusion criteria this paper violates), '
        '"reason" (one concise sentence in the scope language). '
        "Use tier=core for directly on-scope evidence, transferable for adjacent/analogous work worth citing, "
        "background for context-only, exclude for off-scope. Use decision=uncertain only for genuine borderline "
        "cases you cannot resolve from the abstract. Never invent papers or ids; only score the ids provided."
    )
    user = (
        f"Review topic: {topic}\n\n"
        f"Clarified/focused scope:\n{scope or '(no explicit scope; use the topic)'}\n\n"
        f"Candidates ({len(papers)}):\n{_candidate_block(papers)}\n\n"
        "Return the JSON array now."
    )
    return {"system": system, "user": user}


async def llm_adjudicate_batch(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    generate_text: GenerateText,
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> list[dict] | None:
    """Adjudicate one small batch. Returns parsed rows, or None on failure.

    ``max_tokens`` is set high enough that verbose JSON (one object per paper with a
    natural-language reason) is not truncated mid-array, which is the main cause of
    unparseable output. Roughly ~180 completion tokens per paper plus headroom.
    """

    if not papers:
        return []
    prompt = build_screening_prompt(papers, topic=topic, scope=scope)
    budget = max_tokens if max_tokens is not None else max(1536, 400 * len(papers) + 512)
    # Reasoning models spend a long ``reasoning_content`` before emitting JSON. A batch-sized
    # budget (~2–3k) that used to be enough for a fast chat model is consumed by reasoning,
    # the visible content comes back empty/truncated, and the batch is dumped as unparseable.
    # Floor against the configured completion budget so adjudication still gets an answer.
    if config.upstream_max_output_tokens > 0:
        budget = max(budget, config.upstream_max_output_tokens)
    raw = await generate_text(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        model=model,
        temperature=0.0,
        timeout=timeout,
        max_tokens=budget,
    )
    rows = _coerce_rows(parse_json_block(raw))
    return rows or None


async def _adjudicate_with_retry(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    generate_text: GenerateText,
    model: str | None,
    timeout: float | None,
) -> tuple[list[dict] | None, str | None]:
    """Adjudicate a batch; on failure split in half and retry once per sub-batch.

    Returns ``(rows, error)``. ``rows`` is None only when even single-paper calls fail.
    This rescues good papers that a single oversized/garbled JSON response would have
    otherwise dumped into the uncertain queue.
    """

    try:
        rows = await llm_adjudicate_batch(
            papers, topic=topic, scope=scope, generate_text=generate_text, model=model, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to caller, never swallowed.
        rows, err = None, f"{exc.__class__.__name__}: {exc}"
    else:
        err = None if rows else "unparseable or empty JSON"
    if rows:
        return rows, None
    if len(papers) <= 1:
        return None, err
    mid = len(papers) // 2
    left, left_err = await _adjudicate_with_retry(
        papers[:mid], topic=topic, scope=scope, generate_text=generate_text, model=model, timeout=timeout
    )
    right, right_err = await _adjudicate_with_retry(
        papers[mid:], topic=topic, scope=scope, generate_text=generate_text, model=model, timeout=timeout
    )
    merged: list[dict] = []
    for part in (left, right):
        if part:
            merged.extend(part)
    errors = [e for e in (left_err, right_err) if e]
    return (merged or None), ("; ".join(errors) if errors else None)


def _normalize_decision_tier(row: dict, *, keep: bool | None) -> tuple[str, str]:
    decision = str(row.get("decision", "")).strip().lower()
    tier = str(row.get("tier", "")).strip().lower()
    label = str(row.get("label", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        if keep is True:
            decision = "keep"
        elif keep is False:
            decision = "drop"
        else:
            decision = "keep"
    if tier not in VALID_TIERS:
        if label == "core":
            tier = "core"
        elif label == "boundary":
            tier = "transferable"
        else:
            tier = ""
    # Reconcile decision and tier so the gate reads a coherent signal.
    if decision == "drop":
        tier = "exclude"
    elif decision == "uncertain":
        if tier in {"", "exclude", "core"}:
            tier = "background"
    else:  # keep
        if tier in {"", "exclude"}:
            tier = "transferable"
    return decision, tier


def _apply_row(paper: PaperRecord, row: dict) -> None:
    keep = _as_keep(row.get("keep"))
    if keep is None:
        keep = _as_keep(row.get("decision"))
    decision, tier = _normalize_decision_tier(row, keep=keep)
    paper.screen_decision = decision
    paper.screen_tier = tier
    paper.screen_label = "core" if tier == "core" else ("boundary" if tier else paper.screen_label)
    relevance = _as_float(row.get("relevance"))
    if relevance is not None:
        paper.screen_relevance = max(0.0, min(1.0, relevance))
    inclusion = _as_str_list(row.get("matched_inclusion"))
    if inclusion:
        paper.matched_inclusion = list(dict.fromkeys(paper.matched_inclusion + inclusion))
    violation = _as_str_list(row.get("violated_exclusion"))
    if violation:
        paper.violated_exclusion = list(dict.fromkeys(paper.violated_exclusion + violation))
    reason = str(row.get("reason", "")).strip()
    if reason:
        paper.screen_reason = reason


def _mark_kept_background(paper: PaperRecord, reason: str) -> None:
    paper.screen_decision = "keep"
    paper.screen_tier = "background"
    paper.screen_reason = paper.screen_reason or reason


def _mark_unscreened(paper: PaperRecord, reason: str) -> None:
    """Overflow / missed-by-model papers are not judged and must not count as background."""

    paper.screen_decision = ""
    paper.screen_tier = "unscreened"
    paper.screen_reason = paper.screen_reason or reason


def _tier_counts(papers: list[PaperRecord]) -> dict[str, int]:
    counts = {"core": 0, "transferable": 0, "background": 0, "exclude": 0, "unscreened": 0}
    for paper in papers:
        if paper.screen_tier in counts:
            counts[paper.screen_tier] += 1
    return counts


async def screen_papers(
    papers: list[PaperRecord],
    *,
    topic: str,
    scope: str,
    generate_text: GenerateText,
    model: str | None = None,
    max_candidates: int = 40,
    batch_size: int = 8,
    max_concurrency: int = 1,
    hard_exclusion_terms: Sequence[str] = (),
    hard_relevance_floor: float = 0.0,
    embed: EmbedTexts | None = None,
    timeout: float | None = None,
    on_event: OnEvent | None = None,
) -> dict:
    """Run the four-layer screener, annotating ``papers`` in place.

    ``summary['screened']`` is False only when *no* LLM batch could be parsed (the caller
    then falls back to the lexical gate). Partial failures keep ``screened`` True but set
    ``summary['screen_error']`` so degradation is never silent.
    """

    summary: dict = {
        "screened": False,
        "candidate_count": len(papers),
        "hard_excluded": 0,
        "kept": 0,
        "dropped": 0,
        "uncertain": 0,
        "core": 0,
        "transferable": 0,
        "background": 0,
        "unscreened": 0,
        "screen_error": "",
        "rerank_mode": "lexical",
        "batches_total": 0,
        "batches_failed": 0,
    }
    if not papers:
        summary["screened"] = True
        return summary

    # L1
    summary["hard_excluded"] = hard_exclude(
        papers, exclusion_terms=hard_exclusion_terms, relevance_floor=hard_relevance_floor
    )
    if summary["hard_excluded"] and on_event:
        on_event(f"硬排除 {summary['hard_excluded']} 篇明显越界文献")

    # L2
    ordered, mode = await semantic_rerank(
        papers, topic=topic, scope=scope, embed=embed, on_event=on_event
    )
    summary["rerank_mode"] = mode

    sent = ordered[: max(1, max_candidates)]
    overflow = ordered[max(1, max_candidates) :]
    for paper in overflow:
        _mark_unscreened(paper, "候选超出单次筛选上限，未送入 LLM（不计入证据门）。")

    # L3 + L4
    batch = max(1, batch_size)
    batches = [sent[i : i + batch] for i in range(0, len(sent), batch)]
    summary["batches_total"] = len(batches)
    parsed_any = False
    failed = 0
    errors: list[str] = []

    # Batches are independent (each scores only its own papers), so adjudicate them concurrently
    # and cap fan-out with a semaphore to stay under the upstream LLM gateway's rate limit. This
    # turns the wall-clock from "sum of batch latencies" into "sum / concurrency" without changing
    # any decision. Results are applied in batch order below, so error accounting and the
    # progress log stay deterministic regardless of which batch finishes first.
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_batch(group: list[PaperRecord]) -> tuple[list[dict] | None, str | None]:
        async with semaphore:
            return await _adjudicate_with_retry(
                group,
                topic=topic,
                scope=scope,
                generate_text=generate_text,
                model=model,
                timeout=timeout,
            )

    batch_results = await asyncio.gather(*(_run_batch(group) for group in batches))
    for index, (group, (rows, batch_err)) in enumerate(zip(batches, batch_results), start=1):
        if batch_err and rows:
            # Partial rescue via split-retry: some sub-batches recovered.
            errors.append(f"batch {index} (partial): {batch_err}")
        elif batch_err:
            errors.append(f"batch {index}: {batch_err}")
        if not rows:
            failed += 1
            if not errors or not errors[-1].startswith(f"batch {index}"):
                errors.append(f"batch {index}: unparseable or empty JSON")
            for paper in group:
                paper.screen_decision = "uncertain"
                paper.screen_tier = paper.screen_tier or "background"
                paper.screen_reason = paper.screen_reason or "本批 LLM 裁定失败，进入待定队列（未计入证据门）。"
            if on_event:
                on_event(f"筛选批次 {index}/{len(batches)} 失败，{len(group)} 篇进入待定队列")
            continue
        parsed_any = True
        by_id = {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}
        for paper in group:
            row = by_id.get(paper.paper_id)
            if row is None:
                _mark_unscreened(paper, "模型未单独评估，未计入证据门。")
                continue
            _apply_row(paper, row)

    summary["batches_failed"] = failed
    if failed:
        summary["screen_error"] = "; ".join(errors)

    if not parsed_any:
        # Total failure: reset LLM-derived annotations so the caller uses the lexical gate.
        summary["screened"] = False
        if not summary["screen_error"]:
            summary["screen_error"] = "所有筛选批次均失败"
        for paper in sent:
            paper.screen_decision = ""
            paper.screen_tier = ""
            paper.screen_reason = ""
            paper.screen_label = ""
            paper.screen_relevance = None
            paper.matched_inclusion = []
        if on_event:
            on_event(f"LLM 精筛整体失败：{summary['screen_error']}，回退到词面阈值门")
        return summary

    summary["screened"] = True
    counts = _tier_counts(papers)
    summary["core"] = counts["core"]
    summary["transferable"] = counts["transferable"]
    summary["background"] = counts["background"]
    summary["unscreened"] = counts["unscreened"]
    summary["kept"] = sum(1 for paper in papers if paper.screen_decision == "keep")
    summary["dropped"] = sum(1 for paper in papers if paper.screen_decision == "drop")
    summary["uncertain"] = sum(1 for paper in papers if paper.screen_decision == "uncertain")
    return summary


def screening_gate_counts(
    papers: list[PaperRecord], *, screened: bool, lexical_threshold: float = 0.34
) -> tuple[int, int]:
    """Return (relevant_count, traceable_count) for the evidence gate.

    When screening succeeded the gate counts only ``core``/``transferable`` papers the LLM
    kept (``background``/``uncertain`` are recorded but not counted); otherwise it falls back
    to the strict lexical threshold. Traceable = admitted papers carrying an identifier.
    """

    if screened and any(paper.screen_decision for paper in papers):
        admitted = [
            paper
            for paper in papers
            if paper.screen_decision == "keep" and paper.screen_tier in _COUNTED_TIERS
        ]
    else:
        admitted = [paper for paper in papers if paper.relevance_score >= lexical_threshold]
    relevant = len(admitted)
    traceable = sum(paper.verification_status == "traceable_identifier" for paper in admitted)
    return relevant, traceable


def kept_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Papers admitted by screening; if nothing was screened, fall back to all papers."""

    if any(paper.screen_decision for paper in papers):
        return [
            paper
            for paper in papers
            if paper.screen_decision == "keep" and paper.screen_tier not in {"exclude", "unscreened"}
        ]
    return [paper for paper in papers if paper.screen_tier != "unscreened"]


def core_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Papers the screener judged directly on-scope (tier=core, kept)."""

    return [p for p in papers if p.screen_decision == "keep" and p.screen_tier == "core"]


def screening_markdown(papers: list[PaperRecord]) -> str:
    lines = ["## Screening", ""]
    scored = [paper for paper in papers if paper.screen_decision]
    unscreened = [paper for paper in papers if paper.screen_tier == "unscreened"]
    if not scored and not unscreened:
        lines.append("- LLM screening unavailable; lexical relevance filter used instead.")
        return "\n".join(lines) + "\n"
    counts = _tier_counts(papers)
    keep = [paper for paper in scored if paper.screen_decision == "keep"]
    drop = [paper for paper in scored if paper.screen_decision == "drop"]
    uncertain = [paper for paper in scored if paper.screen_decision == "uncertain"]
    lines.append(
        f"- Screened candidates: {len(scored)} "
        f"(keep {len(keep)}, drop {len(drop)}, uncertain {len(uncertain)}, unscreened {len(unscreened)})"
    )
    lines.append(
        f"- Tiers: core {counts['core']}, transferable {counts['transferable']}, "
        f"background {counts['background']}, exclude {counts['exclude']}, unscreened {counts['unscreened']}"
    )
    lines.append("")
    listed = list(scored) + [paper for paper in unscreened if paper not in scored]
    for paper in listed:
        relevance = f"{paper.screen_relevance:.2f}" if paper.screen_relevance is not None else "n/a"
        tier = f"/{paper.screen_tier}" if paper.screen_tier else ""
        title = paper.title or "Untitled"
        decision = paper.screen_decision or "unscreened"
        lines.append(
            f"- [{paper.paper_id}] {decision}{tier} (rel={relevance}): {title}"
        )
        if paper.screen_reason:
            lines.append(f"  - reason: {paper.screen_reason}")
    return "\n".join(lines) + "\n"
