from __future__ import annotations

import asyncio

from research_agent_platform.connectors.scholar import (
    PaperRecord,
    ScholarSearchService,
    merge_paper_records,
)
from research_agent_platform.review_clarify import (
    build_clarify_draft,
    build_clarify_pass2,
    clarify_pass1_markdown,
    default_scope_card,
)
from research_agent_platform.review_directions import (
    UNCLASSIFIED_TITLE,
    directions_from_facets,
    discover_review_directions,
    focused_query_plan,
    focused_scope_markdown,
    is_generic_search_query,
    interpret_selection,
)
from research_agent_platform.review_screening import (
    hard_exclude,
    kept_papers,
    parse_json_block,
    screen_papers,
    screening_gate_counts,
    semantic_rerank,
)


def _gen(response: str):
    async def gen(*, system_prompt: str, user_prompt: str, model=None, temperature=None, timeout=None, max_tokens=None) -> str:
        return response

    return gen


def _gen_map(mapping: dict[str, str], default: str = ""):
    """Return responses keyed by a substring found in the user prompt (per-batch control)."""

    async def gen(*, system_prompt: str, user_prompt: str, model=None, temperature=None, timeout=None, max_tokens=None) -> str:
        for needle, response in mapping.items():
            if needle in user_prompt:
                return response
        return default

    return gen


async def _boom(*, system_prompt: str, user_prompt: str, model=None, temperature=None, timeout=None, max_tokens=None) -> str:
    raise RuntimeError("upstream unavailable")


def _paper(pid: str, title: str, *, abstract: str = "", queries=None, relevance: float = 0.5, doi: str = "") -> PaperRecord:
    record = PaperRecord(
        title=title,
        year=2024,
        abstract=abstract or title,
        authors=["A"],
        url=f"https://doi.org/{doi}" if doi else "https://example.org/" + pid,
        venue="Venue",
        citation_count=3,
        sources=["OpenAlex"],
        identifiers={"doi": doi} if doi else {},
        paper_id=pid,
        relevance_score=relevance,
        matched_queries=list(queries or []),
        verification_status="traceable_identifier",
    )
    return record


# --- Clarify ---------------------------------------------------------------


def test_clarify_pass1_emits_blocking_block_and_default_scope() -> None:
    md = clarify_pass1_markdown(
        "LLM Agent 长期记忆",
        ["只研究基于 LLM 的智能体吗？", "重点关注哪些记忆环节？"],
        default_scope_card("LLM Agent 长期记忆"),
    )
    assert "## Decision Required" in md
    assert "Blocking: Yes" in md
    assert "决策原因:" in md
    assert "Recommended Default:" in md
    assert "Option A:" in md and "Option B:" in md
    assert "## Clarified Scope" in md
    # The two dimensions must both be represented so downstream parsing is stable.
    assert "记忆" in md


def test_clarify_pass2_finalizes_scope_without_blocking() -> None:
    prior = clarify_pass1_markdown("topic", ["q1"], default_scope_card("topic"))
    llm_out = "## Clarified Scope\n\n- 研究主题: topic\n- 时间范围: 2022 年后\n"
    md = asyncio.run(
        build_clarify_pass2("topic", feedback="限定 2022 年后", prior_text=prior, generate_text=_gen(llm_out))
    )
    assert "## Clarified Scope" in md
    assert "## Decision Required" not in md
    assert "2022" in md


def test_clarify_pass2_strips_stray_blocking_and_falls_back_on_error() -> None:
    prior = clarify_pass1_markdown("topic", ["q1"], default_scope_card("topic"))
    # LLM accidentally re-emits a blocking section: it must be stripped.
    stray = "## Decision Required\n\n- Blocking: Yes\n\n## Clarified Scope\n\n- 研究主题: topic\n"
    md = asyncio.run(build_clarify_pass2("topic", feedback="x", prior_text=prior, generate_text=_gen(stray)))
    assert "## Decision Required" not in md
    assert "## Clarified Scope" in md
    # On upstream failure we still return a Clarified Scope carrying the user's answer.
    fallback = asyncio.run(
        build_clarify_pass2("topic", feedback="必须包含隐私", prior_text=prior, generate_text=_boom)
    )
    assert "## Clarified Scope" in fallback
    assert "必须包含隐私" in fallback
    assert "## Decision Required" not in fallback


def test_clarify_draft_prefills_scope_card_from_topic() -> None:
    payload = (
        '{"questions": ["是否只要 LLM 驱动的科研 Agent？"], '
        '"scope_card": {"概念边界": "排除编队控制等传统 MAS", '
        '"时间范围": "2023 至今", "对比对象": "MLAgentBench / CORE-Bench"}}'
    )
    questions, card = asyncio.run(build_clarify_draft("多智能体科研系统Benchmark", generate_text=_gen(payload)))
    assert questions and "科研" in questions[0]
    assert "传统 MAS" in card["概念边界"]
    assert "MLAgentBench" in card["对比对象"]
    assert card["研究主题"] == "多智能体科研系统Benchmark"


# --- Direction discovery ---------------------------------------------------


def test_discover_directions_clusters_themes_and_buckets_unclassifiable() -> None:
    papers = [
        _paper("P001", "Memory storage", queries=["memory storage"]),
        _paper("P002", "Hierarchical memory", queries=["memory storage"]),
        _paper("P003", "Memory retrieval policy", queries=["memory retrieval"]),
        _paper("P004", "Context selection", queries=["memory retrieval"]),
        _paper("P005", "Random unrelated thing", queries=["misc"]),
    ]
    # New schema: per-paper human-readable theme labels (ids never invented by the model).
    response = (
        "[\n"
        '{"id": "P001", "theme": "Agent memory storage", "reason": "stores memories"},\n'
        '{"id": "P002", "theme": "Agent memory storage", "reason": "hierarchical store"},\n'
        '{"id": "P003", "theme": "Agent memory retrieval", "reason": "retrieval policy"},\n'
        '{"id": "P004", "theme": "Agent memory retrieval", "reason": "context selection"},\n'
        '{"id": "P005", "theme": "unclassifiable", "reason": "off topic"},\n'
        '{"id": "P999", "theme": "Ghost theme", "reason": "hallucinated id ignored"}\n'
        "]"
    )
    directions, recommended = asyncio.run(
        discover_review_directions(
            papers, topic="LLM memory", scope="", generate_text=_gen(response),
            max_candidates=40, max_directions=5, min_papers=2,
        )
    )
    titles = [d.title for d in directions]
    assert "Agent memory storage" in titles
    assert "Agent memory retrieval" in titles
    # Unclassifiable bucket is present but never recommended, and hallucinated ids are dropped.
    assert any(d.title == UNCLASSIFIED_TITLE for d in directions)
    all_ids = {pid for d in directions for pid in d.candidate_paper_ids}
    assert "P999" not in all_ids
    rec_titles = [next(d.title for d in directions if d.id == rid) for rid in recommended]
    assert UNCLASSIFIED_TITLE not in rec_titles and recommended


def test_discover_directions_skips_unscreened_and_recommends_core() -> None:
    papers = [
        _paper("P001", "meta audit", queries=["q"]),
        _paper("P002", "FML-bench", queries=["q"]),
        _paper("P003", "noise", queries=["q"]),
    ]
    papers[0].screen_decision = "keep"
    papers[0].screen_tier = "background"
    papers[1].screen_decision = "keep"
    papers[1].screen_tier = "core"
    papers[2].screen_tier = "unscreened"
    response = (
        '[{"id": "P001", "theme": "Transparency audits", "reason": "meta"},'
        ' {"id": "P002", "theme": "ML research automation benchmarks", "reason": "end-to-end"}]'
    )
    directions, recommended = asyncio.run(
        discover_review_directions(
            papers, topic="t", scope="", generate_text=_gen(response),
            max_candidates=30, max_directions=5, min_papers=1,
        )
    )
    all_ids = {pid for d in directions for pid in d.candidate_paper_ids}
    assert "P003" not in all_ids
    rec_titles = [next(d.title for d in directions if d.id == rid) for rid in recommended]
    assert rec_titles == ["ML research automation benchmarks"]
    ml = next(d for d in directions if d.title == "ML research automation benchmarks")
    assert any("FML-bench" in term for term in ml.query_terms)


def test_interpret_selection_ignores_unclassified_bucket() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    directions = [
        ReviewDirection(id="D1", title="ML research automation benchmarks"),
        ReviewDirection(id="D2", title=UNCLASSIFIED_TITLE),
    ]
    assert interpret_selection("D1 和 D2", directions)["selected_ids"] == ["D1"]
    only_bucket = interpret_selection("D2", directions)
    assert only_bucket["selected_ids"] == [] and only_bucket["custom_text"] == ""


def test_focused_query_plan_skips_unclassified_title() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    directions = [
        ReviewDirection(id="D1", title="FML-bench", query_terms=["FML-bench scientific research agent"]),
        ReviewDirection(id="D2", title=UNCLASSIFIED_TITLE),
    ]
    queries = focused_query_plan(
        {"mode": "focused", "selected_ids": ["D1", "D2"], "custom_text": ""},
        directions,
        topic="多智能体科研系统的Benchmark",
        limit=10,
    )
    assert queries
    assert all(UNCLASSIFIED_TITLE not in q for q in queries)
    assert any("FML-bench" in q for q in queries)


def test_direction_discovery_falls_back_to_query_facets() -> None:
    papers = [
        _paper("P001", "A", queries=["memory storage"]),
        _paper("P002", "B", queries=["memory storage"]),
        _paper("P003", "C", queries=["memory retrieval"]),
    ]
    directions, recommended = asyncio.run(
        discover_review_directions(papers, topic="t", scope="", generate_text=_gen("not json at all"))
    )
    assert directions == [] and recommended == []
    fb_directions, fb_recommended = directions_from_facets(papers, max_directions=5, min_papers=2)
    assert len(fb_directions) == 2
    assert fb_recommended == [fb_directions[0].id]
    assert {pid for d in fb_directions for pid in d.candidate_paper_ids} == {"P001", "P002", "P003"}


def test_interpret_selection_handles_ids_panorama_and_custom() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    directions = [ReviewDirection(id="D1", title="a"), ReviewDirection(id="D2", title="b"), ReviewDirection(id="D3", title="c")]
    assert interpret_selection("选择 D2 和 D3", directions) == {
        "mode": "focused",
        "selected_ids": ["D2", "D3"],
        "custom_text": "",
    }
    assert interpret_selection("不聚焦，继续做全景综述", directions)["mode"] == "panorama"
    custom = interpret_selection("我想研究长期记忆中的隐私泄漏", directions)
    assert custom["mode"] == "focused" and custom["selected_ids"] == [] and "隐私" in custom["custom_text"]


def test_focused_query_plan_and_scope_from_selection() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    directions = [
        ReviewDirection(
            id="D1", title="Shared memory", query_terms=["shared memory", "memory consistency"],
            research_questions=["how to sync?"], representative_paper_ids=["P007"],
        )
    ]
    meta = {"mode": "focused", "selected_ids": ["D1"], "custom_text": ""}
    queries = focused_query_plan(meta, directions, topic="LLM agent", limit=10)
    assert any("shared memory" in q.lower() for q in queries)
    assert all("how to sync" not in q.lower() for q in queries)
    scope = focused_scope_markdown(directions, topic="LLM agent", selected_ids=["D1"], mode="focused")
    assert "## Focused Retrieval Scope" in scope and "D1" in scope


def test_extract_named_seeds_from_benchmark_titles() -> None:
    from research_agent_platform.review_directions import extract_named_seeds

    assert "PaperBench" in extract_named_seeds("PaperBench: Evaluating AI's Ability to Replicate AI Research")
    assert "MLE-bench" in extract_named_seeds("MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering")
    assert "FML-bench" in extract_named_seeds("FML-bench: Benchmarking Machine Learning Agents for Scientific Research")
    assert any("AI Scientist" in seed for seed in extract_named_seeds(
        "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"
    ))
    assert extract_named_seeds("Machine learning experimentation agent benchmarks") == []


def test_focused_query_plan_uses_named_seeds_not_theme_title() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    topic = "多智能体科研系统的Benchmark（评测基准）"
    directions = [
        ReviewDirection(
            id="D1",
            title="Machine learning experimentation agent benchmarks",
            representative_paper_ids=["P014", "P046"],
            candidate_paper_ids=["P014", "P046", "P047"],
        ),
        ReviewDirection(
            id="D2",
            title="Scientific paper replication and reproducibility agent benchmarks",
            representative_paper_ids=["P015"],
            candidate_paper_ids=["P015"],
        ),
        ReviewDirection(
            id="D3",
            title="End-to-end autonomous AI scientist systems",
            representative_paper_ids=["P024"],
            candidate_paper_ids=["P024"],
        ),
    ]
    papers = [
        _paper("P014", "FML-bench: Benchmarking Machine Learning Agents for Scientific Research"),
        _paper("P046", "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering"),
        _paper("P047", "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation"),
        _paper("P015", "PaperBench: Evaluating AI's Ability to Replicate AI Research"),
        _paper("P024", "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"),
    ]
    discovery = [
        "LLM research agent benchmark evaluation",
        "agentic research benchmark MLAgentBench MLE-bench PaperBench RE-Bench ScienceAgentBench",
        f"{topic} review",
        topic,
    ]
    queries = focused_query_plan(
        {"mode": "focused", "selected_ids": ["D1", "D2", "D3"], "custom_text": ""},
        directions,
        topic=topic,
        limit=10,
        papers=papers,
        discovery_queries=discovery,
    )
    blob = "\n".join(queries)
    assert queries
    assert "FML-bench" in blob
    assert "PaperBench" in blob
    assert "MLE-bench" in blob
    assert "AI Scientist" in blob
    assert all("End-to-end autonomous" not in q for q in queries)
    assert all(not q.startswith(topic) for q in queries)
    assert topic not in blob
    assert any("RE-Bench" in q or "ScienceAgentBench" in q for q in queries)
    assert "LLM research agent benchmark evaluation" not in queries
    assert all(not is_generic_search_query(q, topic=topic) for q in queries)


def test_focused_query_plan_custom_text_does_not_search_default_direction() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    directions = [
        ReviewDirection(id="D1", title="Shared memory", query_terms=["shared memory"]),
    ]
    queries = focused_query_plan(
        {"mode": "focused", "selected_ids": [], "custom_text": "IdeationBench hypothesis generation"},
        directions,
        topic="LLM agent",
        limit=10,
    )
    assert queries == ["IdeationBench hypothesis generation"]


def test_focused_query_plan_drops_topic_plus_generic_theme() -> None:
    from research_agent_platform.review_directions import ReviewDirection

    topic = "多智能体科研系统的Benchmark（评测基准）"
    queries = focused_query_plan(
        {"mode": "focused", "selected_ids": ["D3"], "custom_text": ""},
        [ReviewDirection(id="D3", title="End-to-end autonomous AI scientist systems")],
        topic=topic,
        limit=10,
    )
    assert all(topic not in q for q in queries)
    assert all("End-to-end autonomous" not in q for q in queries)


# --- Cross-round dedup -----------------------------------------------------


def test_merge_paper_records_collapses_cross_round_duplicates() -> None:
    discovery = _paper("P001", "Shared memory in agents", doi="10.1/x", queries=["Q1"])
    discovery.search_round = "discovery"
    discovery.direction_ids = ["D4"]
    focused = _paper("P055", "Shared memory in agents", doi="10.1/x", queries=["FQ1"])
    focused.search_round = "focused"
    merged = merge_paper_records([discovery, focused])
    assert len(merged) == 1
    survivor = merged[0]
    assert set(survivor.matched_queries) == {"Q1", "FQ1"}
    assert survivor.direction_ids == ["D4"]
    assert "discovery" in survivor.search_round and "focused" in survivor.search_round


# --- Screening + gate ------------------------------------------------------


def test_parse_json_block_tolerates_fences_and_prose() -> None:
    assert parse_json_block("```json\n[{\"a\": 1}]\n```") == [{"a": 1}]
    assert parse_json_block("Sure, here you go: [1, 2, 3] done") == [1, 2, 3]
    assert parse_json_block("{\"x\": true}") == {"x": True}
    assert parse_json_block("totally not json") is None


def test_screen_papers_applies_tiered_keep_drop_and_gate_counts() -> None:
    papers = [
        _paper("P001", "core paper", relevance=0.5),
        _paper("P002", "off-topic", relevance=0.2),
        _paper("P003", "background", relevance=0.4),
    ]
    response = (
        "Here is the screening:\n```json\n"
        '[{"id": "P001", "decision": "keep", "tier": "core", "relevance": 0.9, '
        '"matched_inclusion": ["scope object"], "reason": "on topic"},'
        ' {"id": "P002", "decision": "drop", "tier": "exclude", "relevance": 0.1, '
        '"violated_exclusion": ["off scope"], "reason": "off topic"}]'
        "\n```"
    )
    summary = asyncio.run(
        screen_papers(papers, topic="t", scope="s", generate_text=_gen(response), max_candidates=40)
    )
    assert summary["screened"] is True
    assert papers[0].screen_decision == "keep" and papers[0].screen_tier == "core"
    assert papers[0].matched_inclusion == ["scope object"]
    assert papers[1].screen_decision == "drop" and papers[1].screen_tier == "exclude"
    assert papers[1].violated_exclusion == ["off scope"]
    # P003 was omitted by the model -> unscreened (not counted, not a fake background keep).
    assert papers[2].screen_decision == "" and papers[2].screen_tier == "unscreened"
    # Only core/transferable keeps count toward the gate; unscreened does not.
    relevant, traceable = screening_gate_counts(papers, screened=True)
    assert relevant == 1 and traceable == 1
    assert {p.paper_id for p in kept_papers(papers)} == {"P001"}


def test_screening_gate_falls_back_to_lexical_on_total_failure() -> None:
    papers = [
        _paper("P001", "a", relevance=0.5),
        _paper("P002", "b", relevance=0.2),
        _paper("P003", "c", relevance=0.4),
    ]
    events: list[str] = []
    summary = asyncio.run(
        screen_papers(
            papers, topic="t", scope="s", generate_text=_gen("garbage"), on_event=events.append
        )
    )
    assert summary["screened"] is False
    assert summary["screen_error"]  # failure reason surfaced, not swallowed
    assert events  # degradation was reported through the callback
    assert all(paper.screen_decision == "" for paper in papers)
    relevant, traceable = screening_gate_counts(papers, screened=False)
    assert relevant == 2 and traceable == 2


def test_hard_exclude_drops_terms_before_llm() -> None:
    papers = [
        _paper("P001", "Multi-agent research benchmark", abstract="benchmark for research agents"),
        _paper("P002", "Minecraft village patrol", abstract="game agents patrol a minecraft village"),
    ]
    excluded = hard_exclude(papers, exclusion_terms=["minecraft", "patrol"])
    assert excluded == 1
    assert papers[1].screen_tier == "exclude" and papers[1].screen_decision == "drop"
    assert "minecraft" in papers[1].violated_exclusion
    assert papers[0].screen_tier == ""


def test_screen_papers_partial_batch_failure_is_not_silent() -> None:
    papers = [
        _paper("P001", "alpha good", relevance=0.6),
        _paper("P002", "beta bad", relevance=0.5),
    ]
    # Batch size 1 => two batches; only the P001 batch returns valid JSON.
    responses = {
        "P001": '[{"id": "P001", "decision": "keep", "tier": "core", "relevance": 0.9}]',
        "P002": "totally not json",
    }
    events: list[str] = []
    summary = asyncio.run(
        screen_papers(
            papers,
            topic="t",
            scope="s",
            generate_text=_gen_map(responses),
            batch_size=1,
            on_event=events.append,
        )
    )
    assert summary["screened"] is True  # at least one batch parsed
    assert summary["batches_total"] == 2 and summary["batches_failed"] == 1
    assert summary["screen_error"]
    assert papers[0].screen_decision == "keep" and papers[0].screen_tier == "core"
    # The failed batch is queued as uncertain, never silently kept as relevant.
    assert papers[1].screen_decision == "uncertain"
    relevant, _ = screening_gate_counts(papers, screened=True)
    assert relevant == 1  # only the adjudicated core paper counts


def test_screen_papers_runs_batches_concurrently_and_applies_in_order() -> None:
    # Descending relevance keeps the lexical rerank order == P001..P006, so batch N holds P00N
    # and error accounting is deterministic.
    papers = [_paper(f"P{i:03d}", f"paper {i}", relevance=1.0 - i * 0.05) for i in range(1, 7)]
    active = 0
    peak = 0

    async def gen(*, system_prompt, user_prompt, model=None, temperature=None, timeout=None, max_tokens=None):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.05)
            for paper in papers:
                if f"id: {paper.paper_id}" in user_prompt:
                    # One batch returns garbage to prove order-stable accounting under concurrency.
                    if paper.paper_id == "P004":
                        return "not json"
                    return f'[{{"id": "{paper.paper_id}", "decision": "keep", "tier": "core", "relevance": 0.9}}]'
            return ""
        finally:
            active -= 1

    summary = asyncio.run(
        screen_papers(
            papers,
            topic="t",
            scope="s",
            generate_text=gen,
            batch_size=1,
            max_candidates=40,
            max_concurrency=3,
        )
    )
    # Batches genuinely overlapped, but never exceeded the semaphore cap.
    assert 1 < peak <= 3
    assert summary["batches_total"] == 6 and summary["batches_failed"] == 1
    # Decisions land on the right papers regardless of which batch finished first.
    assert all(
        p.screen_decision == "keep" and p.screen_tier == "core"
        for p in papers
        if p.paper_id != "P004"
    )
    p4 = next(p for p in papers if p.paper_id == "P004")
    assert p4.screen_decision == "uncertain"
    # Error is attributed to the correctly-ordered batch, not whichever finished first.
    assert any("batch 4" in part for part in summary["screen_error"].split("; "))


def test_screen_papers_sequential_default_matches_parallel_decisions() -> None:
    # Default max_concurrency=1 must still produce identical decisions (no behavior drift).
    def _make():
        return [_paper(f"P{i:03d}", f"paper {i}", relevance=1.0 - i * 0.05) for i in range(1, 5)]

    def _gen_keep():
        async def gen(*, system_prompt, user_prompt, model=None, temperature=None, timeout=None, max_tokens=None):
            for i in range(1, 5):
                pid = f"P{i:03d}"
                if f"id: {pid}" in user_prompt:
                    return f'[{{"id": "{pid}", "decision": "keep", "tier": "core", "relevance": 0.9}}]'
            return ""

        return gen

    seq_papers = _make()
    par_papers = _make()
    asyncio.run(screen_papers(seq_papers, topic="t", scope="s", generate_text=_gen_keep(), batch_size=1))
    asyncio.run(
        screen_papers(par_papers, topic="t", scope="s", generate_text=_gen_keep(), batch_size=1, max_concurrency=4)
    )
    assert [(p.screen_decision, p.screen_tier) for p in seq_papers] == [
        (p.screen_decision, p.screen_tier) for p in par_papers
    ]


def test_screen_overflow_is_unscreened_and_excluded_from_gate() -> None:
    papers = [_paper(f"P{i:03d}", f"paper {i}", relevance=1.0 - i * 0.01) for i in range(1, 5)]
    response = (
        '[{"id": "P001", "decision": "keep", "tier": "core", "relevance": 0.9, "reason": "on topic"}]'
    )
    summary = asyncio.run(
        screen_papers(papers, topic="t", scope="s", generate_text=_gen(response), max_candidates=1)
    )
    assert summary["screened"] is True
    assert papers[0].screen_tier == "core"
    assert all(paper.screen_tier == "unscreened" and paper.screen_decision == "" for paper in papers[1:])
    relevant, _ = screening_gate_counts(papers, screened=True)
    assert relevant == 1
    assert summary["unscreened"] == 3


def test_semantic_rerank_uses_embedding_similarity() -> None:
    near = _paper("P001", "on scope", abstract="matches the query closely", relevance=0.1)
    far = _paper("P002", "off scope", abstract="unrelated topic", relevance=0.9)

    async def embed(texts: list[str]) -> list[list[float]]:
        # query -> [1,0]; near -> [1,0]; far -> [0,1]
        vectors = {0: [1.0, 0.0]}
        out = []
        for index, _ in enumerate(texts):
            if index == 0:
                out.append([1.0, 0.0])
            elif "on scope" in texts[index]:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    ordered, mode = asyncio.run(
        semantic_rerank([far, near], topic="q", scope="", embed=embed)
    )
    assert mode == "embedding"
    # Despite far having a higher lexical score, embedding puts the on-scope paper first.
    assert ordered[0].paper_id == "P001"


def test_semantic_rerank_degrades_to_lexical_on_embed_error() -> None:
    async def embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed down")

    events: list[str] = []
    ordered, mode = asyncio.run(
        semantic_rerank(
            [_paper("P001", "a", relevance=0.2), _paper("P002", "b", relevance=0.8)],
            topic="q",
            scope="",
            embed=embed,
            on_event=events.append,
        )
    )
    assert mode == "lexical" and events
    assert ordered[0].paper_id == "P002"  # falls back to relevance_score ordering


# --- Configurable recall prefilter -----------------------------------------


def test_recall_threshold_controls_admission(monkeypatch) -> None:
    async def openalex(_client, _query, _limit):
        return [
            _paper("S1", "Sponge city stormwater control", abstract="sponge city stormwater green infrastructure", doi="10/a"),
            _paper("S2", "Urban culture", abstract="city cultural policy heritage", doi="10/b"),
        ]

    async def empty(_client, _query, _limit):
        return []

    def build(threshold: float) -> ScholarSearchService:
        service = ScholarSearchService(
            timeout_seconds=0.1, crossref_enabled=False, dblp_enabled=False, recall_threshold=threshold
        )
        monkeypatch.setattr(service, "_search_openalex", openalex)
        monkeypatch.setattr(service, "_search_semantic_scholar", empty)
        monkeypatch.setattr(service, "_search_arxiv", empty)
        return service

    strict = asyncio.run(build(0.99).search_bundle("sponge city stormwater", per_source_limit=5, domain="cs"))
    loose = asyncio.run(build(0.0).search_bundle("sponge city stormwater", per_source_limit=5, domain="cs"))
    assert len(loose.papers) >= len(strict.papers)
    assert len(loose.papers) == 2  # the weakly-relevant paper is admitted only at a low threshold
    assert len(strict.papers) == 1
