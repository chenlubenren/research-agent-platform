from __future__ import annotations

import asyncio

import httpx

from research_agent_platform.connectors.scholar import (
    PaperRecord,
    ScholarSearchService,
    _named_artifact_terms,
    _paper_relevance,
)
from research_agent_platform.review_pipeline import clean_review_topic, parse_review_queries


def _paper(
    title: str,
    abstract: str,
    *,
    source: str,
    doi: str = "",
) -> PaperRecord:
    return PaperRecord(
        title=title,
        year=2024,
        abstract=abstract,
        authors=["Researcher"],
        url=f"https://doi.org/{doi}" if doi else "",
        venue="Water Research",
        citation_count=12,
        sources=[source],
        identifiers={"doi": doi} if doi else {},
    )


def test_review_topic_and_query_protocol_are_cleaned() -> None:
    objective = "/review 写一个当前海绵城市相关的综述"
    brief = "Q1: 海绵城市\nQ2: sponge city\nQ3: green stormwater infrastructure review"

    assert clean_review_topic(objective) == "海绵城市"
    # Chinese topics now also emit "综述"/"研究进展" discovery variants (configurable cap).
    assert parse_review_queries(objective, brief) == [
        "海绵城市",
        "sponge city",
        "green stormwater infrastructure review",
        "海绵城市 综述",
        "海绵城市 研究进展",
    ]


def test_parse_review_queries_keeps_internal_phrase_quotes_and_prefers_qn(monkeypatch) -> None:
    from research_agent_platform.config import config

    monkeypatch.setattr(config, "review_query_limit", 3)
    objective = "/review 多智能体科研系统的Benchmark（评测基准）"
    brief = (
        'Q1: LLM-driven agentic scientific research system benchmark\n'
        'Q2: "multi-agent scientific research systems" benchmark evaluation framework\n'
        "Q3: unused leftover query that should be truncated\n"
    )
    queries = parse_review_queries(objective, brief)
    assert queries[0] == "LLM-driven agentic scientific research system benchmark"
    assert queries[1].startswith('"multi-agent scientific research systems"')
    assert queries[1].count('"') == 2
    assert "评测基准" not in queries[0]


def test_multi_query_search_filters_noise_merges_doi_and_survives_provider_failure(monkeypatch) -> None:
    # Disable the newly added backbone/domain sources so this test keeps exercising
    # exactly openalex + semantic_scholar + arxiv; force the cs domain to activate arxiv.
    service = ScholarSearchService(timeout_seconds=0.1, crossref_enabled=False, dblp_enabled=False)
    arxiv_calls = 0

    async def openalex(_client, query, _limit):
        if query == "sponge city":
            return [
                _paper(
                    "Sponge city planning for urban stormwater",
                    "Green infrastructure and runoff management.",
                    source="OpenAlex",
                    doi="10.1000/sponge",
                ),
                _paper(
                    "Contemporary Chinese film and ritual aesthetics",
                    "A study of literature and cinema.",
                    source="OpenAlex",
                ),
                _paper(
                    "Urban cultural development in contemporary cities",
                    "A city policy study without stormwater content.",
                    source="OpenAlex",
                ),
            ]
        return [
            _paper(
                "Urban runoff management through sponge-city systems",
                "Sponge city green infrastructure for stormwater control.",
                source="OpenAlex",
                doi="https://doi.org/10.1000/SPONGE",
            )
        ]

    async def semantic_scholar(_client, query, _limit):
        return [
            _paper(
                f"{query.title()} evidence synthesis",
                f"Review of {query} methods, data, and limitations.",
                source="Semantic Scholar",
                doi=f"10.1000/{query.replace(' ', '-')}",
            )
        ]

    async def arxiv(_client, _query, _limit):
        nonlocal arxiv_calls
        arxiv_calls += 1
        request = httpx.Request("GET", "https://export.arxiv.org/api/query")
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    monkeypatch.setattr(service, "_search_openalex", openalex)
    monkeypatch.setattr(service, "_search_semantic_scholar", semantic_scholar)
    monkeypatch.setattr(service, "_search_arxiv", arxiv)

    bundle = asyncio.run(
        service.search_bundle(
            "sponge city",
            queries=["sponge city", "green stormwater infrastructure"],
            per_source_limit=8,
            domain="cs",
        )
    )

    # 2 queries x 3 attempts (retry backoff now allows up to 3 tries per query).
    assert arxiv_calls == 6
    # The failure reason is carried through: a count alone cannot tell a rate limit from
    # a timeout, which is what made a fully dead provider undiagnosable in the report.
    assert bundle.provider_status["arxiv"] == "error (2/2 queries failed: HTTP 503)"
    assert bundle.excluded_count == 2
    assert all("film" not in paper.title.casefold() for paper in bundle.papers)
    assert all("cultural development" not in paper.title.casefold() for paper in bundle.papers)
    assert len([paper for paper in bundle.papers if paper.identifiers.get("doi", "").casefold().endswith("sponge")]) == 1
    assert [paper.paper_id for paper in bundle.papers] == [
        f"P{index:03d}" for index in range(1, len(bundle.papers) + 1)
    ]
    assert all(paper.verification_status == "traceable_identifier" for paper in bundle.papers)


def test_identifier_and_title_bridge_collapses_existing_duplicate_groups(monkeypatch) -> None:
    service = ScholarSearchService(timeout_seconds=0.1, crossref_enabled=False, dblp_enabled=False)

    async def openalex(_client, _query, _limit):
        return [
            _paper("Sponge city systems", "Sponge city stormwater evidence.", source="OpenAlex", doi="10.1/a"),
            _paper("Urban stormwater systems", "Sponge city stormwater evidence.", source="OpenAlex", doi="10.1/b"),
        ]

    async def semantic_scholar(_client, _query, _limit):
        return [
            _paper("Urban stormwater systems", "Sponge city stormwater evidence.", source="Semantic Scholar", doi="10.1/a"),
        ]

    async def arxiv(_client, _query, _limit):
        return []

    monkeypatch.setattr(service, "_search_openalex", openalex)
    monkeypatch.setattr(service, "_search_semantic_scholar", semantic_scholar)
    monkeypatch.setattr(service, "_search_arxiv", arxiv)

    bundle = asyncio.run(service.search_bundle("sponge city", per_source_limit=8))

    assert len(bundle.papers) == 1
    assert set(bundle.papers[0].sources) == {"OpenAlex", "Semantic Scholar"}


def _named(title: str, *, venue: str = "arXiv") -> PaperRecord:
    return PaperRecord(
        title=title,
        year=2024,
        abstract="",
        authors=["Researcher"],
        url="https://arxiv.org/abs/2410.00001",
        venue=venue,
        citation_count=3,
        sources=["arXiv"],
        identifiers={"arxiv": "2410.00001"},
    )


ENUMERATION_QUERY = "MLAgentBench DiscoveryBench ResearchBench CORE-Bench ScienceAgentBench"


def test_named_artifact_terms_detected_from_original_casing() -> None:
    assert _named_artifact_terms(ENUMERATION_QUERY) == {
        "mlagentbench",
        "discoverybench",
        "researchbench",
        "core-bench",
        "scienceagentbench",
    }
    # Generic phrasing carries no artifact name, so nothing gets the single-hit exemption.
    assert _named_artifact_terms("multi-agent scientific discovery benchmark") == set()
    assert _named_artifact_terms("survey review LLM agent benchmark 2024 2025") == set()


def test_enumeration_query_admits_each_named_benchmark() -> None:
    # A benchmark's own paper mentions only its own name. Requiring two term hits scored
    # every one of them 0.04, below the recall threshold, so the seeds never entered the pool.
    for title in (
        "DiscoveryBench: Towards Data-Driven Discovery with Large Language Models",
        "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation",
        "ScienceAgentBench: Toward Rigorous Assessment of Language Agents",
        "CORE-Bench: Fostering the Credibility of Published Research",
    ):
        score = _paper_relevance(_named(title), [ENUMERATION_QUERY])
        assert score >= 0.5, (title, score)


def test_enumeration_query_still_rejects_unrelated_bench_noise() -> None:
    for title in (
        "Bench Press",
        "Development of a bench test for Type X core gypsum board",
        "MT-Video-Bench: A Holistic Video Understanding Benchmark",
    ):
        score = _paper_relevance(_named(title), [ENUMERATION_QUERY])
        assert score < 0.15, (title, score)


def test_generic_multi_term_query_scoring_is_unchanged() -> None:
    query = "multi-agent scientific discovery benchmark ML research agent"
    # Two generic term hits still pass; a single generic hit still scores zero.
    assert _paper_relevance(_named("Fixed-Time Cooperative Control for Multi-Agent Systems"), [query]) == 0.49
    assert _paper_relevance(_named("Machine Learning for Fluid Mechanics"), [query]) == 0.04
