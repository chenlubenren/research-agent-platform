from __future__ import annotations

from research_agent_platform.review_pipeline import (
    inherited_domain_arg,
    parse_review_domains,
    resolve_key_source_domain,
    source_health,
)


def test_parse_review_domains_reads_and_normalizes_the_brief_line() -> None:
    brief = "Q1: pain deep learning\nDomains: cs, biomed\nSource Coverage: ..."
    assert parse_review_domains(brief) == ["cs", "biomed"]
    # Synonyms normalize to connector keys; order preserved, duplicates collapsed.
    assert parse_review_domains("**Domains:** Medical / ML / clinical") == ["biomed", "cs"]
    # Unmapped or missing lines yield [] so the caller falls back to the keyword classifier.
    assert parse_review_domains("Domains: history, art") == []
    assert parse_review_domains("no domains line here") == []
    assert parse_review_domains("") == []


def test_inherited_domain_arg_joins_every_detected_domain() -> None:
    # Unlike resolve_key_source_domain (single coverage profile), this must keep the whole set so
    # a follow-up round activates both families.
    assert inherited_domain_arg(["cs", "biomed"]) == "cs biomed"
    assert inherited_domain_arg(["general"]) == ""
    assert inherited_domain_arg("cs") == "cs"
    assert inherited_domain_arg(None) == ""


def test_source_health_ok_when_key_index_healthy() -> None:
    health = source_health({"openalex": "ok (10 records across 5/5 queries)"}, domain="cs")
    assert health["coverage_state"] == "ok"
    assert "openalex" in health["healthy_sources"]


def test_source_health_degraded_when_key_index_fails() -> None:
    health = source_health(
        {
            "openalex": "ok (12 records)",
            "dblp": "error (5/5 queries failed)",
            "semantic_scholar": "ok (0 records)",
        },
        domain="cs",
    )
    assert health["coverage_state"] == "degraded"
    assert "dblp" in health["degraded_sources"]
    assert "semantic_scholar" in health["degraded_sources"]


def test_source_health_missing_alone_is_not_degraded() -> None:
    # Only openalex reported (others simply never ran) => still healthy, missing recorded.
    health = source_health({"openalex": "ok (8 records)"}, domain="cs")
    assert health["coverage_state"] == "ok"
    assert "dblp" in health["missing_sources"]


def test_source_health_degraded_when_zero_records_only() -> None:
    health = source_health({"openalex": "ok (0 records)"}, domain="")
    assert health["coverage_state"] == "degraded"
    assert "openalex" in health["degraded_sources"]


def test_source_health_degraded_when_domain_key_index_gated_off() -> None:
    # Proper-noun-only focused queries classify as 'general', which switches arXiv off.
    # The backbone staying healthy must not report the cs round as fully covered.
    health = source_health(
        {
            "arxiv": "inactive: 领域未命中 (domains=general)",
            "openalex": "ok (48 records across 6/9 queries)",
            "crossref": "ok (55 records across 9/9 queries)",
            "semantic_scholar": "ok (8 records across 1/9 queries)",
        },
        domain="cs",
    )
    assert health["coverage_state"] == "degraded"
    assert health["domain_gated_sources"] == ["arxiv"]
    assert health["degraded_sources"] == []


def test_source_health_treats_partial_provider_as_degraded() -> None:
    health = source_health(
        {"openalex": "partial (40 records across 5/8 queries, 3 failed: AttributeError)"},
        domain="",
    )
    assert health["coverage_state"] == "degraded"
    assert "openalex" in health["degraded_sources"]


def test_resolve_key_source_domain_maps_detected_domains() -> None:
    assert resolve_key_source_domain(["cs"]) == "cs"
    assert resolve_key_source_domain(["biomed"]) == "biomed"
    # 'general' is the connector's placeholder for "no domain matched".
    assert resolve_key_source_domain(["general"]) == ""
    assert resolve_key_source_domain(None) == ""
