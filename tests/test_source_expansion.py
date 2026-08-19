from __future__ import annotations

import asyncio

import httpx

from research_agent_platform.connectors.scholar import (
    ScholarSearchService,
    _classify_domain,
)
from research_agent_platform.literature_downloads import _match_source, _resolve_unpaywall_pdf
from research_agent_platform.literature_sources import DownloadSource
from research_agent_platform.connectors.scholar import PaperRecord


def _run_search(method_name: str, handler, *, service: ScholarSearchService | None = None, query: str = "q", limit: int = 5):
    service = service or ScholarSearchService(timeout_seconds=1.0)

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await getattr(service, method_name)(client, query, limit)

    return asyncio.run(go())


def test_classify_domain_matches_english_and_chinese_keywords() -> None:
    assert _classify_domain(["deep learning transformer"]) == {"cs"}
    assert _classify_domain(["肿瘤 clinical trial 基因"]) == {"biomed"}
    assert _classify_domain(["quantum topology 定理"]) == {"physics_math"}
    assert _classify_domain(["a history of renaissance painting"]) == set()
    # multi-domain queries can activate more than one domain
    assert _classify_domain(["deep learning for cancer genomics"]) == {"cs", "biomed"}
    # Word-boundary English matching: "gene" must not fire on "generation".
    assert "biomed" not in _classify_domain(
        ["hypothesis generation experiment design literature review paper writing"]
    )
    assert "biomed" not in _classify_domain(["agentic science benchmark"])


def test_select_providers_routes_backbone_domain_and_disabled_reasons() -> None:
    service = ScholarSearchService(timeout_seconds=1.0)  # repositories default off
    providers, status = service._select_providers({"biomed"})
    names = [name for name, _ in providers]

    # Backbone always on.
    assert {"openalex", "crossref", "semantic_scholar"}.issubset(set(names))
    # Biomed domain sources activate.
    assert {"europepmc", "pmc"}.issubset(set(names))
    # cs/physics domain sources stay inactive for a biomed query.
    assert "arxiv" not in names
    assert "dblp" not in names
    assert status["arxiv"].startswith("inactive")
    assert status["dblp"].startswith("inactive")
    # Optional OA repositories are disabled by default.
    assert status["core"].startswith("disabled")
    assert status["zenodo"].startswith("disabled")
    # WoS/CNKI disabled without credentials.
    assert status["wos"].startswith("disabled")


def test_classify_domain_fires_biomed_on_clinical_and_pain_vocabulary() -> None:
    # AI-for-medicine topics used to fire only cs on their method words; the clinical/biosignal
    # additions must now also fire biomed so Europe PMC/PMC stay eligible.
    assert "biomed" in _classify_domain(["objective pain assessment physiological signals"])
    assert "biomed" in _classify_domain(["deep learning for ICU sedation monitoring"])
    assert _classify_domain(["multimodal deep learning neonatal pain intensity"]) == {"cs", "biomed"}
    assert "biomed" in _classify_domain(["新生儿 疼痛 生理信号 监护"])
    # Word-boundary matching keeps the new stems from over-firing.
    assert "biomed" not in _classify_domain(["explainable planning agents"])


def test_explicit_domain_override_activates_sources() -> None:
    service = ScholarSearchService(timeout_seconds=1.0)
    domains = service._resolve_domains(["a history of painting"], "cs")
    assert domains == {"cs"}


def test_domain_override_unions_with_keyword_scan() -> None:
    # A hint only adds sources: an explicit "cs" hint keeps arXiv/DBLP, while the keyword scan
    # still contributes biomed from the clinical query so Europe PMC/PMC stay active too.
    service = ScholarSearchService(timeout_seconds=1.0)
    domains = service._resolve_domains(["deep learning ICU pain assessment"], "cs")
    assert domains == {"cs", "biomed"}
    # An unmapped hint is ignored but the keyword scan still stands.
    assert service._resolve_domains(["quantum topology theorem"], "not-a-domain") == {"physics_math"}


def test_crossref_parsing() -> None:
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1/x",
                    "title": ["A Great Paper"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "abstract": "<jats:p>Hello world</jats:p>",
                    "issued": {"date-parts": [[2021, 5]]},
                    "container-title": ["Journal of Things"],
                    "is-referenced-by-count": 7,
                    "URL": "https://doi.org/10.1/x",
                    "link": [{"content-type": "application/pdf", "URL": "https://x/pdf"}],
                }
            ]
        }
    }
    records = _run_search("_search_crossref", lambda request: httpx.Response(200, json=payload))
    assert len(records) == 1
    record = records[0]
    assert record.title == "A Great Paper"
    assert record.year == 2021
    assert record.authors == ["Ada Lovelace"]
    assert record.abstract == "Hello world"
    assert record.venue == "Journal of Things"
    assert record.citation_count == 7
    assert record.pdf_url == "https://x/pdf"
    assert record.identifiers["doi"] == "10.1/x"
    assert record.sources == ["Crossref"]


def test_dblp_parsing() -> None:
    payload = {
        "result": {
            "hits": {
                "hit": [
                    {
                        "info": {
                            "title": "Deep Nets.",
                            "authors": {"author": [{"@pid": "1", "text": "Jane Doe"}, {"text": "John Roe"}]},
                            "venue": "NeurIPS",
                            "year": "2020",
                            "doi": "10.2/y",
                            "ee": "https://ee/y",
                            "key": "conf/x",
                        }
                    }
                ]
            }
        }
    }
    records = _run_search("_search_dblp", lambda request: httpx.Response(200, json=payload))
    assert len(records) == 1
    record = records[0]
    assert record.title == "Deep Nets"
    assert record.authors == ["Jane Doe", "John Roe"]
    assert record.venue == "NeurIPS"
    assert record.year == 2020
    assert record.identifiers["doi"] == "10.2/y"
    assert record.identifiers["dblp"] == "conf/x"
    assert record.sources == ["DBLP"]


def test_europepmc_parsing() -> None:
    payload = {
        "resultList": {
            "result": [
                {
                    "id": "123",
                    "source": "MED",
                    "title": "Cancer study",
                    "authorList": {"author": [{"fullName": "A B"}]},
                    "abstractText": "abstract",
                    "doi": "10.3/z",
                    "pmid": "123",
                    "pubYear": "2019",
                    "journalTitle": "Cell",
                    "citedByCount": 3,
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {"documentStyle": "pdf", "url": "https://pmc/pdf"},
                            {"documentStyle": "html", "url": "https://pmc/html"},
                        ]
                    },
                }
            ]
        }
    }
    records = _run_search("_search_europepmc", lambda request: httpx.Response(200, json=payload))
    assert len(records) == 1
    record = records[0]
    assert record.title == "Cancer study"
    assert record.pdf_url == "https://pmc/pdf"
    assert record.url == "https://pmc/html"
    assert record.year == 2019
    assert record.venue == "Cell"
    assert record.identifiers == {"doi": "10.3/z", "pmid": "123"}
    assert record.sources == ["Europe PMC"]


def test_zenodo_parsing() -> None:
    payload = {
        "hits": {
            "hits": [
                {
                    "id": 42,
                    "doi": "10.5281/zenodo.42",
                    "metadata": {
                        "title": "Open Dataset Paper",
                        "creators": [{"name": "Grace Hopper"}],
                        "description": "<p>Some <b>HTML</b> abstract</p>",
                        "publication_date": "2022-03-01",
                    },
                    "links": {"html": "https://zenodo.org/record/42"},
                    "files": [
                        {"key": "paper.pdf", "links": {"self": "https://zenodo.org/api/files/paper.pdf"}}
                    ],
                }
            ]
        }
    }
    service = ScholarSearchService(timeout_seconds=1.0, zenodo_enabled=True)
    records = _run_search("_search_zenodo", lambda request: httpx.Response(200, json=payload), service=service)
    assert len(records) == 1
    record = records[0]
    assert record.title == "Open Dataset Paper"
    assert record.abstract == "Some HTML abstract"
    assert record.pdf_url == "https://zenodo.org/api/files/paper.pdf"
    assert record.year == 2022
    assert record.identifiers["doi"] == "10.5281/zenodo.42"
    assert record.sources == ["Zenodo"]


def test_unpaywall_resolver_returns_best_oa_pdf() -> None:
    payload = {
        "best_oa_location": {"url_for_pdf": "https://oa/best.pdf", "url": "https://oa/landing"},
        "oa_locations": [{"url": "https://oa/other"}],
    }

    async def go(email: str) -> str:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await _resolve_unpaywall_pdf(client, "10.9/oa", email)

    assert asyncio.run(go("me@example.com")) == "https://oa/best.pdf"
    # No email configured -> resolver is skipped without network access.
    assert asyncio.run(go("")) == ""


def test_unpaywall_resolver_handles_missing_record() -> None:
    async def go() -> str:
        transport = httpx.MockTransport(lambda request: httpx.Response(404, json={}))
        async with httpx.AsyncClient(transport=transport) as client:
            return await _resolve_unpaywall_pdf(client, "10.9/missing", "me@example.com")

    assert asyncio.run(go()) == ""


def test_match_source_returns_none_when_no_match() -> None:
    paper = PaperRecord(
        title="Totally unrelated title",
        year=2024,
        abstract="",
        authors=[],
        url="https://example.org/paper",
        venue="",
        citation_count=None,
        identifiers={},
    )
    sources = [DownloadSource(source_path="refs.bib", source_type="bib", title="A different paper", doi="10.1/other")]
    assert _match_source(paper, sources, []) is None
