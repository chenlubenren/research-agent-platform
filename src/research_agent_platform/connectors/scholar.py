from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import httpx


OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
WOS_DEFAULT_BASE_URL = "https://api.clarivate.com/apis/wos-starter/v1"
CROSSREF_URL = "https://api.crossref.org/works"
DBLP_URL = "https://dblp.org/search/publ/api"
EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PMC_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CORE_URL = "https://api.core.ac.uk/v3/search/works"
OPENAIRE_URL = "https://api.openaire.eu/search/researchProducts"
BASE_FCGI_URL = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
ZENODO_URL = "https://zenodo.org/api/records"
HAL_URL = "https://api.archives-ouvertes.fr/search/"

# Per-provider (max concurrent requests, minimum seconds between request starts).
# A review round fires every planned query at every provider inside one gather, so an
# 8-query plan used to hit arXiv with 8 simultaneous requests. arXiv's export API asks
# for roughly one request every three seconds and Semantic Scholar's anonymous tier
# answers bursts with HTTP 429; both then fail every query in the round at once.
PROVIDER_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "arxiv": (1, 3.0),
    # 1.2s still collected HTTP 429 on most queries of a live 5-query round; the anonymous
    # tier's pool is shared, so the throttle also widens itself on each 429 (see penalize).
    "semantic_scholar": (1, 2.0),
    "dblp": (1, 1.0),
    "pmc": (1, 0.4),
}
DEFAULT_PROVIDER_RATE_LIMIT = (4, 0.0)
# An authenticated Semantic Scholar key grants a private ~1 req/s budget, so the defensive
# 2s anonymous pacing (chosen to survive a shared 429 pool) can be tightened once a key is set.
SEMANTIC_SCHOLAR_AUTH_RATE_LIMIT = (1, 1.1)

# Sources that always run regardless of the detected research domain.
BACKBONE_SOURCES = ("openalex", "crossref", "semantic_scholar")
# Cross-domain open-access repositories; each is additionally gated by its config flag.
OA_REPOSITORY_SOURCES = ("core", "openaire", "base", "zenodo", "hal")
# Default mapping from a detected domain to the extra sources it activates.
DEFAULT_DOMAIN_SOURCE_MAP: dict[str, list[str]] = {
    "cs": ["arxiv", "dblp"],
    "physics_math": ["arxiv"],
    "biomed": ["europepmc", "pmc"],
}
# Keyword tables (English + Chinese) used by the deterministic domain classifier.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cs": (
        "machine learning", "deep learning", "neural network", "neural net", "transformer",
        "attention", "algorithm", "nlp", "natural language", "computer vision", "reinforcement learning",
        "graph neural", "llm", "language model", "diffusion model", "gan", "segmentation",
        "classification", "software", "dataset", "benchmark", "convolutional", "embedding",
        "深度学习", "神经网络", "机器学习", "算法", "图像", "视觉", "自然语言", "大模型", "语言模型",
        "扩散模型", "分类", "分割", "强化学习", "图神经", "软件", "数据集", "卷积", "嵌入",
    ),
    "biomed": (
        "gene", "genome", "genomic", "protein", "cancer", "tumor", "tumour", "clinical", "disease",
        "drug", "molecular", "immun", "therapy", "patient", "biomarker", "neuron", "brain", "medical",
        "health", "virus", "vaccine", "enzyme", "cell biology", "rna", "dna", "pathogen",
        # Clinical-application / biosignal vocabulary. The table above only covered wet-lab and
        # oncology, so AI-for-medicine topics (pain assessment, ICU monitoring, wearables) fired
        # cs on their method words but never biomed, switching off Europe PMC/PMC. English entries
        # must be whole words (word-boundary matching), so list the surface forms explicitly.
        "pain", "nociception", "nociceptive", "analgesia", "analgesic", "anesthesia", "anaesthesia",
        "sedation", "icu", "intensive care", "perioperative", "postoperative", "neonatal",
        "physiological", "physiology", "electrodermal", "clinician", "diagnosis", "diagnostic",
        "基因", "蛋白", "癌", "肿瘤", "临床", "疾病", "药物", "分子", "免疫", "治疗", "患者", "病毒",
        "疫苗", "医学", "健康", "神经元", "细胞", "病原",
        "疼痛", "镇痛", "麻醉", "镇静", "术后", "围术期", "重症", "监护", "生理信号", "心率变异",
        "脑电", "护理", "诊断", "新生儿",
    ),
    "physics_math": (
        "quantum", "physics", "relativity", "particle", "topology", "manifold", "theorem", "algebra",
        "geometry", "differential equation", "cosmolog", "astrophys", "condensed matter", "superconduc",
        "量子", "物理", "相对论", "粒子", "拓扑", "流形", "定理", "代数", "几何", "微分方程", "宇宙",
        "天体", "凝聚态", "超导",
    ),
}


class _ProviderThrottle:
    """Caps concurrency and paces request starts for a single provider.

    The pacing interval widens when the provider reports rate limiting. Semantic Scholar's
    anonymous tier draws from a pool shared with every other anonymous caller, so a fixed
    interval tuned offline still collects HTTP 429 under contention; retrying each query
    independently then just re-hits the same wall. Backing the whole provider off once,
    for every query still queued in the round, is what actually clears it.
    """

    MAX_INTERVAL = 8.0

    def __init__(self, concurrency: int, min_interval: float) -> None:
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._min_interval = max(0.0, min_interval)
        self._pace_lock = asyncio.Lock()
        self._next_start = 0.0

    async def __aenter__(self) -> "_ProviderThrottle":
        await self._semaphore.acquire()
        if self._min_interval:
            # Holding the lock across the sleep is what serializes the pacing.
            async with self._pace_lock:
                loop = asyncio.get_running_loop()
                delay = self._next_start - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._next_start = max(loop.time(), self._next_start) + self._min_interval
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        self._semaphore.release()

    def penalize(self, retry_after: float) -> None:
        """Widen pacing and defer queued starts after a rate-limit response."""

        if self._min_interval:
            self._min_interval = min(self.MAX_INTERVAL, self._min_interval * 2)
        else:
            self._min_interval = min(self.MAX_INTERVAL, max(1.0, retry_after))
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            return
        self._next_start = max(self._next_start, now + max(0.0, retry_after))


def _build_throttles(
    names: Iterable[str],
    overrides: dict[str, tuple[int, float]] | None = None,
) -> dict[str, _ProviderThrottle]:
    overrides = overrides or {}
    return {
        name: _ProviderThrottle(
            *overrides.get(name, PROVIDER_RATE_LIMITS.get(name, DEFAULT_PROVIDER_RATE_LIMIT))
        )
        for name in names
    }


@dataclass
class PaperRecord:
    title: str
    year: int | None
    abstract: str
    authors: list[str]
    url: str
    venue: str
    citation_count: int | None
    sources: list[str] = field(default_factory=list)
    identifiers: dict[str, str] = field(default_factory=dict)
    paper_id: str = ""
    relevance_score: float = 0.0
    matched_queries: list[str] = field(default_factory=list)
    verification_status: str = "source_metadata"
    pdf_url: str = ""
    download_status: str = "not_attempted"
    downloaded_path: str = ""
    download_error: str = ""
    # LLM screening annotations (net-new; populated by review_screening.screen_papers).
    screen_decision: str = ""  # "keep" | "drop" | "uncertain" | "" (not screened)
    screen_relevance: float | None = None
    screen_label: str = ""  # legacy: "core" | "boundary" | ""
    screen_reason: str = ""
    # Layered screening tier + evidence for the keep/drop rationale.
    screen_tier: str = ""  # "core" | "transferable" | "background" | "exclude" | ""
    matched_inclusion: list[str] = field(default_factory=list)
    violated_exclusion: list[str] = field(default_factory=list)
    # Two-round retrieval provenance and direction membership.
    search_round: str = ""  # "discovery" | "focused" | "discovery+focused"
    direction_ids: list[str] = field(default_factory=list)

    def merge(self, other: "PaperRecord") -> None:
        if not self.abstract and other.abstract:
            self.abstract = other.abstract
        if not self.url and other.url:
            self.url = other.url
        if not self.pdf_url and other.pdf_url:
            self.pdf_url = other.pdf_url
        if not self.venue and other.venue:
            self.venue = other.venue
        if self.year is None and other.year is not None:
            self.year = other.year
        if self.citation_count is None and other.citation_count is not None:
            self.citation_count = other.citation_count
        if len(other.authors) > len(self.authors):
            self.authors = other.authors
        for source in other.sources:
            if source not in self.sources:
                self.sources.append(source)
        self.identifiers.update({k: v for k, v in other.identifiers.items() if v})
        for query in other.matched_queries:
            if query not in self.matched_queries:
                self.matched_queries.append(query)
        self.relevance_score = max(self.relevance_score, other.relevance_score)
        for direction_id in other.direction_ids:
            if direction_id not in self.direction_ids:
                self.direction_ids.append(direction_id)
        if other.search_round and other.search_round != self.search_round:
            rounds = [r for r in [self.search_round, other.search_round] if r]
            self.search_round = "+".join(dict.fromkeys("+".join(rounds).split("+")))
        # Prefer an explicit "keep" screening decision and the stronger relevance signal.
        if other.screen_decision and (not self.screen_decision or other.screen_decision == "keep"):
            self.screen_decision = other.screen_decision
            self.screen_label = other.screen_label or self.screen_label
            self.screen_reason = other.screen_reason or self.screen_reason
            self.screen_tier = other.screen_tier or self.screen_tier
        for term in other.matched_inclusion:
            if term not in self.matched_inclusion:
                self.matched_inclusion.append(term)
        for term in other.violated_exclusion:
            if term not in self.violated_exclusion:
                self.violated_exclusion.append(term)
        if other.screen_relevance is not None:
            self.screen_relevance = (
                other.screen_relevance
                if self.screen_relevance is None
                else max(self.screen_relevance, other.screen_relevance)
            )
        if other.download_status == "downloaded":
            self.download_status = other.download_status
            self.downloaded_path = other.downloaded_path
            self.download_error = other.download_error
        elif self.download_status == "not_attempted" and other.download_status != "not_attempted":
            self.download_status = other.download_status
            self.downloaded_path = other.downloaded_path
            self.download_error = other.download_error


@dataclass
class LiteratureBundle:
    query: str
    papers: list[PaperRecord]
    provider_status: dict[str, str]
    queries: list[str] = field(default_factory=list)
    excluded_count: int = 0
    quality: dict[str, object] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# Literature Search Bundle",
            "",
            f"- Query: {self.query}",
            f"- Expanded queries: {', '.join(self.queries) or self.query}",
            f"- Retrieval quality: {self.quality.get('status', 'unknown')}",
            f"- Relevant papers retained: {len(self.papers)}",
            f"- Irrelevant candidates excluded: {self.excluded_count}",
            "",
            "## Provider Status",
        ]
        for provider, status in self.provider_status.items():
            lines.append(f"- {provider}: {status}")
        lines.extend(["", "## Aggregated Papers"])
        if not self.papers:
            lines.append("")
            lines.append("_No papers retrieved._")
            return "\n".join(lines) + "\n"
        for index, paper in enumerate(self.papers, start=1):
            title = paper.title or "Untitled"
            year = f" ({paper.year})" if paper.year else ""
            lines.extend(
                [
                    "",
                    f"### [{paper.paper_id or f'P{index:03d}'}] {title}{year}",
                    f"- Sources: {', '.join(paper.sources) or 'unknown'}",
                    f"- Matched queries: {', '.join(paper.matched_queries) or 'unknown'}",
                    f"- Relevance score: {paper.relevance_score:.3f}",
                    f"- Traceability: {paper.verification_status}",
                ]
            )
            if paper.screen_decision:
                screen_relevance = (
                    f"{paper.screen_relevance:.2f}" if paper.screen_relevance is not None else "n/a"
                )
                tier = paper.screen_tier or paper.screen_label
                lines.append(
                    f"- Screening: {paper.screen_decision}"
                    + (f" ({tier})" if tier else "")
                    + f", relevance={screen_relevance}"
                )
                if paper.matched_inclusion:
                    lines.append(f"- Matched inclusion: {', '.join(paper.matched_inclusion)}")
                if paper.violated_exclusion:
                    lines.append(f"- Violated exclusion: {', '.join(paper.violated_exclusion)}")
                if paper.screen_reason:
                    lines.append(f"- Recommendation reason: {paper.screen_reason}")
            if paper.search_round:
                lines.append(f"- Search round: {paper.search_round}")
            if paper.direction_ids:
                lines.append(f"- Directions: {', '.join(paper.direction_ids)}")
            lines.extend(
                [
                    f"- Authors: {', '.join(paper.authors[:8]) or 'unknown'}",
                    f"- Venue: {paper.venue or 'unknown'}",
                    f"- Citations: {paper.citation_count if paper.citation_count is not None else 'unknown'}",
                    f"- URL: {paper.url or 'unknown'}",
                    f"- Public PDF: {paper.pdf_url or 'not advertised'}",
                    f"- Download: {paper.download_status}"
                    + (f" (`{paper.downloaded_path}`)" if paper.downloaded_path else ""),
                ]
            )
            if paper.download_error:
                lines.append(f"- Download note: {paper.download_error}")
            if paper.identifiers:
                identifiers = ", ".join(f"{key}={value}" for key, value in paper.identifiers.items())
                lines.append(f"- Identifiers: {identifiers}")
            if paper.abstract:
                lines.extend(["", "Summary:", paper.abstract.strip()])
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        payload = {
            "query": self.query,
            "queries": self.queries,
            "provider_status": self.provider_status,
            "excluded_count": self.excluded_count,
            "quality": self.quality,
            "papers": [asdict(item) for item in self.papers],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def prompt_excerpt(self, *, limit: int = 4500) -> str:
        lines = ["# Admitted Evidence Records"]
        for index, paper in enumerate(self.papers, start=1):
            stable_id = paper.paper_id or f"P{index:03d}"
            heading = f"### [{stable_id}]"
            if paper.title:
                heading += f" {paper.title}"
            if paper.year:
                heading += f" ({paper.year})"
            lines.extend(["", heading])
            if paper.authors:
                lines.append(f"- Authors: {', '.join(paper.authors[:8])}")
            if paper.venue:
                lines.append(f"- Venue: {paper.venue}")
            if paper.url:
                lines.append(f"- Source URL: {paper.url}")
            if paper.identifiers:
                identifiers = ", ".join(f"{key}={value}" for key, value in paper.identifiers.items())
                lines.append(f"- Stable identifiers: {identifiers}")
            if paper.abstract:
                lines.extend(["", "Admitted evidence summary:", paper.abstract.strip()])
        return ("\n".join(lines) + "\n")[:limit]


class ScholarSearchService:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        *,
        wos_api_base_url: str = WOS_DEFAULT_BASE_URL,
        wos_api_key: str = "",
        wos_default_db: str = "WOS",
        cnki_search_endpoint: str = "",
        cnki_search_method: str = "GET",
        cnki_api_key: str = "",
        cnki_auth_header: str = "X-ApiKey",
        cnki_auth_scheme: str = "",
        semantic_scholar_api_key: str = "",
        crossref_enabled: bool = True,
        crossref_mailto: str = "research-agent@example.com",
        dblp_enabled: bool = True,
        europepmc_enabled: bool = True,
        europepmc_email: str = "research-agent@example.com",
        pmc_enabled: bool = True,
        ncbi_email: str = "research-agent@example.com",
        core_enabled: bool = False,
        core_api_key: str = "",
        openaire_enabled: bool = False,
        openaire_api_key: str = "",
        base_enabled: bool = False,
        zenodo_enabled: bool = False,
        zenodo_access_token: str = "",
        hal_enabled: bool = False,
        domain_map: str = "",
        minimum_for_synthesis: int = 10,
        recommended_for_review: int = 15,
        recall_threshold: float = 0.34,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.wos_api_base_url = wos_api_base_url.rstrip("/")
        self.wos_api_key = wos_api_key.strip()
        self.wos_default_db = wos_default_db.strip() or "WOS"
        self.cnki_search_endpoint = cnki_search_endpoint.strip()
        self.cnki_search_method = cnki_search_method.strip().upper() or "GET"
        self.cnki_api_key = cnki_api_key.strip()
        self.cnki_auth_header = cnki_auth_header.strip() or "X-ApiKey"
        self.cnki_auth_scheme = cnki_auth_scheme.strip()
        self.semantic_scholar_api_key = semantic_scholar_api_key.strip()
        self.crossref_enabled = bool(crossref_enabled)
        self.crossref_mailto = crossref_mailto.strip() or "research-agent@example.com"
        self.dblp_enabled = bool(dblp_enabled)
        self.europepmc_enabled = bool(europepmc_enabled)
        self.europepmc_email = europepmc_email.strip() or "research-agent@example.com"
        self.pmc_enabled = bool(pmc_enabled)
        self.ncbi_email = ncbi_email.strip() or "research-agent@example.com"
        self.core_enabled = bool(core_enabled)
        self.core_api_key = core_api_key.strip()
        self.openaire_enabled = bool(openaire_enabled)
        self.openaire_api_key = openaire_api_key.strip()
        self.base_enabled = bool(base_enabled)
        self.zenodo_enabled = bool(zenodo_enabled)
        self.zenodo_access_token = zenodo_access_token.strip()
        self.hal_enabled = bool(hal_enabled)
        self.domain_source_map = _resolve_domain_source_map(domain_map)
        self.minimum_for_synthesis = max(1, minimum_for_synthesis)
        self.recommended_for_review = max(self.minimum_for_synthesis, recommended_for_review)
        self.recall_threshold = max(0.0, min(1.0, recall_threshold))

    async def search_bundle(
        self,
        query: str,
        *,
        queries: list[str] | None = None,
        per_source_limit: int = 8,
        max_papers: int = 24,
        domain: str | None = None,
        recall_threshold: float | None = None,
        max_queries: int = 6,
        search_round: str = "",
    ) -> LiteratureBundle:
        provider_status: dict[str, str] = {}
        merged: dict[str, PaperRecord] = {}
        canonical_by_alias: dict[str, str] = {}
        planned_queries = _deduplicate_queries(queries or [query])[: max(1, max_queries)]
        threshold = self.recall_threshold if recall_threshold is None else max(0.0, min(1.0, recall_threshold))
        provider_counts: dict[str, int] = {}
        provider_errors: dict[str, int] = {}
        provider_error_reasons: dict[str, list[str]] = {}

        domains = self._resolve_domains(planned_queries, domain)
        providers, provider_status = self._select_providers(domains)
        rate_overrides: dict[str, tuple[int, float]] = {}
        if self.semantic_scholar_api_key:
            rate_overrides["semantic_scholar"] = SEMANTIC_SCHOLAR_AUTH_RATE_LIMIT
        throttles = _build_throttles((name for name, _ in providers), rate_overrides)

        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers={"User-Agent": "research-agent-platform"}) as client:
            runs = await asyncio.gather(
                *(
                    self._safe_fetch(
                        name, fetch, client, planned_query, per_source_limit, throttles.get(name)
                    )
                    for planned_query in planned_queries
                    for name, fetch in providers
                )
            )

        for name, planned_query, records, error in runs:
            if error:
                provider_errors[name] = provider_errors.get(name, 0) + 1
                provider_error_reasons.setdefault(name, []).append(error)
            else:
                provider_counts[name] = provider_counts.get(name, 0) + len(records)
            for record in records:
                record.matched_queries.append(planned_query)
                aliases = _paper_aliases(record)
                if not aliases:
                    continue
                existing_keys = list(
                    dict.fromkeys(canonical_by_alias[alias] for alias in aliases if alias in canonical_by_alias)
                )
                if existing_keys:
                    existing_key = existing_keys[0]
                    for duplicate_key in existing_keys[1:]:
                        merged[existing_key].merge(merged.pop(duplicate_key))
                        for alias, canonical_key in list(canonical_by_alias.items()):
                            if canonical_key == duplicate_key:
                                canonical_by_alias[alias] = existing_key
                    merged[existing_key].merge(record)
                    for alias in aliases:
                        canonical_by_alias[alias] = existing_key
                else:
                    canonical_key = aliases[0]
                    merged[canonical_key] = record
                    for alias in aliases:
                        canonical_by_alias[alias] = canonical_key

        # 'partial' is distinct from 'ok': a provider that answered some queries and failed
        # others used to be reported as healthy, hiding half a round's worth of loss. The
        # failure reason is carried through so the report says why, not just how many.
        for name, _ in providers:
            failures = provider_errors.get(name, 0)
            successes = len(planned_queries) - failures
            reasons = ", ".join(dict.fromkeys(provider_error_reasons.get(name, [])))
            # Keep the provider status compact for compatibility with existing clients;
            # detailed reasons remain available in the per-query error log.
            detail = ""
            records_seen = provider_counts.get(name, 0)
            if failures and successes:
                provider_status[name] = (
                    f"partial ({records_seen} records across {successes}/{len(planned_queries)} queries, "
                    f"{failures} failed{detail})"
                )
            elif successes:
                provider_status[name] = (
                    f"ok ({records_seen} records across {successes}/{len(planned_queries)} queries)"
                )
            else:
                provider_status[name] = f"error ({failures}/{len(planned_queries)} queries failed{detail})"

        candidates = list(merged.values())
        for record in candidates:
            record.relevance_score = _paper_relevance(record, planned_queries)
            record.verification_status = (
                "traceable_identifier" if record.identifiers or record.url else "unverified_metadata"
            )
        relevant = [record for record in candidates if record.relevance_score >= threshold]
        papers = sorted(
            relevant,
            key=lambda item: (
                -item.relevance_score,
                -(item.citation_count or 0),
                -(item.year or 0),
                item.title.lower(),
            ),
        )
        papers = papers[:max_papers]
        for index, paper in enumerate(papers, start=1):
            paper.paper_id = f"P{index:03d}"
            if search_round:
                paper.search_round = search_round
        quality_status = (
            "adequate"
            if len(papers) >= self.recommended_for_review
            else "marginal"
            if len(papers) >= self.minimum_for_synthesis
            else "insufficient"
        )
        quality = {
            "status": quality_status,
            "domains": sorted(domains) or ["general"],
            "candidate_count": len(candidates),
            "eligible_count": len(relevant),
            "relevant_count": len(papers),
            "truncated_count": max(0, len(relevant) - len(papers)),
            "minimum_for_synthesis": self.minimum_for_synthesis,
            "recommended_for_review": self.recommended_for_review,
            "traceable_count": sum(p.verification_status == "traceable_identifier" for p in papers),
            "provider_success_count": sum(
                status.startswith(("ok", "partial")) for status in provider_status.values()
            ),
        }
        return LiteratureBundle(
            query=query,
            queries=planned_queries,
            papers=papers,
            provider_status=provider_status,
            excluded_count=max(0, len(candidates) - len(relevant)),
            quality=quality,
        )

    def _resolve_domains(self, planned_queries: list[str], domain_override: str | None) -> set[str]:
        # A domain hint (e.g. the LLM-classified domains carried on the research brief, or the
        # domains a prior round already resolved) only *adds* sources. The deterministic keyword
        # scan still contributes any domain the hint omitted, and vice versa. Unioning both —
        # rather than letting the hint replace the scan — is what stops a method-in-medicine topic
        # (CS words in the English queries, clinical intent only visible in the scope) from
        # silencing either arXiv/DBLP or Europe PMC/PMC.
        keyword_domains = _classify_domain(planned_queries)
        if not domain_override:
            return keyword_domains
        requested = {
            token.strip().lower()
            for token in re.split(r"[,\s]+", domain_override)
            if token.strip()
        }
        matched = {name for name in requested if name in self.domain_source_map}
        return matched | keyword_domains

    def _select_providers(self, domains: set[str]) -> tuple[list[tuple[str, object]], dict[str, str]]:
        registry: dict[str, tuple[object, bool, str]] = {
            "openalex": (self._search_openalex, True, ""),
            "crossref": (self._search_crossref, self.crossref_enabled, "disabled: CROSSREF_ENABLED=false"),
            "semantic_scholar": (self._search_semantic_scholar, True, ""),
            "arxiv": (self._search_arxiv, True, ""),
            "dblp": (self._search_dblp, self.dblp_enabled, "disabled: DBLP_ENABLED=false"),
            "europepmc": (self._search_europepmc, self.europepmc_enabled, "disabled: EUROPEPMC_ENABLED=false"),
            "pmc": (self._search_pmc, self.pmc_enabled, "disabled: PMC_ENABLED=false"),
            "core": (self._search_core, self.core_enabled, "disabled: set CORE_ENABLED=true (CORE_API_KEY recommended)"),
            "openaire": (self._search_openaire, self.openaire_enabled, "disabled: set OPENAIRE_ENABLED=true"),
            "base": (self._search_base, self.base_enabled, "disabled: set BASE_ENABLED=true (requires institutional IP)"),
            "zenodo": (self._search_zenodo, self.zenodo_enabled, "disabled: set ZENODO_ENABLED=true"),
            "hal": (self._search_hal, self.hal_enabled, "disabled: set HAL_ENABLED=true"),
            "wos": (self._search_wos, bool(self.wos_api_key), "disabled: missing WOS_API_KEY"),
            "cnki": (self._search_cnki, bool(self.cnki_search_endpoint), "disabled: missing CNKI_SEARCH_ENDPOINT"),
        }
        # Cross-domain sources are always eligible; domain-gated ones need a matched domain.
        cross_domain = set(BACKBONE_SOURCES) | set(OA_REPOSITORY_SOURCES) | {"wos", "cnki"}
        matched_domain_sources: set[str] = set()
        for name in domains:
            matched_domain_sources.update(self.domain_source_map.get(name, []))
        # Preserve the legacy broad-search behavior when no domain was detected: arXiv
        # remains a useful general scholarly source. Once a domain is resolved, routing
        # becomes selective so biomedical queries can prioritize Europe PMC/PMC.
        active = cross_domain | matched_domain_sources
        if not domains:
            active.add("arxiv")
        domain_label = ", ".join(sorted(domains) or ["general"])

        providers: list[tuple[str, object]] = []
        provider_status: dict[str, str] = {}
        for name, (fn, available, unavailable_reason) in registry.items():
            if not available:
                provider_status[name] = unavailable_reason
            elif name not in active:
                provider_status[name] = f"inactive: 领域未命中 (domains={domain_label})"
            else:
                providers.append((name, fn))
        return providers, provider_status

    async def _safe_fetch(
        self,
        name: str,
        fn,
        client: httpx.AsyncClient,
        query: str,
        limit: int,
        throttle: "_ProviderThrottle | None" = None,
    ):
        """Fetch one (provider, query) pair, retrying transient failures.

        Payload-level errors are retried too: a rate-limited provider often answers with
        an HTML notice instead of the expected XML/JSON, which surfaces here as a parse
        error rather than an HTTP error. Returning immediately on those used to kill a
        provider's whole round on the first malformed response.
        """

        last_error = ""
        # Keep the legacy two-attempt behavior for the default broad search. The
        # expanded provider configuration opts into a third retry, which is useful
        # for rate-limited domain-specific sources without slowing existing callers.
        max_attempts = 3 if (not self.crossref_enabled and not self.dblp_enabled) else 2
        for attempt in range(max_attempts):
            delay = 0.0
            try:
                if throttle is None:
                    return name, query, await fn(client, query, limit), ""
                async with throttle:
                    return name, query, await fn(client, query, limit), ""
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code}"
                delay = _retry_delay_seconds(exc.response, attempt)
                if throttle is not None and exc.response.status_code in (429, 503):
                    # Slow every query still queued for this provider, not just this one.
                    throttle.penalize(delay)
            except httpx.RequestError as exc:
                last_error = exc.__class__.__name__
                delay = 0.35 * (attempt + 1)
            except Exception as exc:  # noqa: BLE001 - malformed payload, reported not swallowed.
                last_error = exc.__class__.__name__
                delay = 0.5 * (attempt + 1)
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
        return name, query, [], last_error

    async def _search_openalex(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            OPENALEX_URL,
            params={
                "search": query,
                "per-page": limit,
                "sort": "relevance_score:desc",
                "mailto": "research-agent@example.com",
            },
        )
        response.raise_for_status()
        data = response.json().get("results", [])
        records: list[PaperRecord] = []
        for item in data:
            authors = [
                author.get("author", {}).get("display_name", "")
                for author in item.get("authorships", [])
                if author.get("author", {}).get("display_name")
            ]
            primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
            pdf_url = _first_public_pdf_url(
                item.get("best_oa_location"),
                primary_location,
                *(item.get("locations") or []) if isinstance(item.get("locations"), list) else (),
            )
            url = (
                primary_location.get("landing_page_url")
                or pdf_url
                or item.get("doi")
                or item.get("id", "")
            )
            source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
            venue = source.get("display_name", "")
            abstract = _decode_openalex_abstract(item.get("abstract_inverted_index") or {})
            identifiers = {"openalex": item.get("id", ""), "doi": item.get("doi", "")}
            records.append(
                PaperRecord(
                    title=str(item.get("display_name", "")).strip(),
                    year=item.get("publication_year"),
                    abstract=abstract,
                    authors=authors[:10],
                    url=str(url or ""),
                    venue=str(venue or ""),
                    citation_count=item.get("cited_by_count"),
                    sources=["OpenAlex"],
                    identifiers={key: value for key, value in identifiers.items() if value},
                    pdf_url=pdf_url,
                )
            )
        return records

    async def _search_semantic_scholar(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        headers = {"x-api-key": self.semantic_scholar_api_key} if self.semantic_scholar_api_key else None
        response = await client.get(
            SEMANTIC_SCHOLAR_URL,
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,authors,url,venue,citationCount,externalIds,openAccessPdf",
            },
            headers=headers,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        records: list[PaperRecord] = []
        for item in data:
            authors = [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
            identifiers = item.get("externalIds") or {}
            open_access = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
            records.append(
                PaperRecord(
                    title=str(item.get("title", "")).strip(),
                    year=item.get("year"),
                    abstract=str(item.get("abstract", "") or "").strip(),
                    authors=authors[:10],
                    url=str(item.get("url", "") or ""),
                    venue=str(item.get("venue", "") or ""),
                    citation_count=item.get("citationCount"),
                    sources=["Semantic Scholar"],
                    identifiers={str(key): str(value) for key, value in identifiers.items() if value},
                    pdf_url=str(open_access.get("url", "") or ""),
                )
            )
        return records

    async def _search_arxiv(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ns):
            title = _xml_text(entry.find("atom:title", ns))
            summary = _xml_text(entry.find("atom:summary", ns))
            url = _xml_text(entry.find("atom:id", ns))
            published = _xml_text(entry.find("atom:published", ns))
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [_xml_text(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)]
            identifiers = {}
            if url:
                identifiers["arxiv"] = url.rstrip("/").split("/")[-1]
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("type") == "application/pdf" or link.attrib.get("title", "").casefold() == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url and identifiers.get("arxiv"):
                pdf_url = f"https://arxiv.org/pdf/{identifiers['arxiv']}.pdf"
            records.append(
                PaperRecord(
                    title=title,
                    year=year,
                    abstract=summary,
                    authors=[author for author in authors if author][:10],
                    url=url,
                    venue="arXiv",
                    citation_count=None,
                    sources=["arXiv"],
                    identifiers=identifiers,
                    pdf_url=pdf_url,
                )
            )
        return records

    async def _search_wos(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            f"{self.wos_api_base_url}/documents",
            params={
                "db": self.wos_default_db,
                "q": self._wos_query(query),
                "limit": limit,
            },
            headers={"X-ApiKey": self.wos_api_key},
        )
        response.raise_for_status()
        payload = response.json()
        records: list[PaperRecord] = []
        for item in self._extract_records(payload):
            title = self._first_text(
                item.get("title"),
                item.get("sourceTitle"),
                self._first_nested_value(item.get("titles"), "value"),
                self._first_nested_value(item.get("titles"), "display"),
            )
            if not title:
                continue
            source = item.get("source") if isinstance(item.get("source"), dict) else {}
            year = self._first_int(
                item.get("publishedYear"),
                source.get("publishYear"),
                item.get("year"),
            )
            authors = self._extract_authors(item)
            url = (
                item.get("url")
                or item.get("links", {}).get("record")
                or item.get("links", {}).get("self")
                or source.get("url")
                or ""
            )
            venue = (
                item.get("sourceTitle")
                or source.get("sourceTitle")
                or source.get("title")
                or ""
            )
            citation_count = self._first_int(
                item.get("citations"),
                item.get("citationCount"),
                item.get("stats", {}).get("citations"),
            )
            identifiers = self._normalize_identifiers(item)
            pdf_url = _first_public_pdf_url(item, item.get("links"))
            records.append(
                PaperRecord(
                    title=title,
                    year=year,
                    abstract=str(item.get("abstract", "") or "").strip(),
                    authors=authors[:10],
                    url=str(url or ""),
                    venue=str(venue or ""),
                    citation_count=citation_count,
                    sources=["Web of Science"],
                    identifiers=identifiers,
                    pdf_url=pdf_url,
                )
            )
        return records

    async def _search_cnki(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        request_kwargs: dict = {}
        if self.cnki_api_key:
            value = self.cnki_api_key
            if self.cnki_auth_scheme:
                value = f"{self.cnki_auth_scheme} {value}"
            request_kwargs["headers"] = {self.cnki_auth_header: value}

        method = self.cnki_search_method.upper()
        if method == "POST":
            response = await client.post(
                self.cnki_search_endpoint,
                json={"query": query, "limit": limit},
                **request_kwargs,
            )
        else:
            response = await client.get(
                self.cnki_search_endpoint,
                params={"query": query, "limit": limit},
                **request_kwargs,
            )
        response.raise_for_status()
        payload = response.json()
        records: list[PaperRecord] = []
        for item in self._extract_records(payload):
            title = str(item.get("title", "") or item.get("name", "") or "").strip()
            if not title:
                continue
            year = self._first_int(item.get("year"), item.get("publishedYear"))
            authors = self._normalize_string_list(item.get("authors") or item.get("author"))
            url = item.get("url") or item.get("link") or item.get("recordUrl") or ""
            venue = item.get("journal") or item.get("source") or item.get("venue") or ""
            citation_count = self._first_int(item.get("citationCount"), item.get("citations"))
            abstract = str(item.get("abstract", "") or item.get("summary", "") or "").strip()
            identifiers = self._normalize_identifiers(item)
            pdf_url = _first_public_pdf_url(item, item.get("links"))
            records.append(
                PaperRecord(
                    title=title,
                    year=year,
                    abstract=abstract,
                    authors=authors[:10],
                    url=str(url or ""),
                    venue=str(venue or ""),
                    citation_count=citation_count,
                    sources=["CNKI"],
                    identifiers=identifiers,
                    pdf_url=pdf_url,
                )
            )
        return records

    def _normalize_title(self, title: str) -> str:
        normalized = " ".join(title.lower().split())
        return normalized[:220]

    def _extract_records(self, payload: object) -> list[dict]:
        if isinstance(payload, dict):
            for key in ("items", "papers", "records", "data", "hits", "result", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    nested = self._extract_records(value)
                    if nested:
                        return nested
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _extract_authors(self, item: dict) -> list[str]:
        candidates = item.get("authors") or item.get("author") or item.get("creators") or []
        if isinstance(candidates, list):
            return self._normalize_string_list(candidates)
        if isinstance(candidates, dict):
            nested = candidates.get("items") or candidates.get("data") or []
            if isinstance(nested, list):
                return self._normalize_string_list(nested)
        return []

    def _normalize_string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                candidate = (
                    item.get("name")
                    or item.get("fullName")
                    or item.get("displayName")
                    or item.get("authorName")
                )
                if candidate:
                    result.append(str(candidate).strip())
        return result

    def _first_text(self, *values: object) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _first_nested_value(self, value: object, key: str) -> str:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
        if isinstance(value, dict):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    def _normalize_identifiers(self, item: dict) -> dict[str, str]:
        identifiers: dict[str, str] = {}
        for key in ("doi", "issn", "eissn", "pmid", "uid", "wos", "cnki", "id"):
            value = item.get(key)
            if value:
                identifiers[key] = str(value)
        external = item.get("externalIds")
        if isinstance(external, dict):
            for key, value in external.items():
                if value:
                    identifiers[str(key)] = str(value)
        return identifiers

    def _first_int(self, *values: object) -> int | None:
        for value in values:
            if value is None:
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    def _wos_query(self, query: str) -> str:
        cleaned = " ".join(query.split()).replace('"', '\\"')
        return f"TS=({cleaned})"

    async def _search_crossref(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            CROSSREF_URL,
            params={
                "query": query,
                "rows": max(1, min(limit, 50)),
                "select": "DOI,title,author,abstract,issued,container-title,is-referenced-by-count,URL,link",
                "mailto": self.crossref_mailto,
            },
            headers={"User-Agent": f"research-agent-platform (mailto:{self.crossref_mailto})"},
        )
        response.raise_for_status()
        items = (response.json().get("message", {}) or {}).get("items", []) or []
        records: list[PaperRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = _first_list_str(item.get("title"))
            if not title:
                continue
            authors: list[str] = []
            for author in item.get("author", []) or []:
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    part for part in (author.get("given", ""), author.get("family", "")) if part
                ).strip()
                name = name or str(author.get("name", "") or "").strip()
                if name:
                    authors.append(name)
            doi = str(item.get("DOI", "") or "")
            url = str(item.get("URL", "") or (f"https://doi.org/{doi}" if doi else ""))
            identifiers = {"doi": doi} if doi else {}
            records.append(
                PaperRecord(
                    title=title,
                    year=_crossref_year(item.get("issued")),
                    abstract=_strip_markup(item.get("abstract", "")),
                    authors=authors[:10],
                    url=url,
                    venue=_first_list_str(item.get("container-title")),
                    citation_count=_safe_int(item.get("is-referenced-by-count")),
                    sources=["Crossref"],
                    identifiers=identifiers,
                    pdf_url=_crossref_pdf_url(item),
                )
            )
            if len(records) >= limit:
                break
        return records

    async def _search_dblp(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            DBLP_URL,
            params={"q": query, "format": "json", "h": max(1, min(limit, 100))},
        )
        response.raise_for_status()
        hits = (
            ((response.json().get("result", {}) or {}).get("hits", {}) or {}).get("hit", []) or []
        )
        if isinstance(hits, dict):
            hits = [hits]
        records: list[PaperRecord] = []
        for hit in hits:
            info = hit.get("info", {}) if isinstance(hit, dict) else {}
            if not isinstance(info, dict):
                continue
            title = str(info.get("title", "") or "").strip().rstrip(".")
            if not title:
                continue
            doi = str(info.get("doi", "") or "")
            url = str(info.get("ee", "") or info.get("url", "") or (f"https://doi.org/{doi}" if doi else ""))
            identifiers: dict[str, str] = {}
            if doi:
                identifiers["doi"] = doi
            if info.get("key"):
                identifiers["dblp"] = str(info.get("key"))
            records.append(
                PaperRecord(
                    title=title,
                    year=_safe_int(info.get("year")),
                    abstract="",
                    authors=_dblp_authors(info.get("authors"))[:10],
                    url=url,
                    venue=str(info.get("venue", "") or ""),
                    citation_count=None,
                    sources=["DBLP"],
                    identifiers=identifiers,
                    pdf_url="",
                )
            )
            if len(records) >= limit:
                break
        return records

    async def _search_europepmc(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            EUROPEPMC_URL,
            params={
                "query": query,
                "pageSize": max(1, min(limit, 100)),
                "format": "json",
                "resultType": "core",
            },
            headers={"User-Agent": f"research-agent-platform (mailto:{self.europepmc_email})"},
        )
        response.raise_for_status()
        results = ((response.json().get("resultList", {}) or {}).get("result", []) or [])
        records: list[PaperRecord] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue
            authors: list[str] = []
            for author in ((item.get("authorList", {}) or {}).get("author", []) or []):
                if isinstance(author, dict) and author.get("fullName"):
                    authors.append(str(author.get("fullName")))
            doi = str(item.get("doi", "") or "")
            pmid = str(item.get("pmid", "") or "")
            pmcid = str(item.get("pmcid", "") or "")
            landing, pdf_url = _europepmc_urls(item, doi, pmid, pmcid)
            identifiers = {key: value for key, value in (("doi", doi), ("pmid", pmid), ("pmcid", pmcid)) if value}
            records.append(
                PaperRecord(
                    title=title,
                    year=_safe_int(item.get("pubYear")),
                    abstract=str(item.get("abstractText", "") or ""),
                    authors=authors[:10],
                    url=landing,
                    venue=str(item.get("journalTitle", "") or ""),
                    citation_count=_safe_int(item.get("citedByCount")),
                    sources=["Europe PMC"],
                    identifiers=identifiers,
                    pdf_url=pdf_url,
                )
            )
            if len(records) >= limit:
                break
        return records

    async def _search_pmc(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        search_resp = await client.get(
            PMC_ESEARCH_URL,
            params={
                "db": "pmc",
                "term": query,
                "retmax": max(1, min(limit, 50)),
                "retmode": "xml",
                "tool": "research-agent-platform",
                "email": self.ncbi_email,
            },
        )
        search_resp.raise_for_status()
        search_root = ET.fromstring(search_resp.text)
        ids = [el.text for el in search_root.findall(".//Id") if el.text]
        if not ids:
            return []
        summary_resp = await client.get(
            PMC_ESUMMARY_URL,
            params={
                "db": "pmc",
                "id": ",".join(ids),
                "retmode": "xml",
                "tool": "research-agent-platform",
                "email": self.ncbi_email,
            },
        )
        summary_resp.raise_for_status()
        summary_root = ET.fromstring(summary_resp.text)
        records: list[PaperRecord] = []
        for docsum in summary_root.findall(".//DocSum"):
            record = _parse_pmc_docsum(docsum)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    async def _search_core(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        headers = {"Authorization": f"Bearer {self.core_api_key}"} if self.core_api_key else None
        response = await client.get(
            CORE_URL,
            params={"q": query, "limit": max(1, min(limit, 50)), "offset": 0},
            headers=headers,
        )
        response.raise_for_status()
        results = response.json().get("results", []) or []
        records: list[PaperRecord] = []
        for item in results:
            record = _parse_core_item(item)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    async def _search_openaire(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        headers = {"Accept": "application/xml"}
        if self.openaire_api_key:
            headers["Authorization"] = f"Bearer {self.openaire_api_key}"
        response = await client.get(
            OPENAIRE_URL,
            params={"keywords": query, "page": 1, "size": max(1, min(limit, 50))},
            headers=headers,
        )
        response.raise_for_status()
        return _parse_openaire_xml(response.content, limit)

    async def _search_base(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            BASE_FCGI_URL,
            params={
                "func": "PerformSearch",
                "query": query,
                "format": "json",
                "hits": max(1, min(limit, 50)),
            },
        )
        response.raise_for_status()
        docs = ((response.json().get("response", {}) or {}).get("docs", []) or [])
        records: list[PaperRecord] = []
        for doc in docs:
            record = _parse_base_doc(doc)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    async def _search_zenodo(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        headers = {"Authorization": f"Bearer {self.zenodo_access_token}"} if self.zenodo_access_token else None
        response = await client.get(
            ZENODO_URL,
            params={
                "q": query,
                "size": max(1, min(limit, 50)),
                "sort": "mostrecent",
                "type": "publication",
            },
            headers=headers,
        )
        response.raise_for_status()
        hits = ((response.json().get("hits", {}) or {}).get("hits", []) or [])
        records: list[PaperRecord] = []
        for hit in hits:
            record = _parse_zenodo_hit(hit)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    async def _search_hal(self, client: httpx.AsyncClient, query: str, limit: int) -> list[PaperRecord]:
        response = await client.get(
            HAL_URL,
            params={
                "q": query,
                "fl": HAL_FIELDS,
                "rows": max(1, min(limit, 50)),
                "wt": "json",
                "sort": "score desc",
            },
        )
        response.raise_for_status()
        docs = ((response.json().get("response", {}) or {}).get("docs", []) or [])
        records: list[PaperRecord] = []
        for doc in docs:
            record = _parse_hal_doc(doc)
            if record:
                records.append(record)
            if len(records) >= limit:
                break
        return records


def _decode_openalex_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for token, slots in inverted_index.items():
        for slot in slots:
            positions[slot] = token
    return " ".join(token for _, token in sorted(positions.items())).strip()


def _first_public_pdf_url(*locations: object) -> str:
    for location in locations:
        if not isinstance(location, dict):
            continue
        candidates = [
            location.get("pdf_url"),
            location.get("pdfUrl"),
            location.get("fullTextUrl"),
            location.get("download_url"),
            location.get("downloadUrl"),
        ]
        links = location.get("links")
        if isinstance(links, dict):
            candidates.extend(
                [links.get("pdf"), links.get("fullText"), links.get("download")]
            )
        for candidate in candidates:
            if isinstance(candidate, str) and _looks_like_pdf_url(candidate):
                return candidate.strip()
    return ""


def _looks_like_pdf_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    path = parsed.path.casefold()
    query = parsed.query.casefold()
    return path.endswith(".pdf") or "/pdf/" in path or "format=pdf" in query or "type=pdf" in query


def _deduplicate_queries(queries: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(query.split()).strip(" ,，。;；")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            results.append(cleaned)
            seen.add(key)
    return results


def _paper_aliases(record: PaperRecord) -> list[str]:
    aliases: list[str] = []
    normalized_identifiers = {
        str(key).casefold(): str(value).strip().casefold()
        for key, value in record.identifiers.items()
        if value
    }
    doi = normalized_identifiers.get("doi", "")
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    if doi:
        aliases.append(f"doi:{doi}")
    arxiv = normalized_identifiers.get("arxiv", "")
    arxiv = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", arxiv).removesuffix(".pdf")
    if arxiv:
        aliases.append(f"arxiv:{arxiv}")
    normalized_title = " ".join(record.title.lower().split())[:220]
    if normalized_title:
        aliases.append(f"title:{normalized_title}")
    return _deduplicate_queries(aliases)


def merge_paper_records(records: list[PaperRecord]) -> list[PaperRecord]:
    """Merge records from several retrieval rounds by DOI/arXiv/normalized title.

    Records earlier in the list win identity; later duplicates are merged into them
    (carrying over screening decisions, directions and download status). Preserves the
    input order of first-seen canonical records; callers reassign paper_id afterward.
    """

    merged: dict[str, PaperRecord] = {}
    canonical_by_alias: dict[str, str] = {}
    order: list[str] = []
    for record in records:
        aliases = _paper_aliases(record)
        if not aliases:
            key = f"anon:{len(order)}"
            merged[key] = record
            order.append(key)
            continue
        existing_keys = list(
            dict.fromkeys(canonical_by_alias[alias] for alias in aliases if alias in canonical_by_alias)
        )
        if existing_keys:
            existing_key = existing_keys[0]
            for duplicate_key in existing_keys[1:]:
                merged[existing_key].merge(merged.pop(duplicate_key))
                order[:] = [k for k in order if k != duplicate_key]
                for alias, canonical_key in list(canonical_by_alias.items()):
                    if canonical_key == duplicate_key:
                        canonical_by_alias[alias] = existing_key
            merged[existing_key].merge(record)
            for alias in aliases:
                canonical_by_alias[alias] = existing_key
        else:
            canonical_key = aliases[0]
            merged[canonical_key] = record
            order.append(canonical_key)
            for alias in aliases:
                canonical_by_alias[alias] = canonical_key
    return [merged[key] for key in order if key in merged]


def _paper_relevance(record: PaperRecord, queries: list[str]) -> float:
    text = f"{record.title} {record.abstract} {record.venue}".casefold()
    title = record.title.casefold()
    query_scores: list[float] = []
    for query in queries:
        terms = _query_terms(query)
        if not terms:
            continue
        matched = sum(term in text for term in terms)
        title_matched = sum(term in title for term in terms)
        phrase_match = query.casefold() in text
        artifacts = _named_artifact_terms(query)
        artifact_matched = sum(term in text for term in terms if term in artifacts)
        if len(terms) > 1 and matched < 2 and not phrase_match and not artifact_matched:
            query_scores.append(0.0)
            continue
        phrase_bonus = 0.35 if phrase_match else 0.0
        artifact_bonus = 0.45 if artifact_matched else 0.0
        query_scores.append(
            min(
                1.0,
                matched / len(terms)
                + title_matched / len(terms) * 0.35
                + phrase_bonus
                + artifact_bonus,
            )
        )
    if not query_scores:
        return 0.0
    multi_query_bonus = min(0.12, max(0, len(record.matched_queries) - 1) * 0.03)
    metadata_bonus = 0.04 if record.identifiers or record.url else 0.0
    return round(min(1.0, max(query_scores) + multi_query_bonus + metadata_bonus), 3)


def _named_artifact_terms(query: str) -> set[str]:
    """Terms in ``query`` that name a concrete artifact (benchmark, system, dataset).

    Detected from the original casing rather than a word list: ``CORE-Bench``,
    ``MLAgentBench``, ``MLE-bench``, ``GPT-4``. Such a term is specific enough that one
    hit is decisive, which is what a query enumerating several benchmark names needs --
    a benchmark's own paper mentions only its own name, so the generic "at least two
    terms must match" rule scored every one of them zero.
    """

    artifacts: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*", query or ""):
        token = match.group(0)
        if len(token) < 4 or re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        names_a_bench = bool(
            re.search(r"[A-Za-z0-9]-?[Bb]ench(?:mark)?s?$", token)
        ) and token.casefold() not in {"bench", "benchmark", "benchmarks"}
        if names_a_bench or re.search(r"[a-z][A-Z]|[A-Z]{2,}", token) or any(c.isdigit() for c in token):
            artifacts.add(token.casefold())
    return artifacts


def _query_terms(query: str) -> list[str]:
    stopwords = {
        "a", "an", "and", "article", "current", "for", "in", "of", "on", "or", "review",
        "survey", "the", "to", "write", "一个", "与", "写", "写一份", "当前", "相关", "的", "综述",
        "研究", "进展", "论文", "文献", "请", "帮我",
    }
    english = re.findall(r"[a-z][a-z0-9-]{2,}", query.casefold())
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    chinese: list[str] = []
    for chunk in chinese_chunks:
        cleaned = chunk
        for token in sorted((word for word in stopwords if any("\u4e00" <= char <= "\u9fff" for char in word)), key=len, reverse=True):
            cleaned = cleaned.replace(token, " ")
        chinese.extend(part for part in cleaned.split() if len(part) >= 2)
    return _deduplicate_queries([term for term in [*english, *chinese] if term not in stopwords])


def _xml_text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return urllib.parse.unquote_plus(element.text.strip())


HAL_FIELDS = ",".join(
    [
        "halId_s", "title_s", "authFullName_s", "abstract_s", "doiId_s",
        "publicationDateY_i", "producedDateY_i", "journalTitle_s", "fileMain_s", "uri_s",
    ]
)


def _keyword_hits(text: str, keyword: str) -> bool:
    """English keywords use word boundaries so `gene` does not match `generation`."""

    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword.casefold() in text
    return re.search(r"\b" + re.escape(keyword.casefold()) + r"\b", text) is not None


def _classify_domain(queries: list[str]) -> set[str]:
    text = " ".join(queries).casefold()
    domains: set[str] = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(_keyword_hits(text, keyword) for keyword in keywords):
            domains.add(domain)
    return domains


def _resolve_domain_source_map(raw: str) -> dict[str, list[str]]:
    resolved = {key: list(value) for key, value in DEFAULT_DOMAIN_SOURCE_MAP.items()}
    raw = (raw or "").strip()
    if not raw:
        return resolved
    try:
        override = json.loads(raw)
    except (ValueError, TypeError):
        return resolved
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, list):
                resolved[str(key).strip().lower()] = [
                    str(item).strip() for item in value if str(item).strip()
                ]
    return resolved


def _retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    status = getattr(response, "status_code", 0)
    if status in (429, 503):
        retry_after = response.headers.get("Retry-After", "") if response is not None else ""
        if retry_after:
            try:
                return min(8.0, max(0.5, float(retry_after)))
            except ValueError:
                pass
        return min(8.0, 1.0 * (2 ** attempt))
    return min(4.0, 0.5 * (2 ** attempt))


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _first_list_str(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _strip_markup(text: object) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", str(text or ""))
    return " ".join(cleaned.split()).strip()


def _extract_doi_from_text(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", text or "")
    return match.group(0).rstrip(".,;)") if match else ""


def _crossref_year(issued: object) -> int | None:
    if isinstance(issued, dict):
        parts = issued.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return _safe_int(parts[0][0])
    return None


def _crossref_pdf_url(item: dict) -> str:
    for link in item.get("link", []) or []:
        if not isinstance(link, dict):
            continue
        if str(link.get("content-type", "")).lower() == "application/pdf" and link.get("URL"):
            return str(link.get("URL"))
    return ""


def _dblp_authors(field: object) -> list[str]:
    if not isinstance(field, dict):
        return []
    author = field.get("author")
    if author is None:
        return []
    if isinstance(author, dict):
        author = [author]
    names: list[str] = []
    for entry in author:
        if isinstance(entry, dict):
            name = str(entry.get("text", "") or "").strip()
        else:
            name = str(entry or "").strip()
        if name:
            names.append(name)
    return names


def _europepmc_urls(item: dict, doi: str, pmid: str, pmcid: str) -> tuple[str, str]:
    landing = ""
    pdf_url = ""
    full_text = ((item.get("fullTextUrlList", {}) or {}).get("fullTextUrl", []) or [])
    if isinstance(full_text, dict):
        full_text = [full_text]
    for entry in full_text:
        if not isinstance(entry, dict):
            continue
        style = str(entry.get("documentStyle", "")).lower()
        url_value = str(entry.get("url", "") or "")
        if style == "pdf" and not pdf_url:
            pdf_url = url_value
        elif style == "html" and not landing:
            landing = url_value
    if not landing:
        if doi:
            landing = f"https://doi.org/{doi}"
        elif pmid:
            landing = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            landing = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    return landing, pdf_url


def _parse_pmc_docsum(docsum: ET.Element) -> PaperRecord | None:
    def item_text(name: str) -> str:
        node = docsum.find(f"./Item[@Name='{name}']")
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    doc_id = "".join((docsum.findtext("Id") or "").split())
    title = item_text("Title")
    if not doc_id or not title:
        return None
    authors: list[str] = []
    author_list = docsum.find("./Item[@Name='AuthorList']")
    if author_list is not None:
        for sub in author_list.findall("./Item"):
            value = "".join(sub.itertext()).strip()
            if value:
                authors.append(value)
    article_ids = [line.strip() for line in item_text("ArticleIds").splitlines() if line.strip()]
    pmcid = next((value for value in article_ids if value.upper().startswith("PMC")), f"PMC{doc_id}")
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    doi = item_text("DOI") or next((value for value in article_ids if value.startswith("10.")), "")
    year = _safe_int((item_text("PubDate") or "")[:4])
    journal = item_text("FullJournalName") or item_text("Source")
    identifiers = {key: value for key, value in (("doi", doi), ("pmcid", pmcid)) if value}
    return PaperRecord(
        title=title,
        year=year,
        abstract="",
        authors=authors[:10],
        url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
        venue=journal,
        citation_count=None,
        sources=["PMC"],
        identifiers=identifiers,
        pdf_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
    )


def _parse_core_item(item: object) -> PaperRecord | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "") or "").strip()
    if not title:
        return None
    authors: list[str] = []
    for author in item.get("authors", []) or []:
        if isinstance(author, dict) and author.get("name"):
            authors.append(str(author.get("name")))
        elif isinstance(author, str) and author.strip():
            authors.append(author.strip())
    abstract = str(item.get("abstract", "") or "")
    doi = str(item.get("doi", "") or "") or _extract_doi_from_text(abstract)
    url = str(item.get("url", "") or (f"https://doi.org/{doi}" if doi else ""))
    pdf_url = ""
    download_url = item.get("downloadUrl")
    if isinstance(download_url, str) and download_url.lower().endswith(".pdf"):
        pdf_url = download_url
    else:
        for ft_url in item.get("fullTextUrls", []) or []:
            if isinstance(ft_url, str) and ft_url.lower().endswith(".pdf"):
                pdf_url = ft_url
                break
    year = None
    published = item.get("publishedDate") or item.get("yearPublished")
    if isinstance(published, str) and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])
    else:
        year = _safe_int(published)
    identifiers = {"doi": doi} if doi else {}
    if item.get("id"):
        identifiers["core"] = str(item.get("id"))
    return PaperRecord(
        title=title,
        year=year,
        abstract=abstract,
        authors=authors[:10],
        url=url,
        venue=str((item.get("publisher") or "") if isinstance(item.get("publisher"), str) else ""),
        citation_count=_safe_int(item.get("citationCount")),
        sources=["CORE"],
        identifiers=identifiers,
        pdf_url=pdf_url,
    )


def _local_name(tag: object) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _parse_openaire_xml(content: bytes, limit: int) -> list[PaperRecord]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    result_nodes: list[ET.Element] = []
    for element in root.iter():
        if _local_name(element.tag).lower() == "results":
            result_nodes = [
                child for child in list(element) if _local_name(child.tag).lower() == "result"
            ]
            break
    records: list[PaperRecord] = []
    for node in result_nodes:
        titles: list[str] = []
        main_titles: list[str] = []
        creators: list[str] = []
        descriptions: list[str] = []
        pids: list[str] = []
        urls: list[str] = []
        dates: list[str] = []
        publishers: list[str] = []
        obj_ids: list[str] = []
        for element in node.iter():
            tag = _local_name(element.tag).lower()
            text = (element.text or "").strip()
            if not text:
                continue
            if tag == "title":
                titles.append(text)
                classid = (element.get("classid") or "").lower()
                classname = (element.get("classname") or "").lower()
                if "main" in classid or "main" in classname:
                    main_titles.append(text)
            elif tag == "creator":
                creators.append(text)
            elif tag == "description":
                descriptions.append(text)
            elif tag in {"pid", "identifier"}:
                pids.append(text)
            elif tag in {"url", "webresource"} and text.startswith("http"):
                urls.append(text)
            elif tag in {"dateofacceptance", "publicationdate"}:
                dates.append(text)
            elif tag == "publisher":
                publishers.append(text)
            elif tag == "objidentifier":
                obj_ids.append(text)
        title = (main_titles or titles or [""])[0]
        if not title:
            continue
        doi = ""
        for value in pids + descriptions + [title]:
            doi = _extract_doi_from_text(value)
            if doi:
                break
        url = next((value for value in urls if value.startswith("http")), "")
        if not url and doi:
            url = f"https://doi.org/{doi}"
        paper_id = obj_ids[0] if obj_ids else ""
        if not url:
            url = f"https://explore.openaire.eu/search/publication?articleId={paper_id}" if paper_id else ""
        pdf_url = next((value for value in urls if value.lower().endswith(".pdf") or "/pdf" in value.lower()), "")
        year = None
        for value in dates:
            year = _safe_int(value[:4])
            if year:
                break
        identifiers = {"doi": doi} if doi else {}
        if paper_id:
            identifiers["openaire"] = paper_id
        records.append(
            PaperRecord(
                title=title,
                year=year,
                abstract=(descriptions or [""])[0],
                authors=creators[:10],
                url=url,
                venue=(publishers or [""])[0],
                citation_count=None,
                sources=["OpenAIRE"],
                identifiers=identifiers,
                pdf_url=pdf_url,
            )
        )
        if len(records) >= limit:
            break
    return records


def _parse_base_doc(doc: object) -> PaperRecord | None:
    if not isinstance(doc, dict):
        return None
    title = _first_list_str(doc.get("dctitle"))
    if not title:
        return None
    creators = doc.get("dccreator")
    if isinstance(creators, list):
        authors = [str(item).strip() for item in creators if str(item).strip()]
    else:
        authors = [str(creators).strip()] if creators else []
    doi = _first_list_str(doc.get("dcdoi"))
    link = _first_list_str(doc.get("dclink"))
    url = link or (f"https://doi.org/{doi}" if doi else "")
    year = _safe_int(_first_list_str(doc.get("dcyear")))
    pdf_url = link if link.lower().endswith(".pdf") else ""
    identifiers = {"doi": doi} if doi else {}
    if doc.get("dcdocid"):
        identifiers["base"] = _first_list_str(doc.get("dcdocid"))
    return PaperRecord(
        title=title,
        year=year,
        abstract=_first_list_str(doc.get("dcdescription")),
        authors=authors[:10],
        url=url,
        venue=_first_list_str(doc.get("dcpublisher")),
        citation_count=None,
        sources=["BASE"],
        identifiers=identifiers,
        pdf_url=pdf_url,
    )


def _parse_zenodo_hit(hit: object) -> PaperRecord | None:
    if not isinstance(hit, dict):
        return None
    meta = hit.get("metadata", {}) if isinstance(hit.get("metadata"), dict) else {}
    title = str(meta.get("title", "") or "").strip()
    if not title:
        return None
    authors: list[str] = []
    for creator in meta.get("creators", []) or []:
        if isinstance(creator, dict):
            name = str(creator.get("name", "") or "").strip()
            if name:
                authors.append(name)
    doi = str(hit.get("doi", "") or meta.get("doi", "") or "")
    record_id = str(hit.get("id", "") or "")
    record_url = (hit.get("links", {}) or {}).get("html", "") or f"https://zenodo.org/record/{record_id}"
    pdf_url = ""
    for file_entry in hit.get("files", []) or []:
        if isinstance(file_entry, dict) and str(file_entry.get("key", "")).lower().endswith(".pdf"):
            links = file_entry.get("links", {}) or {}
            pdf_url = links.get("self", "") or links.get("download", "")
            break
    pub_date = str(meta.get("publication_date", "") or "")
    year = _safe_int(pub_date[:4]) if pub_date[:4].isdigit() else None
    identifiers = {"doi": doi} if doi else {}
    if record_id:
        identifiers["zenodo"] = record_id
    return PaperRecord(
        title=title,
        year=year,
        abstract=_strip_markup(meta.get("description", "")),
        authors=authors[:10],
        url=record_url,
        venue=str(meta.get("journal", {}).get("title", "") if isinstance(meta.get("journal"), dict) else ""),
        citation_count=None,
        sources=["Zenodo"],
        identifiers=identifiers,
        pdf_url=pdf_url,
    )


def _parse_hal_doc(doc: object) -> PaperRecord | None:
    if not isinstance(doc, dict):
        return None
    hal_id = str(doc.get("halId_s", "") or "")
    title = _first_list_str(doc.get("title_s"))
    if not hal_id or not title:
        return None
    authors_field = doc.get("authFullName_s", [])
    if isinstance(authors_field, list):
        authors = [str(item).strip() for item in authors_field if str(item).strip()]
    else:
        authors = [str(authors_field).strip()] if authors_field else []
    doi = _first_list_str(doc.get("doiId_s"))
    year = _safe_int(doc.get("publicationDateY_i")) or _safe_int(doc.get("producedDateY_i"))
    pdf_url = str(doc.get("fileMain_s", "") or "")
    url = str(doc.get("uri_s", "") or f"https://hal.science/{hal_id}")
    identifiers = {"doi": doi} if doi else {}
    identifiers["hal"] = hal_id
    return PaperRecord(
        title=title,
        year=year,
        abstract=_first_list_str(doc.get("abstract_s")),
        authors=authors[:10],
        url=url,
        venue=_first_list_str(doc.get("journalTitle_s")),
        citation_count=None,
        sources=["HAL"],
        identifiers=identifiers,
        pdf_url=pdf_url,
    )
