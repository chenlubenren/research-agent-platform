from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    load_dotenv(_project_root() / ".env", override=True)


_load_project_env()


def _docs_root() -> Path:
    return _project_root().parent


def _default_aris_repo_root() -> str:
    return str(_project_root())


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AppConfig(BaseModel):
    project_name: str = "research-agent-platform"
    enable_hitl: bool = True
    router_mode: str = "rule_first"
    graph_backend: str = "langgraph"
    memory_backend: str = "research_wiki"
    project_root: str = Field(default_factory=lambda: str(_project_root()))
    docs_root: str = Field(default_factory=lambda: str(_docs_root()))
    artifact_root: str = Field(default_factory=lambda: str(_project_root() / "agent-workspace"))
    state_root: str = Field(default_factory=lambda: str(_project_root() / ".agent-state"))
    prd_path: str = Field(default_factory=lambda: str(_docs_root() / "research-agent-prd.md"))
    tech_spec_path: str = Field(default_factory=lambda: str(_docs_root() / "research-agent-tech-spec.md"))
    aris_repo_root: str = Field(
        default_factory=lambda: os.getenv("ARIS_REPO_ROOT", _default_aris_repo_root())
    )
    upstream_base_url: str = Field(
        default_factory=lambda: os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    )
    upstream_api_key: str = Field(default_factory=lambda: os.getenv("UPSTREAM_API_KEY", ""))
    upstream_model: str = Field(default_factory=lambda: os.getenv("UPSTREAM_MODEL", ""))
    idea_generator_model: str = Field(
        default_factory=lambda: os.getenv("IDEA_GENERATOR_MODEL", "")
    )
    idea_critic_model: str = Field(
        default_factory=lambda: os.getenv("IDEA_CRITIC_MODEL", "")
    )
    idea_final_model: str = Field(
        default_factory=lambda: os.getenv("IDEA_FINAL_MODEL", "")
    )
    public_api_key: str = Field(default_factory=lambda: os.getenv("PUBLIC_API_KEY", ""))
    image_model: str = Field(default_factory=lambda: os.getenv("IMAGE_MODEL", "gpt-image-2"))
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    )
    image_request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("IMAGE_REQUEST_TIMEOUT_SECONDS", "300"))
    )
    presentation_template: str = Field(
        default_factory=lambda: os.getenv("PRESENTATION_TEMPLATE", "auto")
    )
    presentation_max_slides: int = Field(
        default_factory=lambda: int(os.getenv("PRESENTATION_MAX_SLIDES", "12"))
    )
    presentation_source_limit: int = Field(
        default_factory=lambda: int(os.getenv("PRESENTATION_SOURCE_LIMIT", "40"))
    )
    write_source_limit: int = Field(
        default_factory=lambda: int(os.getenv("WRITE_SOURCE_LIMIT", "50"))
    )
    upload_max_files: int = Field(
        default_factory=lambda: int(os.getenv("UPLOAD_MAX_FILES", "20"))
    )
    upload_max_file_mb: int = Field(
        default_factory=lambda: int(os.getenv("UPLOAD_MAX_FILE_MB", "100"))
    )
    scholar_request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("SCHOLAR_REQUEST_TIMEOUT_SECONDS", "30"))
    )
    scholar_results_per_source: int = Field(
        default_factory=lambda: int(os.getenv("SCHOLAR_RESULTS_PER_SOURCE", "8"))
    )
    review_minimum_sources: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_MINIMUM_SOURCES", "10"))
    )
    review_recommended_sources: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_RECOMMENDED_SOURCES", "15"))
    )
    review_download_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_DOWNLOAD_ENABLED", True)
    )
    review_download_limit: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DOWNLOAD_LIMIT", "30"))
    )
    review_download_max_mb: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DOWNLOAD_MAX_MB", "50"))
    )
    review_download_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REVIEW_DOWNLOAD_TIMEOUT_SECONDS", "30"))
    )
    # Concurrent PDF fetches during the download stage.
    review_download_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DOWNLOAD_CONCURRENCY", "6"))
    )
    # --- Two-round /review LLM stages (clarify -> discovery -> direction selection -> deep dive) ---
    # Interactive scope clarification (round-0). Requires enable_hitl to actually pause.
    review_clarify_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_CLARIFY_ENABLED", True)
    )
    # LLM facet/HyDE query planning that feeds the retrieval query list.
    review_query_planner_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_QUERY_PLANNER_ENABLED", True)
    )
    # Max query variants used for the first (discovery/broad) retrieval round.
    review_query_limit: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_QUERY_LIMIT", "8"))
    )
    # LLM keep/drop screening with per-paper recommendation reasons.
    review_screening_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_SCREENING_ENABLED", True)
    )
    # Loosened lexical prefilter so relevant papers are not dropped before the LLM screens them.
    review_recall_threshold: float = Field(
        default_factory=lambda: float(os.getenv("REVIEW_RECALL_THRESHOLD", "0.15"))
    )
    # Cap on candidates handed to a single screening LLM call.
    review_screen_max_candidates: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_SCREEN_MAX_CANDIDATES", "40"))
    )
    # Optional dedicated model for screening/direction JSON calls (defaults to upstream model).
    review_screen_model: str = Field(default_factory=lambda: os.getenv("REVIEW_SCREEN_MODEL", ""))
    # Layered screening: LLM adjudication batch size keeps each JSON call small enough to avoid timeout/truncation.
    review_screen_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_SCREEN_BATCH_SIZE", "8"))
    )
    # Per-call timeout (seconds) for a single screening batch; larger than the batch is small.
    review_screen_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REVIEW_SCREEN_TIMEOUT_SECONDS", "180"))
    )
    # Comma-separated hard-exclusion terms (L1); a title/abstract hit drops a paper before the LLM sees it.
    review_hard_exclusion_terms: str = Field(
        default_factory=lambda: os.getenv("REVIEW_HARD_EXCLUSION_TERMS", "")
    )
    # Relevance floor for L1 hard exclusion (0 disables the score-based hard drop).
    review_hard_relevance_floor: float = Field(
        default_factory=lambda: float(os.getenv("REVIEW_HARD_RELEVANCE_FLOOR", "0"))
    )
    # Optional pluggable embedding provider for semantic rerank (L2). OpenAI-compatible /embeddings.
    embedding_enabled: bool = Field(
        default_factory=lambda: _env_bool("EMBEDDING_ENABLED", False)
    )
    embedding_base_url: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", ""))
    embedding_api_key: str = Field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))
    # Max inputs per /embeddings request; Doubao caps this at 10.
    embedding_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    )
    # Post-retrieval direction discovery + user selection (round-1 focusing).
    review_direction_selection_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_DIRECTION_SELECTION_ENABLED", True)
    )
    review_direction_count: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DIRECTION_COUNT", "5"))
    )
    review_direction_max_candidates: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DIRECTION_MAX_CANDIDATES", "40"))
    )
    review_direction_min_papers: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_DIRECTION_MIN_PAPERS", "2"))
    )
    # Second (focused/deep-dive) retrieval round around the selected direction.
    review_focused_query_limit: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_FOCUSED_QUERY_LIMIT", "10"))
    )
    review_focused_max_papers: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_FOCUSED_MAX_PAPERS", "30"))
    )
    # First version supports exactly one broad->focus round; keep fixed to avoid infinite recursion.
    review_max_focus_rounds: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_MAX_FOCUS_ROUNDS", "1"))
    )
    # Minimum core evidence required before research-gap claims may be stated as 'likely' (P5).
    review_sufficiency_min_core: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_SUFFICIENCY_MIN_CORE", "8"))
    )
    institution_access_enabled: bool = Field(
        default_factory=lambda: _env_bool("INSTITUTION_ACCESS_ENABLED", True)
    )
    institution_name: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_NAME", "Tsinghua University")
    )
    institution_gateway_base: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_GATEWAY_BASE", "https://tlink.lib.tsinghua.edu.cn/")
    )
    institution_eproxy_base: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_EPROXY_BASE", "https://eproxy.lib.tsinghua.edu.cn/reader/home")
    )
    institution_openurl_base: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_OPENURL_BASE", "")
    )
    institution_proxy_prefix: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_PROXY_PREFIX", "")
    )
    institution_download_mode: str = Field(
        default_factory=lambda: os.getenv("INSTITUTION_DOWNLOAD_MODE", "handoff")
    )
    wos_api_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "WOS_API_BASE_URL", "https://api.clarivate.com/apis/wos-starter/v1"
        )
    )
    wos_api_key: str = Field(default_factory=lambda: os.getenv("WOS_API_KEY", ""))
    wos_default_db: str = Field(default_factory=lambda: os.getenv("WOS_DEFAULT_DB", "WOS"))
    cnki_search_endpoint: str = Field(default_factory=lambda: os.getenv("CNKI_SEARCH_ENDPOINT", ""))
    cnki_search_method: str = Field(default_factory=lambda: os.getenv("CNKI_SEARCH_METHOD", "GET"))
    cnki_api_key: str = Field(default_factory=lambda: os.getenv("CNKI_API_KEY", ""))
    cnki_auth_header: str = Field(default_factory=lambda: os.getenv("CNKI_AUTH_HEADER", "X-ApiKey"))
    cnki_auth_scheme: str = Field(default_factory=lambda: os.getenv("CNKI_AUTH_SCHEME", ""))
    # Backbone sources (always-on, cross-domain). Crossref is newly added here.
    crossref_enabled: bool = Field(default_factory=lambda: _env_bool("CROSSREF_ENABLED", True))
    crossref_mailto: str = Field(
        default_factory=lambda: os.getenv("CROSSREF_MAILTO", "research-agent@example.com")
    )
    # Domain sources (activated by matched research domain).
    dblp_enabled: bool = Field(default_factory=lambda: _env_bool("DBLP_ENABLED", True))
    europepmc_enabled: bool = Field(default_factory=lambda: _env_bool("EUROPEPMC_ENABLED", True))
    europepmc_email: str = Field(
        default_factory=lambda: os.getenv("EUROPEPMC_EMAIL", "research-agent@example.com")
    )
    pmc_enabled: bool = Field(default_factory=lambda: _env_bool("PMC_ENABLED", True))
    ncbi_email: str = Field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "research-agent@example.com")
    )
    # Optional open-access repository tier (cross-domain full-text; default off).
    core_enabled: bool = Field(default_factory=lambda: _env_bool("CORE_ENABLED", False))
    core_api_key: str = Field(default_factory=lambda: os.getenv("CORE_API_KEY", ""))
    openaire_enabled: bool = Field(default_factory=lambda: _env_bool("OPENAIRE_ENABLED", False))
    openaire_api_key: str = Field(default_factory=lambda: os.getenv("OPENAIRE_API_KEY", ""))
    base_enabled: bool = Field(default_factory=lambda: _env_bool("BASE_ENABLED", False))
    zenodo_enabled: bool = Field(default_factory=lambda: _env_bool("ZENODO_ENABLED", False))
    zenodo_access_token: str = Field(default_factory=lambda: os.getenv("ZENODO_ACCESS_TOKEN", ""))
    hal_enabled: bool = Field(default_factory=lambda: _env_bool("HAL_ENABLED", False))
    # Unpaywall is a DOI to open-access PDF resolver used in the download step, not a search source.
    unpaywall_email: str = Field(default_factory=lambda: os.getenv("UNPAYWALL_EMAIL", ""))
    # Optional JSON override of the domain-to-source routing map.
    scholar_domain_map: str = Field(default_factory=lambda: os.getenv("SCHOLAR_DOMAIN_MAP", ""))
    cloud_sync_enabled: bool = Field(default_factory=lambda: _env_bool("CLOUD_SYNC_ENABLED"))
    cloud_delivery_required: bool = Field(
        default_factory=lambda: _env_bool("CLOUD_DELIVERY_REQUIRED")
    )
    seafile_base_url: str = Field(default_factory=lambda: os.getenv("SEAFILE_BASE_URL", ""))
    seafile_api_token: str = Field(default_factory=lambda: os.getenv("SEAFILE_API_TOKEN", ""))
    seafile_username: str = Field(default_factory=lambda: os.getenv("SEAFILE_USERNAME", ""))
    seafile_password: str = Field(default_factory=lambda: os.getenv("SEAFILE_PASSWORD", ""))
    seafile_repo_id: str = Field(default_factory=lambda: os.getenv("SEAFILE_REPO_ID", ""))
    seafile_repo_name: str = Field(
        default_factory=lambda: os.getenv("SEAFILE_REPO_NAME", "Research Agent")
    )
    seafile_remote_root: str = Field(
        default_factory=lambda: os.getenv("SEAFILE_REMOTE_ROOT", "research-agent")
    )
    seafile_share_links: bool = Field(
        default_factory=lambda: _env_bool("SEAFILE_SHARE_LINKS", True)
    )
    seafile_share_password: str = Field(
        default_factory=lambda: os.getenv("SEAFILE_SHARE_PASSWORD", "")
    )
    seafile_sync_retries: int = Field(
        default_factory=lambda: int(os.getenv("SEAFILE_SYNC_RETRIES", "3"))
    )
    # --- Companion product (SaaS, single-user single-project) ---
    # Root for per-user persistent project workspaces (library, profile, papers).
    companion_workspace_root: str = Field(
        default_factory=lambda: os.getenv(
            "COMPANION_WORKSPACE_ROOT",
            str(_project_root() / "companion-workspace"),
        )
    )
    # Fixed persona identity ("温柔姐姐"). Kept configurable only for copy tuning.
    companion_persona_name: str = Field(
        default_factory=lambda: os.getenv("COMPANION_PERSONA_NAME", "夏琳学姐")
    )
    # Minimal multi-tenant auth. When empty, a permissive dev token is accepted.
    companion_auth_required: bool = Field(
        default_factory=lambda: _env_bool("COMPANION_AUTH_REQUIRED", False)
    )
    # Daily pull-based recommendation size (精选、稀缺: kept small, product cap ≤ 6).
    # This is the *fallback* default; each user can tune it in 个人与设置 within [1, 6].
    companion_daily_feed_size: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_DAILY_FEED_SIZE", "5"))
    )
    # Hard product ceiling for the daily 今日精选 feed (scarcity / precision).
    companion_daily_feed_max: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_DAILY_FEED_MAX", "6"))
    )
    companion_calibration_target: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_CALIBRATION_TARGET", "5"))
    )
    companion_accumulation_target: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_ACCUMULATION_TARGET", "12"))
    )
    # --- Companion direction engine + two-layer feed (reuses /review assets) ---
    # How many candidate research directions to cluster during onboarding探索.
    companion_direction_candidates: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_DIRECTION_CANDIDATES", "5"))
    )
    # Broad discovery recall size when exploring/clustering directions.
    companion_discovery_size: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_DISCOVERY_SIZE", "40"))
    )
    # Page size for the infinite "还有文献" explore stream (下拉续搜 each fetches this many).
    companion_explore_page_size: int = Field(
        default_factory=lambda: int(os.getenv("COMPANION_EXPLORE_PAGE_SIZE", "12"))
    )
    # Looser recall threshold for the explore stream (wider than daily).
    companion_explore_recall_threshold: float = Field(
        default_factory=lambda: float(os.getenv("COMPANION_EXPLORE_RECALL_THRESHOLD", "0.08"))
    )
    # Daily feed screening layer switches (L3 LLM adjudication stays off for cost).
    companion_daily_hard_exclude: bool = Field(
        default_factory=lambda: _env_bool("COMPANION_DAILY_HARD_EXCLUDE", True)
    )
    companion_daily_semantic_rerank: bool = Field(
        default_factory=lambda: _env_bool("COMPANION_DAILY_SEMANTIC_RERANK", True)
    )
    # Run L3 LLM tier adjudication when a paper is ingested into the library.
    companion_ingest_deep_screen: bool = Field(
        default_factory=lambda: _env_bool("COMPANION_INGEST_DEEP_SCREEN", True)
    )
    # MinerU PDF->Markdown service (runs server-side, never on the client).
    mineru_enabled: bool = Field(default_factory=lambda: _env_bool("MINERU_ENABLED", True))
    mineru_base_url: str = Field(
        default_factory=lambda: os.getenv("MINERU_BASE_URL", "http://127.0.0.1:8010")
    )
    mineru_backend: str = Field(default_factory=lambda: os.getenv("MINERU_BACKEND", "pipeline"))
    mineru_lang: str = Field(default_factory=lambda: os.getenv("MINERU_LANG", "ch"))
    mineru_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("MINERU_TIMEOUT_SECONDS", "600"))
    )
    mineru_poll_interval_seconds: float = Field(
        default_factory=lambda: float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "3"))
    )
    # --- Review writer (LaTeX -> PDF pipeline) ---
    # Dedicated model for long-form review prose. Decoupled from the (coding) upstream
    # model because coding models write poor Chinese long-form. Empty -> falls back to
    # upstream_model / auto-resolved chat model.
    review_writer_model: str = Field(
        default_factory=lambda: os.getenv("REVIEW_WRITER_MODEL", "")
    )
    # Default template per language (used when the client does not pass one).
    review_default_template_zh: str = Field(
        default_factory=lambda: os.getenv("REVIEW_DEFAULT_TEMPLATE_ZH", "zh-generic")
    )
    review_default_template_en: str = Field(
        default_factory=lambda: os.getenv("REVIEW_DEFAULT_TEMPLATE_EN", "ieee")
    )
    # Templates registry directory (ships in-repo; overridable for custom templates).
    review_templates_dir: str = Field(
        default_factory=lambda: os.getenv(
            "REVIEW_TEMPLATES_DIR",
            str(Path(__file__).resolve().parent / "writing" / "templates"),
        )
    )
    # Enable the optional concept/framework TikZ schematic figure.
    review_figure_schematic_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_FIGURE_SCHEMATIC_ENABLED", True)
    )
    # Run the submission-tier improvement loop + claim/citation audit gates.
    review_submission_audit: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_SUBMISSION_AUDIT", True)
    )
    # Number of review→fix rounds in the submission improvement loop.
    review_submission_rounds: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_SUBMISSION_ROUNDS", "2"))
    )
    # Max sources synthesised into a review draft (per-section grounding budget).
    review_pipeline_max_sources: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_PIPELINE_MAX_SOURCES", "16"))
    )
    # Theme cluster count for the draft outline (companion keeps this small).
    review_pipeline_theme_count: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_PIPELINE_THEME_COUNT", "4"))
    )
    # LaTeX compile: preferred driver + per-attempt timeout + max auto-fix attempts.
    review_latexmk_enabled: bool = Field(
        default_factory=lambda: _env_bool("REVIEW_LATEXMK_ENABLED", True)
    )
    review_compile_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REVIEW_COMPILE_TIMEOUT_SECONDS", "180"))
    )
    review_compile_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("REVIEW_COMPILE_MAX_ATTEMPTS", "3"))
    )


config = AppConfig()
