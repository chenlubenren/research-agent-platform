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
    upstream_review_model: str = Field(
        default_factory=lambda: os.getenv("UPSTREAM_REVIEW_MODEL", "")
    )
    idea_generator_model: str = Field(default_factory=lambda: os.getenv("IDEA_GENERATOR_MODEL", ""))
    idea_critic_model: str = Field(default_factory=lambda: os.getenv("IDEA_CRITIC_MODEL", ""))
    idea_final_model: str = Field(default_factory=lambda: os.getenv("IDEA_FINAL_MODEL", ""))
    public_api_key: str = Field(default_factory=lambda: os.getenv("PUBLIC_API_KEY", ""))
    image_model: str = Field(default_factory=lambda: os.getenv("IMAGE_MODEL", "gpt-image-2"))
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    )
    workflow_stage_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("WORKFLOW_STAGE_TIMEOUT_SECONDS", "360"))
    )
    image_request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("IMAGE_REQUEST_TIMEOUT_SECONDS", "300"))
    )
    edit_banana_base_url: str = Field(
        default_factory=lambda: os.getenv("EDIT_BANANA_BASE_URL", "").rstrip("/")
    )
    edit_banana_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("EDIT_BANANA_TIMEOUT_SECONDS", "300"))
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
        default_factory=lambda: float(os.getenv("REVIEW_DOWNLOAD_TIMEOUT_SECONDS", "60"))
    )
    review_download_concurrency: int = Field(default_factory=lambda: int(os.getenv("REVIEW_DOWNLOAD_CONCURRENCY", "6")))
    upstream_max_output_tokens: int = Field(default_factory=lambda: int(os.getenv("UPSTREAM_MAX_OUTPUT_TOKENS", "32768")))
    review_screening_enabled: bool = Field(default_factory=lambda: _env_bool("REVIEW_SCREENING_ENABLED", False))
    review_screen_model: str = Field(default_factory=lambda: os.getenv("REVIEW_SCREEN_MODEL", ""))
    review_screen_batch_size: int = Field(default_factory=lambda: int(os.getenv("REVIEW_SCREEN_BATCH_SIZE", "8")))
    review_screen_concurrency: int = Field(default_factory=lambda: int(os.getenv("REVIEW_SCREEN_CONCURRENCY", "4")))
    review_screen_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("REVIEW_SCREEN_TIMEOUT_SECONDS", "180")))
    review_screen_max_candidates: int = Field(default_factory=lambda: int(os.getenv("REVIEW_SCREEN_MAX_CANDIDATES", "50")))
    review_hard_exclusion_terms: str = Field(default_factory=lambda: os.getenv("REVIEW_HARD_EXCLUSION_TERMS", ""))
    review_hard_relevance_floor: float = Field(default_factory=lambda: float(os.getenv("REVIEW_HARD_RELEVANCE_FLOOR", "0")))
    embedding_enabled: bool = Field(default_factory=lambda: _env_bool("EMBEDDING_ENABLED", False))
    embedding_base_url: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", ""))
    embedding_api_key: str = Field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))
    embedding_batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "10")))
    wiki_reference_expansion_enabled: bool = Field(
        default_factory=lambda: _env_bool("WIKI_REFERENCE_EXPANSION_ENABLED", False)
    )
    wiki_reference_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_LIMIT", "50"))
    )
    wiki_reference_source_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_SOURCE_LIMIT", "3"))
    )
    wiki_reference_download_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_DOWNLOAD_LIMIT", "5"))
    )
    wiki_reference_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("WIKI_REFERENCE_TIMEOUT_SECONDS", "20"))
    )
    wiki_reference_max_pdf_mb: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_MAX_PDF_MB", "20"))
    )
    wiki_reference_max_total_mb: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_MAX_TOTAL_MB", "50"))
    )
    wiki_idea_query_paper_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_IDEA_QUERY_PAPER_LIMIT", "12"))
    )
    wiki_idea_query_character_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_IDEA_QUERY_CHARACTER_LIMIT", "12000"))
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
    semantic_scholar_api_key: str = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""))
    crossref_enabled: bool = Field(default_factory=lambda: _env_bool("CROSSREF_ENABLED", True))
    crossref_mailto: str = Field(default_factory=lambda: os.getenv("CROSSREF_MAILTO", "research-agent@example.com"))
    dblp_enabled: bool = Field(default_factory=lambda: _env_bool("DBLP_ENABLED", True))
    europepmc_enabled: bool = Field(default_factory=lambda: _env_bool("EUROPEPMC_ENABLED", True))
    europepmc_email: str = Field(default_factory=lambda: os.getenv("EUROPEPMC_EMAIL", "research-agent@example.com"))
    pmc_enabled: bool = Field(default_factory=lambda: _env_bool("PMC_ENABLED", True))
    ncbi_email: str = Field(default_factory=lambda: os.getenv("NCBI_EMAIL", "research-agent@example.com"))
    core_enabled: bool = Field(default_factory=lambda: _env_bool("CORE_ENABLED", False))
    core_api_key: str = Field(default_factory=lambda: os.getenv("CORE_API_KEY", ""))
    openaire_enabled: bool = Field(default_factory=lambda: _env_bool("OPENAIRE_ENABLED", False))
    openaire_api_key: str = Field(default_factory=lambda: os.getenv("OPENAIRE_API_KEY", ""))
    base_enabled: bool = Field(default_factory=lambda: _env_bool("BASE_ENABLED", False))
    zenodo_enabled: bool = Field(default_factory=lambda: _env_bool("ZENODO_ENABLED", False))
    zenodo_access_token: str = Field(default_factory=lambda: os.getenv("ZENODO_ACCESS_TOKEN", ""))
    hal_enabled: bool = Field(default_factory=lambda: _env_bool("HAL_ENABLED", False))
    unpaywall_email: str = Field(default_factory=lambda: os.getenv("UNPAYWALL_EMAIL", ""))
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


config = AppConfig()
