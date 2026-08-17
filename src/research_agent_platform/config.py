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
        default_factory=lambda: float(os.getenv("REVIEW_DOWNLOAD_TIMEOUT_SECONDS", "60"))
    )
    wiki_reference_expansion_enabled: bool = Field(
        default_factory=lambda: _env_bool("WIKI_REFERENCE_EXPANSION_ENABLED", True)
    )
    wiki_reference_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_LIMIT", "100"))
    )
    wiki_reference_download_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_REFERENCE_DOWNLOAD_LIMIT", "100"))
    )
    wiki_reference_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("WIKI_REFERENCE_TIMEOUT_SECONDS", "20"))
    )
    wiki_idea_query_paper_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_IDEA_QUERY_PAPER_LIMIT", "100"))
    )
    wiki_idea_query_character_limit: int = Field(
        default_factory=lambda: int(os.getenv("WIKI_IDEA_QUERY_CHARACTER_LIMIT", "36000"))
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
