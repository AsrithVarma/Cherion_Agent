"""Application configuration.

Settings are loaded from environment variables (case-insensitive) or a local
``.env`` file via ``pydantic-settings``. Secrets such as the Anthropic API key
must never be committed — keep them in ``.env`` (git-ignored), not in
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Cheiron service.

    Attributes:
        anthropic_api_key: Key for the Anthropic SDK (interpreter stage only).
        ctgov_base_url: Base URL of the ClinicalTrials.gov v2 API.
        max_pages: Hard cap on pages followed during pagination (polite ~1 req/s).
        max_page_size: Cap on ``pageSize``; the API itself allows at most 1000.
        max_comparison_targets: Cap on entities compared in a single request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Secrets / endpoints
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"

    # Fetch / request caps
    max_pages: int = 20
    max_page_size: int = 1000
    max_comparison_targets: int = 4
    max_network_nodes: int = 50


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
