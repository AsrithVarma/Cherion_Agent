"""Request schema for the ``POST /visualize`` endpoint.

Structured fields are optional. When supplied they are treated as ground truth
that overrides the LLM's extraction downstream; that override happens in the
pipeline, not here. ``phase`` / ``status`` / ``intervention_type`` are accepted
as a string or a list of strings and normalized against the enum whitelist
later — this schema does not validate their values.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """A natural-language question plus optional structured constraints."""

    query: str = Field(
        ...,
        min_length=1,
        description="The natural-language question about clinical trials (required, non-empty).",
    )

    drug_name: str | None = Field(
        default=None,
        description="Optional drug/intervention name; overrides LLM extraction when set.",
    )
    condition: str | None = Field(
        default=None,
        description="Optional medical condition; overrides LLM extraction when set.",
    )
    sponsor: str | None = Field(
        default=None,
        description="Optional sponsor name; overrides LLM extraction when set.",
    )
    country: str | None = Field(
        default=None,
        description="Optional location/country filter; overrides LLM extraction when set.",
    )

    phase: str | list[str] | None = Field(
        default=None,
        description="Optional trial phase(s); string or list, normalized against the enum whitelist later.",
    )
    status: str | list[str] | None = Field(
        default=None,
        description="Optional overall status(es); string or list, normalized against the enum whitelist later.",
    )
    intervention_type: str | list[str] | None = Field(
        default=None,
        description="Optional intervention type(s); string or list, normalized against the enum whitelist later.",
    )

    start_year: int | None = Field(
        default=None,
        description="Optional lower bound (inclusive) on study start year.",
    )
    end_year: int | None = Field(
        default=None,
        description="Optional upper bound (inclusive) on study start year.",
    )

    compare: list[str] | None = Field(
        default=None,
        description="Two named entities to contrast; when present, forces a comparison intent later.",
    )
    max_results: int | None = Field(
        default=None,
        description="Optional cap on the number of studies fetched/aggregated.",
    )

    include_citations: bool = Field(
        default=True,
        description="Whether to attach source citations (NCT id / field / excerpt) to output values.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        """Reject a query that is empty or only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped
