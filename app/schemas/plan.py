"""Analysis plan schema — the validated intermediate representation.

The LLM (or the rule-based fallback) fills an :class:`AnalysisPlan`; this is the
*only* thing the LLM produces. It never emits counts, NCT ids, or any data
value — those are computed deterministically from the ClinicalTrials.gov API.
This module defines shape only; no logic.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """The kind of analysis the query maps to."""

    distribution = "distribution"
    time_trend = "time_trend"
    comparison = "comparison"
    geographic = "geographic"
    ranking = "ranking"
    network = "network"
    distribution_continuous = "distribution_continuous"
    correlation = "correlation"
    single_value = "single_value"


class Measure(str, Enum):
    """The quantity plotted on the primary (value) axis."""

    study_count = "study_count"
    enrollment_count = "enrollment_count"
    start_year = "start_year"


class Dimension(str, Enum):
    """A groupable/splittable dimension backed by the dimension registry."""

    phase = "phase"
    start_year = "start_year"
    country = "country"
    condition = "condition"
    overall_status = "overall_status"
    study_type = "study_type"
    intervention_type = "intervention_type"
    lead_sponsor_class = "lead_sponsor_class"
    lead_sponsor_name = "lead_sponsor_name"


class TimeGranularity(str, Enum):
    """Granularity for time-trend bucketing."""

    year = "year"
    month = "month"


class BinStrategy(str, Enum):
    """How histogram bins are chosen."""

    auto = "auto"
    fixed = "fixed"


class NetworkNodeType(str, Enum):
    """Entity type a network node represents."""

    drug = "drug"
    sponsor = "sponsor"
    condition = "condition"


class NetworkEdgeKind(str, Enum):
    """The relationship an edge encodes."""

    co_occurrence = "co_occurrence"


class PlanEntities(BaseModel):
    """Entities extracted from the query, used to build API filters."""

    condition: str | None = Field(default=None, description="Medical condition of interest.")
    intervention: str | None = Field(default=None, description="Intervention/drug of interest.")
    sponsor: str | None = Field(default=None, description="Sponsor/organization of interest.")
    phase: list[str] | None = Field(default=None, description="Trial phase values to filter on.")
    status: list[str] | None = Field(default=None, description="Overall status values to filter on.")
    intervention_type: list[str] | None = Field(
        default=None, description="Intervention type values to filter on."
    )
    country: str | None = Field(default=None, description="Location/country to filter on.")
    term: str | None = Field(default=None, description="Free-text term for query.term.")


class ComparisonTarget(BaseModel):
    """One labeled side of a comparison."""

    label: str = Field(..., description="Human-readable label for this comparison arm.")
    entities: PlanEntities = Field(..., description="Entities defining this comparison arm's filters.")


class Binning(BaseModel):
    """Histogram binning configuration (continuous distributions only)."""

    strategy: BinStrategy = Field(..., description="Bin selection strategy: auto or fixed.")
    bin_width: float | None = Field(
        default=None, description="Bin width when strategy is fixed; ignored for auto."
    )


class YearRange(BaseModel):
    """Inclusive year bounds for time filtering."""

    start: int | None = Field(default=None, description="Lower bound year (inclusive).")
    end: int | None = Field(default=None, description="Upper bound year (inclusive).")


class Ranking(BaseModel):
    """Ranking configuration (top-N selection)."""

    top_n: int = Field(..., description="Number of top-ranked groups to return.")


class Network(BaseModel):
    """Network-graph configuration (co-occurrence graphs)."""

    node_types: list[NetworkNodeType] = Field(
        ..., description="Entity types that become nodes (drug, sponsor, condition)."
    )
    edge: NetworkEdgeKind = Field(
        default=NetworkEdgeKind.co_occurrence,
        description="Edge relationship; currently co_occurrence only.",
    )


class AnalysisPlan(BaseModel):
    """The structured plan produced by the interpreter and consumed by the engine."""

    intent: Intent = Field(..., description="The analysis type the query maps to.")
    entities: PlanEntities = Field(
        default_factory=PlanEntities, description="Entities extracted from the query."
    )
    group_by: Dimension | None = Field(
        default=None, description="Primary dimension to group results by."
    )
    split_by: Dimension | None = Field(
        default=None, description="Secondary dimension to split series by (grouped charts)."
    )
    measure: Measure = Field(
        default=Measure.study_count, description="Quantity to compute per group."
    )
    measure_y: Dimension | None = Field(
        default=None, description="Second axis dimension for scatter/correlation plots."
    )
    comparison_targets: list[ComparisonTarget] | None = Field(
        default=None, description="Labeled arms to contrast (comparison intent only)."
    )
    binning: Binning | None = Field(
        default=None, description="Histogram binning config (continuous distributions only)."
    )
    time_granularity: TimeGranularity | None = Field(
        default=None, description="Bucketing granularity for time trends (year or month)."
    )
    year_range: YearRange | None = Field(
        default=None, description="Inclusive year bounds to constrain the query."
    )
    ranking: Ranking | None = Field(
        default=None, description="Top-N ranking config (ranking intent only)."
    )
    network: Network | None = Field(
        default=None, description="Network-graph config (network intent only)."
    )
    viz_hint: str | None = Field(
        default=None, description="Advisory hint about the desired visualization; non-binding."
    )
