"""Response schema — the output contract of ``POST /visualize``.

Every value in a response is computed deterministically from ClinicalTrials.gov
data; the schema here only describes the shape. Chart-type visualizations carry
``data``; ``network_graph`` carries ``nodes`` + ``edges``.

Supported ``VisualizationSpec.type`` values: ``bar_chart``, ``grouped_bar_chart``,
``time_series``, ``histogram``, ``scatter_plot``, ``network_graph``, ``stat``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A pointer back to the source study that produced a value."""

    nct_id: str = Field(..., description="NCT identifier of the source study.")
    field: str = Field(..., description="Source field/piece the value was drawn from.")
    excerpt: str = Field(..., description="Short verbatim excerpt supporting the value.")


class Datum(BaseModel):
    """A flexible chart row.

    Common chart fields are declared for convenience; arbitrary additional keys
    are allowed so datasets can carry whatever the chosen chart type needs.
    """

    model_config = ConfigDict(extra="allow")

    phase: str | None = Field(default=None, description="Phase label for phase-grouped rows.")
    trial_count: int | None = Field(default=None, description="Number of studies in this row.")
    bin: str | None = Field(default=None, description="Histogram bin label.")
    bin_start: float | None = Field(default=None, description="Histogram bin lower edge.")
    bin_end: float | None = Field(default=None, description="Histogram bin upper edge.")
    series: str | None = Field(default=None, description="Series name for grouped/split charts.")
    start_year: int | str | None = Field(
        default=None,
        description="Study start year (int) or a categorical bucket label like '(unknown)'.",
    )
    enrollment_count: int | None = Field(default=None, description="Enrollment value for this row.")
    nct_id: str | None = Field(default=None, description="NCT id when a row maps to a single study.")

    citations: list[Citation] = Field(
        default_factory=list, description="Citations backing this row's value(s)."
    )
    citation_count: int = Field(default=0, description="Number of studies backing this row.")


class Node(BaseModel):
    """A node in a network graph."""

    id: str = Field(..., description="Unique node identifier.")
    type: str = Field(..., description="Node entity type (e.g. drug, sponsor, condition).")
    trial_count: int = Field(..., description="Number of studies associated with this node.")
    citations: list[Citation] | None = Field(
        default=None, description="Citations backing this node."
    )
    citation_count: int | None = Field(
        default=None, description="Number of studies backing this node."
    )


class Edge(BaseModel):
    """An edge in a network graph."""

    source: str = Field(..., description="Source node id.")
    target: str = Field(..., description="Target node id.")
    weight: int = Field(..., description="Edge weight (e.g. co-occurrence count).")
    citations: list[Citation] | None = Field(
        default=None, description="Citations backing this edge."
    )
    citation_count: int | None = Field(
        default=None, description="Number of studies backing this edge."
    )


class VisualizationSpec(BaseModel):
    """The visualization specification. Charts use ``data``; networks use ``nodes``/``edges``."""

    type: str = Field(
        ...,
        description=(
            "One of: bar_chart, grouped_bar_chart, time_series, histogram, "
            "scatter_plot, network_graph, stat."
        ),
    )
    title: str = Field(..., description="Human-readable chart title.")
    encoding: dict[str, Any] = Field(
        ..., description="Axis/field encoding mapping for the chosen visualization type."
    )
    data: list[Datum] | None = Field(
        default=None, description="Row data for chart types; null for network graphs."
    )
    nodes: list[Node] | None = Field(
        default=None, description="Nodes for network_graph; null for chart types."
    )
    edges: list[Edge] | None = Field(
        default=None, description="Edges for network_graph; null for chart types."
    )


class Meta(BaseModel):
    """Provenance and interpretation metadata accompanying a visualization.

    Extra keys are allowed so the visualization selector can attach hints
    (e.g. ``geo: true`` for geographic charts).
    """

    model_config = ConfigDict(extra="allow")

    interpretation: dict[str, Any] = Field(
        ..., description="How the query was interpreted (plan summary)."
    )
    filters_applied: dict[str, Any] = Field(
        ..., description="The concrete API filters that were applied."
    )
    source: str = Field(..., description="Data source attribution string.")
    data_timestamp: str | None = Field(
        default=None, description="ClinicalTrials.gov data timestamp, if known."
    )
    total_matched: int = Field(..., description="Total studies matching the query on the API.")
    truncated: bool = Field(..., description="Whether results were truncated by paging/result caps.")
    missing_count: int | None = Field(
        default=None, description="Number of matched studies missing the measured field."
    )
    measure: str = Field(..., description="The measure computed (e.g. study_count).")
    units: str | None = Field(default=None, description="Units of the measure, if applicable.")
    sorting: dict[str, Any] | None = Field(
        default=None, description="Sorting applied to the output, if any."
    )
    grouping: dict[str, Any] | None = Field(
        default=None, description="Grouping applied to the output, if any."
    )
    time_granularity: str | None = Field(
        default=None, description="Time bucketing granularity, if applicable."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made during interpretation/aggregation."
    )
    notes: str | None = Field(default=None, description="Free-form notes about the result.")


class VisualizeResponse(BaseModel):
    """The full response returned by ``POST /visualize``."""

    visualization: VisualizationSpec = Field(..., description="The computed visualization spec.")
    meta: Meta = Field(..., description="Provenance and interpretation metadata.")
