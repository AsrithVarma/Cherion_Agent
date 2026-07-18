"""Visualization selector.

The SELECTOR — not the LLM — decides the final ``VisualizationSpec.type`` and
shapes the aggregator's intermediate structure into chart data rows (or
``nodes``/``edges`` for network graphs), with the right encoding channels. When
an ``index`` and ``include_citations`` are supplied it also attaches up to five
``{nct_id, field, excerpt}`` citations per artifact (``citation_count`` always
carries the true total).

Type mapping:
    distribution (categorical group_by, study_count)      -> bar_chart
    distribution + split_by                               -> grouped_bar_chart
    comparison (targets over a shared group_by)           -> grouped_bar_chart
                                                             (time_series if the
                                                              shared group_by is
                                                              start_year)
    time_trend (study_count over start_year)              -> time_series
    geographic / country group_by                         -> bar_chart (+ geo hint)
    ranking (study_count, top_n)                          -> bar_chart, top N only
    distribution_continuous (one continuous measure)      -> histogram
    correlation (two continuous measures)                 -> scatter_plot
    single_value                                          -> stat
    network                                               -> network_graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.dimensions.registry import get_dimension
from app.engine import citations as cite
from app.engine.aggregator import (
    Aggregation,
    ComparisonAggregation,
    HistogramAggregation,
    ScatterAggregation,
)
from app.engine.network import NetworkResult
from app.schemas.plan import AnalysisPlan, Intent
from app.schemas.response import Datum, Edge, Node, VisualizationSpec

# measure -> (chart field name, human units)
MEASURE_FIELD = {
    "study_count": "trial_count",
    "enrollment_count": "enrollment_count",
    "start_year": "start_year",
}
MEASURE_UNITS = {
    "study_count": "studies",
    "enrollment_count": "participants",
    "start_year": None,
}

# Network node type -> the CT.gov field a node's citation excerpt is drawn from.
_NODE_CITATION_FIELD = {
    "sponsor": "LeadSponsorName",
    "drug": "InterventionName",
    "condition": "Condition",
}


@dataclass
class VizResult:
    """A selected visualization plus any hints to fold into Meta."""

    spec: VisualizationSpec
    meta_hints: dict[str, Any] = field(default_factory=dict)


def select_visualization(
    plan: AnalysisPlan,
    title: str,
    *,
    single: Optional[Aggregation] = None,
    comparison: Optional[object] = None,
    histogram: Optional[HistogramAggregation] = None,
    scatter: Optional[ScatterAggregation] = None,
    network: Optional[NetworkResult] = None,
    total_matched: int = 0,
    index: Optional[dict[str, Any]] = None,
    include_citations: bool = False,
) -> VizResult:
    """Choose the chart type and shape the aggregation into a VisualizationSpec."""
    viz_type = _select_type(plan, comparison)
    measure_field = MEASURE_FIELD.get(plan.measure.value, "trial_count")
    cites = index if include_citations else None
    hints: dict[str, Any] = {}

    if viz_type == "stat":
        return VizResult(
            VisualizationSpec(
                type="stat",
                title=title,
                encoding={"value": measure_field},
                data=[Datum(**{measure_field: total_matched})],
            )
        )

    if viz_type == "network_graph":
        return VizResult(_network_spec(title, network, cites))

    field_name = plan.group_by.value if plan.group_by is not None else None

    if viz_type == "grouped_bar_chart":
        encoding = {"x": field_name, "y": measure_field, "series": "series", "color": "series"}
        data = _comparison_rows(comparison, plan, measure_field, cites) if comparison else (
            _bucket_rows(single, plan, measure_field, cites) if single else []
        )
        return VizResult(VisualizationSpec(type=viz_type, title=title, encoding=encoding, data=data))

    if viz_type == "time_series":
        encoding: dict[str, Any] = {"x": field_name, "y": measure_field}
        if comparison:
            encoding["series"] = "series"
            encoding["color"] = "series"
            data = _comparison_rows(comparison, plan, measure_field, cites)
        else:
            data = _bucket_rows(single, plan, measure_field, cites) if single else []
        return VizResult(VisualizationSpec(type=viz_type, title=title, encoding=encoding, data=data))

    if viz_type == "bar_chart":
        buckets = single.buckets if single else []
        if plan.intent is Intent.ranking and plan.ranking is not None:
            buckets = buckets[: plan.ranking.top_n]
        data = _rows_from_buckets(buckets, plan, measure_field, cites)
        if field_name == "country" or plan.intent is Intent.geographic:
            hints["geo"] = True
        return VizResult(
            VisualizationSpec(
                type="bar_chart",
                title=title,
                encoding={"x": field_name, "y": measure_field},
                data=data,
            ),
            hints,
        )

    if viz_type == "histogram":
        data = _histogram_rows(histogram, cites) if histogram is not None else []
        return VizResult(
            VisualizationSpec(
                type="histogram",
                title=title,
                encoding={"x": "bin", "y": "trial_count"},
                data=data,
            )
        )

    if viz_type == "scatter_plot":
        if scatter is not None:
            encoding = {"x": scatter.x_dimension, "y": scatter.y_dimension, "color": "color"}
            data = _scatter_rows(scatter, cites)
        else:
            x_channel = plan.measure_y.value if plan.measure_y is not None else "x"
            encoding = {"x": x_channel, "y": measure_field, "color": "color"}
            data = []
        return VizResult(
            VisualizationSpec(type="scatter_plot", title=title, encoding=encoding, data=data)
        )

    # Defensive default.
    return VizResult(
        VisualizationSpec(
            type="bar_chart",
            title=title,
            encoding={"x": field_name, "y": measure_field},
            data=_bucket_rows(single, plan, measure_field, cites) if single else [],
        )
    )


def _select_type(plan: AnalysisPlan, comparison: Optional[object]) -> str:
    """Deterministically map the plan (and available aggregations) to a chart type."""
    intent = plan.intent

    if intent is Intent.single_value:
        return "stat"
    if intent is Intent.network:
        return "network_graph"
    if intent is Intent.correlation:
        return "scatter_plot"
    if intent is Intent.distribution_continuous:
        return "histogram"

    if comparison is not None:
        if plan.group_by is not None and plan.group_by.value == "start_year":
            return "time_series"
        return "grouped_bar_chart"

    if intent is Intent.time_trend:
        return "time_series"
    if intent is Intent.geographic:
        return "bar_chart"
    if intent is Intent.ranking:
        return "bar_chart"
    if intent is Intent.distribution:
        return "grouped_bar_chart" if plan.split_by is not None else "bar_chart"

    return "bar_chart" if plan.group_by is not None else "stat"


def _bucket_rows(
    agg: Aggregation,
    plan: AnalysisPlan,
    measure_field: str,
    index: Optional[dict[str, Any]],
    series: Optional[str] = None,
) -> list[Datum]:
    return _rows_from_buckets(agg.buckets, plan, measure_field, index, series)


def _rows_from_buckets(
    buckets: list,
    plan: AnalysisPlan,
    measure_field: str,
    index: Optional[dict[str, Any]],
    series: Optional[str] = None,
) -> list[Datum]:
    dimension = get_dimension(plan.group_by.value)
    field_name = plan.group_by.value
    rows: list[Datum] = []
    for bucket in buckets:
        # Numeric-looking keys (e.g. years) stay ints; labels like "(unknown)" stay strings.
        key: Any = int(bucket.key) if bucket.key.isdigit() else bucket.key
        payload: dict[str, Any] = {
            field_name: key,
            measure_field: bucket.count,
            "citation_count": len(bucket.nct_ids),
        }
        if series is not None:
            payload["series"] = series
        if index is not None and dimension is not None:
            payload["citations"] = cite.dimension_citations(
                bucket.nct_ids, index, dimension, match_key=bucket.key
            )
        rows.append(Datum(**payload))
    return rows


def _comparison_rows(
    comparison: object, plan: AnalysisPlan, measure_field: str, index: Optional[dict[str, Any]]
) -> list[Datum]:
    if isinstance(comparison, ComparisonAggregation):
        dimension = get_dimension(comparison.dimension)
        field_name = comparison.dimension
        rows: list[Datum] = []
        for sb in comparison.buckets:
            payload: dict[str, Any] = {
                field_name: (int(sb.key) if sb.key.isdigit() else sb.key),
                measure_field: sb.count,
                "series": sb.series,
                "citation_count": len(sb.nct_ids),
            }
            if index is not None and dimension is not None:
                payload["citations"] = cite.dimension_citations(
                    sb.nct_ids, index, dimension, match_key=sb.key
                )
            rows.append(Datum(**payload))
        return rows
    # Legacy list[(label, Aggregation)].
    rows = []
    for label, agg in comparison:  # type: ignore[misc]
        rows.extend(_bucket_rows(agg, plan, measure_field, index, series=label))
    return rows


def _histogram_rows(
    histogram: HistogramAggregation, index: Optional[dict[str, Any]]
) -> list[Datum]:
    enrollment_dim = get_dimension("enrollment_count")
    rows: list[Datum] = []
    for b in histogram.bins:
        payload: dict[str, Any] = {
            "bin": b.bin,
            "bin_start": b.bin_start,
            "bin_end": b.bin_end,
            "trial_count": b.count,
            "citation_count": len(b.nct_ids),
        }
        if index is not None and enrollment_dim is not None:
            payload["citations"] = cite.dimension_citations(b.nct_ids, index, enrollment_dim)
        rows.append(Datum(**payload))
    return rows


def _scatter_rows(
    scatter: ScatterAggregation, index: Optional[dict[str, Any]]
) -> list[Datum]:
    rows: list[Datum] = []
    for p in scatter.points:
        payload: dict[str, Any] = {
            "nct_id": p.nct_id,
            scatter.x_dimension: p.x,
            scatter.y_dimension: p.y,
            "color": p.color,
            "citation_count": 1,  # each point's single citation is its own trial
        }
        if index is not None and p.nct_id is not None:
            payload["citations"] = cite.title_citations([p.nct_id], index, limit=1)
        rows.append(Datum(**payload))
    return rows


def _network_spec(
    title: str, network: Optional[NetworkResult], index: Optional[dict[str, Any]]
) -> VisualizationSpec:
    encoding = {
        "node": {"id": "id", "group": "type", "size": "trial_count"},
        "edge": {"source": "source", "target": "target", "weight": "weight"},
    }
    nodes: Optional[list[Node]] = None
    edges: Optional[list[Edge]] = None
    if network is not None:
        nodes = []
        for n in network.nodes:
            node_citations = None
            if index is not None:
                field = _NODE_CITATION_FIELD.get(n.type, "InterventionName")
                node_citations = cite.value_citations(n.nct_ids, index, field, n.id)
            nodes.append(
                Node(
                    id=n.id,
                    type=n.type,
                    trial_count=n.trial_count,
                    citations=node_citations,
                    citation_count=len(n.nct_ids),
                )
            )
        edges = []
        for e in network.edges:
            edge_citations = cite.title_citations(e.nct_ids, index) if index is not None else None
            edges.append(
                Edge(
                    source=e.source,
                    target=e.target,
                    weight=e.weight,
                    citations=edge_citations,
                    citation_count=len(e.nct_ids),
                )
            )
    return VisualizationSpec(
        type="network_graph", title=title, encoding=encoding, data=None, nodes=nodes, edges=edges
    )
