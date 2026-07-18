"""Tests for the visualization selector — one per mapping row (offline)."""

from app.engine.aggregator import Aggregation, Bucket
from app.engine.viz_selector import select_visualization
from app.schemas.plan import (
    AnalysisPlan,
    ComparisonTarget,
    Dimension,
    Intent,
    Measure,
    PlanEntities,
    Ranking,
)


def _agg(dimension, pairs):
    buckets = [
        Bucket(key=key, count=count, nct_ids=[f"NCT{i}" for i in range(count)])
        for key, count in pairs
    ]
    return Aggregation(
        dimension=dimension,
        measure="study_count",
        buckets=buckets,
        missing_count=0,
        total_studies=sum(c for _, c in pairs),
    )


def _targets(*labels):
    return [ComparisonTarget(label=lbl, entities=PlanEntities()) for lbl in labels]


def test_distribution_bar_chart():
    plan = AnalysisPlan(intent=Intent.distribution, group_by=Dimension.phase, measure=Measure.study_count)
    spec = select_visualization(plan, "t", single=_agg("phase", [("PHASE2", 25), ("PHASE1", 17)])).spec
    assert spec.type == "bar_chart"
    assert spec.encoding == {"x": "phase", "y": "trial_count"}
    assert spec.data is not None and len(spec.data) == 2
    assert spec.nodes is None and spec.edges is None


def test_distribution_with_split_by_grouped_bar():
    plan = AnalysisPlan(
        intent=Intent.distribution,
        group_by=Dimension.phase,
        split_by=Dimension.overall_status,
        measure=Measure.study_count,
    )
    spec = select_visualization(plan, "t", single=_agg("phase", [("PHASE2", 25)])).spec
    assert spec.type == "grouped_bar_chart"
    assert spec.encoding["x"] == "phase"
    assert spec.encoding["y"] == "trial_count"
    assert spec.encoding["series"] == "series"
    assert spec.encoding["color"] == "series"


def test_comparison_over_phase_grouped_bar():
    plan = AnalysisPlan(
        intent=Intent.comparison,
        group_by=Dimension.phase,
        measure=Measure.study_count,
        comparison_targets=_targets("A", "B"),
    )
    comparison = [("A", _agg("phase", [("PHASE2", 10)])), ("B", _agg("phase", [("PHASE2", 5), ("PHASE3", 2)]))]
    spec = select_visualization(plan, "t", comparison=comparison).spec
    assert spec.type == "grouped_bar_chart"
    assert spec.encoding["series"] == "series"
    # rows carry the series label from each target
    assert {d.series for d in spec.data} == {"A", "B"}


def test_comparison_over_start_year_time_series():
    plan = AnalysisPlan(
        intent=Intent.comparison,
        group_by=Dimension.start_year,
        measure=Measure.study_count,
        comparison_targets=_targets("A", "B"),
    )
    comparison = [("A", _agg("start_year", [("2020", 3)])), ("B", _agg("start_year", [("2020", 4)]))]
    spec = select_visualization(plan, "t", comparison=comparison).spec
    assert spec.type == "time_series"
    assert spec.encoding["x"] == "start_year"
    assert spec.encoding["series"] == "series"


def test_time_trend_time_series():
    plan = AnalysisPlan(intent=Intent.time_trend, group_by=Dimension.start_year, measure=Measure.study_count)
    spec = select_visualization(plan, "t", single=_agg("start_year", [("2020", 5), ("2021", 8)])).spec
    assert spec.type == "time_series"
    assert spec.encoding == {"x": "start_year", "y": "trial_count"}


def test_geographic_bar_chart_with_geo_hint():
    plan = AnalysisPlan(intent=Intent.geographic, group_by=Dimension.country, measure=Measure.study_count)
    result = select_visualization(plan, "t", single=_agg("country", [("United States", 100), ("France", 20)]))
    assert result.spec.type == "bar_chart"
    assert result.spec.encoding == {"x": "country", "y": "trial_count"}
    assert result.meta_hints.get("geo") is True


def test_ranking_bar_chart_top_n_only():
    plan = AnalysisPlan(
        intent=Intent.ranking,
        group_by=Dimension.lead_sponsor_class,
        measure=Measure.study_count,
        ranking=Ranking(top_n=2),
    )
    spec = select_visualization(
        plan, "t",
        single=_agg("lead_sponsor_class", [("INDUSTRY", 50), ("NIH", 30), ("OTHER", 10), ("FED", 5)]),
    ).spec
    assert spec.type == "bar_chart"
    assert len(spec.data) == 2
    assert [d.model_dump()["lead_sponsor_class"] for d in spec.data] == ["INDUSTRY", "NIH"]


def test_distribution_continuous_histogram():
    plan = AnalysisPlan(intent=Intent.distribution_continuous, measure=Measure.enrollment_count)
    spec = select_visualization(plan, "t").spec
    assert spec.type == "histogram"
    # A histogram's y is the per-bin frequency (study count).
    assert spec.encoding == {"x": "bin", "y": "trial_count"}


def test_correlation_scatter_plot():
    plan = AnalysisPlan(
        intent=Intent.correlation,
        measure=Measure.enrollment_count,
        measure_y=Dimension.start_year,
    )
    spec = select_visualization(plan, "t").spec
    assert spec.type == "scatter_plot"
    # Without a ScatterAggregation, axes fall back to measure_y (x) and measure (y).
    assert spec.encoding == {"x": "start_year", "y": "enrollment_count", "color": "color"}


def test_single_value_stat():
    plan = AnalysisPlan(intent=Intent.single_value, measure=Measure.study_count)
    spec = select_visualization(plan, "t", total_matched=1234).spec
    assert spec.type == "stat"
    assert spec.encoding == {"value": "trial_count"}
    assert spec.data[0].model_dump()["trial_count"] == 1234


def test_network_graph_reserves_nodes_edges():
    plan = AnalysisPlan(intent=Intent.network, measure=Measure.study_count)
    spec = select_visualization(plan, "t").spec
    assert spec.type == "network_graph"
    assert spec.encoding == {
        "node": {"id": "id", "group": "type", "size": "trial_count"},
        "edge": {"source": "source", "target": "target", "weight": "weight"},
    }
    assert spec.data is None
    # Without a NetworkResult, nodes/edges are reserved (None).
    assert spec.nodes is None and spec.edges is None
