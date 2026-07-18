"""Per-request orchestration pipeline.

Runs the six stages for the count-based path:

    interpret -> validate -> compile -> fetch+aggregate -> viz shaping -> assemble

Every value in the response comes from the ClinicalTrials.gov API plus this
deterministic Python; the LLM only produces the plan (interpret stage).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.ctgov.client import CTGovClient
from app.ctgov.introspection import get_introspection
from app.engine.aggregator import (
    Aggregation,
    ComparisonAggregation,
    HistogramAggregation,
    ScatterAggregation,
    aggregate,
    aggregate_comparison,
    aggregate_histogram,
    aggregate_scatter,
)
from app.engine.citations import build_index
from app.engine.compiler import compile_plan
from app.engine.network import NetworkResult, build_network
from app.engine.viz_selector import MEASURE_FIELD, MEASURE_UNITS, select_visualization
from app.interpreter.llm_interpreter import interpret_query
from app.interpreter.rule_fallback import interpret_query_rule
from app.schemas.plan import AnalysisPlan, Dimension, Intent
from app.schemas.request import QueryRequest
from app.schemas.response import Meta, VisualizeResponse

logger = logging.getLogger(__name__)

_SOURCE = "ClinicalTrials.gov API v2"

# Map a measure / dimension to the numeric registry dimension used for scatter axes.
_NUMERIC_DIM_FOR_MEASURE = {
    "enrollment_count": "enrollment_count",
    "start_year": "start_year_numeric",
    "study_count": "enrollment_count",  # study_count isn't per-study numeric; fall back
}


def _numeric_dim_for_measure(measure: str) -> str:
    return _NUMERIC_DIM_FOR_MEASURE.get(measure, "enrollment_count")


def _numeric_dim_for_measure_y(dim: Optional[Dimension]) -> str:
    if dim is not None and dim.value == "start_year":
        return "start_year_numeric"
    if dim is not None and dim.value == "enrollment_count":
        return "enrollment_count"
    return "start_year_numeric"


# --- stage 1: interpret ------------------------------------------------------


async def interpret(
    request: QueryRequest,
    *,
    client: Optional[AsyncAnthropic] = None,
) -> AnalysisPlan:
    """Interpret a request into an AnalysisPlan.

    Uses the LLM interpreter when ``ANTHROPIC_API_KEY`` is set; otherwise (or if
    the LLM call raises) uses the deterministic rule-based fallback.
    """
    if get_settings().anthropic_api_key or client is not None:
        try:
            return await interpret_query(request, client=client)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully to rules
            logger.warning("LLM interpretation failed (%s); using rule fallback.", exc)
            return interpret_query_rule(request)

    logger.info("No Anthropic API key configured; using rule-based interpreter.")
    return interpret_query_rule(request)


# --- stage 2: validate -------------------------------------------------------


def _validate_plan(plan: AnalysisPlan, assumptions: list[str]) -> None:
    """Enforce request caps and record assumptions (mutates plan/assumptions)."""
    settings = get_settings()
    if plan.comparison_targets and len(plan.comparison_targets) > settings.max_comparison_targets:
        dropped = len(plan.comparison_targets) - settings.max_comparison_targets
        plan.comparison_targets = plan.comparison_targets[: settings.max_comparison_targets]
        assumptions.append(
            f"Comparison limited to {settings.max_comparison_targets} targets "
            f"({dropped} dropped)."
        )


# --- stage 4: fetch + aggregate ---------------------------------------------


async def _fetch_all(
    client: CTGovClient, params: dict[str, Any], max_pages: int
) -> tuple[list[dict], int]:
    """Fetch studies across pages; return (studies, total_matched)."""
    studies: list[dict] = []
    total = 0
    first = True
    async for page in client.paginate(params, max_pages=max_pages):
        if first:
            total = int(page.get("totalCount") or 0)
            first = False
        studies.extend(page.get("studies") or [])
    return studies, total


# --- stage 6: assemble Meta --------------------------------------------------


def _applied_filters(params: dict[str, Any]) -> dict[str, Any]:
    """The query.*/filter.* portion of a compiled request (drops projection/paging)."""
    return {
        key: value
        for key, value in params.items()
        if key.startswith("query.") or key.startswith("filter.")
    }


def _assemble_meta(
    plan: AnalysisPlan,
    filters_applied: dict[str, Any],
    total_matched: int,
    truncated: bool,
    missing_count: Optional[int],
    assumptions: list[str],
    meta_hints: dict[str, Any],
) -> Meta:
    measure = plan.measure.value
    grouping: dict[str, Any] = {}
    if plan.group_by is not None:
        grouping["group_by"] = plan.group_by.value
    if plan.split_by is not None:
        grouping["split_by"] = plan.split_by.value

    return Meta(
        interpretation=plan.model_dump(mode="json", exclude_none=True),
        filters_applied=filters_applied,
        source=_SOURCE,
        data_timestamp=get_introspection().data_timestamp,
        total_matched=total_matched,
        truncated=truncated,
        missing_count=missing_count,
        measure=measure,
        units=MEASURE_UNITS.get(measure),
        sorting={"by": MEASURE_FIELD.get(measure, "trial_count"), "order": "desc"}
        if plan.group_by is not None
        else None,
        grouping=grouping or None,
        time_granularity=plan.time_granularity.value if plan.time_granularity else None,
        assumptions=assumptions,
        **meta_hints,  # selector hints (e.g. geo=True)
    )


# --- orchestration -----------------------------------------------------------


async def run_visualization(
    request: QueryRequest,
    *,
    anthropic_client: Optional[AsyncAnthropic] = None,
    ctgov_client: Optional[CTGovClient] = None,
) -> VisualizeResponse:
    """Run the full count-based pipeline and return a VisualizeResponse."""
    assumptions: list[str] = []

    # 1. interpret
    plan = await interpret(request, client=anthropic_client)

    # 2. validate
    _validate_plan(plan, assumptions)

    # 3. compile
    intro = get_introspection()
    compiled = compile_plan(plan, introspection=intro)

    # 4. fetch + aggregate
    settings = get_settings()
    client = ctgov_client or CTGovClient()
    own_client = ctgov_client is None
    measure = plan.measure.value

    single_agg: Optional[Aggregation] = None
    comparison_agg: Optional[ComparisonAggregation] = None
    histogram_agg: Optional[HistogramAggregation] = None
    scatter_agg: Optional[ScatterAggregation] = None
    network_result: Optional[NetworkResult] = None
    total_matched = 0
    truncated = False
    missing_count: Optional[int] = None
    skipped_count: Optional[int] = None
    fetched_studies: list[dict] = []

    is_comparison = (
        plan.intent is Intent.comparison
        and bool(plan.comparison_targets)
        and plan.group_by is not None
    )

    try:
        if is_comparison:
            # Fetch each target, then merge series on the shared group_by.
            series_studies: list[tuple[str, list[dict]]] = []
            for target, params in zip(plan.comparison_targets, compiled):
                studies, total = await _fetch_all(client, params, settings.max_pages)
                total_matched += total
                truncated = truncated or len(studies) < total
                series_studies.append((target.label, studies))
            fetched_studies = [s for _, studies in series_studies for s in studies]
            comparison_agg = aggregate_comparison(series_studies, plan.group_by, measure)
            missing_count = comparison_agg.missing_count
            filters_applied = {
                "targets": [
                    {"label": t.label, "filters": _applied_filters(p)}
                    for t, p in zip(plan.comparison_targets, compiled)
                ]
            }
        else:
            params = compiled[0]
            studies, total = await _fetch_all(client, params, settings.max_pages)
            fetched_studies = studies
            total_matched = total
            truncated = len(studies) < total
            filters_applied = _applied_filters(params)

            if plan.intent is Intent.network:
                node_types = (
                    [nt.value for nt in plan.network.node_types]
                    if plan.network
                    else ["drug", "drug"]
                )
                network_result = build_network(
                    studies, node_types, settings.max_network_nodes
                )
                truncated = truncated or network_result.truncated
                skipped_count = network_result.skipped_count
            elif plan.intent is Intent.distribution_continuous:
                strategy = plan.binning.strategy.value if plan.binning else "auto"
                bin_width = plan.binning.bin_width if plan.binning else None
                histogram_agg = aggregate_histogram(
                    studies, "enrollment_count", strategy=strategy, bin_width=bin_width
                )
                missing_count = histogram_agg.missing_count
            elif plan.intent is Intent.correlation:
                scatter_agg = aggregate_scatter(
                    studies,
                    x_dimension=_numeric_dim_for_measure_y(plan.measure_y),
                    y_dimension=_numeric_dim_for_measure(measure),
                    color_dimension="phase",
                )
                missing_count = scatter_agg.dropped_count
            elif plan.group_by is not None:
                single_agg = aggregate(studies, plan.group_by, measure)
                missing_count = single_agg.missing_count
    finally:
        if own_client:
            await client.aclose()

    # record missing-data assumption
    if missing_count:
        if plan.intent is Intent.distribution_continuous:
            assumptions.append(f"{missing_count} studies missing an enrollment value were dropped.")
        elif plan.intent is Intent.correlation:
            assumptions.append(f"{missing_count} studies missing an axis value were dropped.")
        elif plan.group_by is not None:
            assumptions.append(
                f"{missing_count} studies missing '{plan.group_by.value}' were grouped as (unknown)."
            )

    # 5. viz selection + shaping (with citations behind include_citations)
    index = build_index(fetched_studies) if request.include_citations else None
    viz_result = select_visualization(
        plan,
        request.query,
        single=single_agg,
        comparison=comparison_agg,
        histogram=histogram_agg,
        scatter=scatter_agg,
        network=network_result,
        total_matched=total_matched,
        index=index,
        include_citations=request.include_citations,
    )

    # 6. assemble Meta (folding in selector hints + network skipped_count)
    meta_hints = dict(viz_result.meta_hints)
    if skipped_count is not None:
        meta_hints["skipped_count"] = skipped_count
    meta = _assemble_meta(
        plan,
        filters_applied,
        total_matched,
        truncated,
        missing_count,
        assumptions,
        meta_hints,
    )

    return VisualizeResponse(visualization=viz_result.spec, meta=meta)
