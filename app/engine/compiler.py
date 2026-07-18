"""Compiler: turn a validated :class:`AnalysisPlan` into /studies request params.

Pure and deterministic — no network. It reads the cached introspection only to
validate ``AREA[...]`` field names (that cache is populated once at startup).

Returns a *list* of request param dicts. For ``intent == comparison`` it emits
one param-set per ``comparison_target`` (in order); otherwise a single-element
list.

Reconciled against the live API (the project brief was inaccurate on two points):

* ``filter.phase`` is **not** a real parameter — the API returns
  ``400 "filter.phase is unknown parameter"``. Phase filtering is expressed via
  ``filter.advanced`` as ``AREA[Phase](PHASE2 OR PHASE3)``.
* ``AREA[...]`` names are validated against the cached **metadata pieces**
  (where ``StartDate`` / ``Phase`` actually live), unioned with the search-area
  names. The 19 search-area names alone are ``*Search`` routing areas and do
  **not** contain ``StartDate``, yet ``AREA[StartDate]RANGE[...]`` works live.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import get_settings
from app.dimensions.registry import get_dimension
from app.schemas.plan import AnalysisPlan, Intent, Measure, PlanEntities

logger = logging.getLogger(__name__)

# AREA field names (sourced from the registry pieces so they stay consistent).
_START_DATE_AREA = "StartDate"
_PHASE_AREA = "Phase"

# Measure -> registry dimension whose api_field must be projected.
_MEASURE_TO_DIMENSION: dict[str, str] = {
    Measure.enrollment_count.value: "enrollment_count",
    Measure.start_year.value: "start_year",
    # study_count needs no extra field.
}


def compile_plan(
    plan: AnalysisPlan, introspection: Optional[Any] = None
) -> list[dict[str, Any]]:
    """Compile a plan into one or more /studies request param dicts."""
    intro = introspection if introspection is not None else _get_introspection()

    if plan.intent == Intent.comparison and plan.comparison_targets:
        return [
            _compile_one(plan, target.entities, intro)
            for target in plan.comparison_targets
        ]
    return [_compile_one(plan, plan.entities, intro)]


def _compile_one(
    plan: AnalysisPlan, entities: PlanEntities, intro: Any
) -> dict[str, Any]:
    """Build a single request param dict for one entity set."""
    params: dict[str, Any] = {}

    # --- entity mapping: query.* / filter.* -------------------------------
    if entities.condition:
        params["query.cond"] = entities.condition
    if entities.intervention:
        params["query.intr"] = entities.intervention
    if entities.sponsor:
        params["query.spons"] = entities.sponsor
    if entities.country:
        params["query.locn"] = entities.country
    if entities.term:
        params["query.term"] = entities.term
    if entities.status:
        params["filter.overallStatus"] = ",".join(entities.status)

    # --- filter.advanced: phase + start-year range ------------------------
    advanced_clauses: list[str] = []
    phase_clause = _phase_clause(entities.phase, intro)
    if phase_clause:
        advanced_clauses.append(phase_clause)
    year_clause = _year_range_clause(plan, intro)
    if year_clause:
        advanced_clauses.append(year_clause)
    if advanced_clauses:
        params["filter.advanced"] = " AND ".join(advanced_clauses)

    # --- tight fields projection ------------------------------------------
    params["fields"] = ",".join(_projection_fields(plan))

    # --- paging / totals --------------------------------------------------
    settings = get_settings()
    params["pageSize"] = settings.max_page_size
    params["countTotal"] = "true"

    return params


def _phase_clause(phases: Optional[list[str]], intro: Any) -> Optional[str]:
    """Build an ``AREA[Phase](...)`` clause, or None if empty/invalid area."""
    if not phases:
        return None
    if not _area_is_valid(intro, _PHASE_AREA):
        logger.warning("Skipping phase filter: AREA[%s] not recognized.", _PHASE_AREA)
        return None
    values = [p for p in phases if p]
    if not values:
        return None
    if len(values) == 1:
        return f"AREA[{_PHASE_AREA}]{values[0]}"
    return f"AREA[{_PHASE_AREA}]({' OR '.join(values)})"


def _year_range_clause(plan: AnalysisPlan, intro: Any) -> Optional[str]:
    """Build an ``AREA[StartDate]RANGE[start,end]`` clause (MIN/MAX open ends)."""
    year_range = plan.year_range
    if year_range is None:
        return None
    start, end = year_range.start, year_range.end
    if start is None and end is None:
        return None
    if not _area_is_valid(intro, _START_DATE_AREA):
        logger.warning(
            "Skipping year_range filter: AREA[%s] not recognized.", _START_DATE_AREA
        )
        return None
    low = str(start) if start is not None else "MIN"
    high = str(end) if end is not None else "MAX"
    return f"AREA[{_START_DATE_AREA}]RANGE[{low},{high}]"


def _projection_fields(plan: AnalysisPlan) -> list[str]:
    """Tight projection: NCTId,BriefTitle plus group_by/split_by/measure(_y) fields.

    Continuous (histogram) and correlation (scatter) intents also need the
    numeric per-study fields, so EnrollmentCount and StartDate are projected
    alongside Phase for those.
    """
    fields = ["NCTId", "BriefTitle"]

    if plan.intent in (Intent.distribution_continuous, Intent.correlation):
        fields += ["Phase", "EnrollmentCount", "StartDate"]
    if plan.intent is Intent.network:
        fields += ["InterventionType", "InterventionName", "LeadSponsorName", "Condition"]

    dimension_names: list[str] = []
    for dim in (plan.group_by, plan.split_by, plan.measure_y):
        if dim is not None:
            dimension_names.append(dim.value)
    measure_dim = _MEASURE_TO_DIMENSION.get(plan.measure.value)
    if measure_dim:
        dimension_names.append(measure_dim)

    for name in dimension_names:
        dimension = get_dimension(name)
        if dimension is not None:
            fields.append(dimension.api_field)

    return _dedupe(fields)


def _area_is_valid(intro: Any, area_name: str) -> bool:
    """Whether ``area_name`` is a usable AREA[...] field name.

    Validates against metadata pieces unioned with search-area names. If the
    introspection cache is empty (e.g. it failed to load at startup), we cannot
    validate, so we allow the clause rather than silently dropping the filter.
    """
    valid = set(getattr(intro, "metadata_paths", {}).keys()) | set(
        getattr(intro, "area_names", set())
    )
    if not valid:
        return True
    return area_name in valid


def _dedupe(values: list[str]) -> list[str]:
    """Return values with duplicates removed, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _get_introspection() -> Any:
    """Lazy import to avoid a hard import cycle at module load."""
    from app.ctgov.introspection import get_introspection

    return get_introspection()
