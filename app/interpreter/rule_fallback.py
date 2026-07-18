"""Rule-based interpreter — a no-LLM fallback that maps keywords to a plan.

Used when no Anthropic API key is configured (or as a graceful degrade if the
LLM call fails). It is deterministic and needs no network: intent comes from
keyword matching, entities come from the request's structured fields, and a
year range is pulled from the query text via regex.

Enum values (phase/status/intervention_type) are normalized through
:func:`~app.ctgov.introspection.normalize_enum`; unmappable values are dropped
(the fallback favors resilience over strictness).
"""

from __future__ import annotations

import re
from typing import Optional, Union

from app.ctgov.introspection import normalize_enum
from app.schemas.plan import (
    AnalysisPlan,
    BinStrategy,
    Binning,
    ComparisonTarget,
    Dimension,
    Intent,
    Measure,
    PlanEntities,
    Ranking,
    TimeGranularity,
    YearRange,
)
from app.schemas.request import QueryRequest

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Intent keyword rules in precedence order (first match wins). Correlation is
# checked before comparison so "vs enrollment" maps to correlation, not compare.
_INTENT_RULES: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.correlation, ("correlation", "relationship between", "vs enrollment", "versus enrollment")),
    (Intent.comparison, ("compare", " vs ", " vs.", "versus")),
    (Intent.network, ("network", "co-occur", "cooccur", "co occur")),
    (Intent.time_trend, ("over time", "per year", "trend")),
    (Intent.geographic, ("countries", "country", "where")),
    (Intent.distribution_continuous, ("enrollment size", "how large", "how big")),
    (Intent.ranking, ("top ", "most ", "ranking", "rank ")),
    (Intent.distribution, ("distribution", "across", "by phase", "breakdown", "by status")),
]

# Group-by dimension keyword hints.
_DIMENSION_HINTS: list[tuple[Dimension, tuple[str, ...]]] = [
    (Dimension.phase, ("phase",)),
    (Dimension.overall_status, ("status", "recruiting")),
    (Dimension.country, ("country", "countries", "where")),
    (Dimension.lead_sponsor_class, ("sponsor", "funder")),
    (Dimension.study_type, ("study type", "observational", "interventional")),
    (Dimension.intervention_type, ("intervention type",)),
    (Dimension.start_year, ("year", "over time", "per year")),
]


def interpret_query_rule(request: QueryRequest) -> AnalysisPlan:
    """Build an :class:`AnalysisPlan` from a request with no LLM call."""
    query = request.query.lower()

    intent = Intent.comparison if request.compare else _detect_intent(query)
    entities = _entities_from_request(request)
    group_by = _group_by_for(intent, query)

    plan = AnalysisPlan(
        intent=intent,
        entities=entities,
        group_by=group_by,
        measure=Measure.study_count,
    )

    if intent is Intent.time_trend:
        plan.time_granularity = TimeGranularity.year
    if intent is Intent.ranking:
        plan.ranking = Ranking(top_n=10)
    if intent is Intent.distribution_continuous:
        plan.binning = Binning(strategy=BinStrategy.auto)

    year_range = _year_range(request, query)
    if year_range is not None:
        plan.year_range = year_range

    if request.compare:
        plan.comparison_targets = [
            ComparisonTarget(label=name, entities=entities.model_copy(update={"term": name}))
            for name in request.compare
        ]

    return plan


def _detect_intent(query: str) -> Intent:
    for intent, keywords in _INTENT_RULES:
        if any(keyword in query for keyword in keywords):
            return intent
    return Intent.distribution


def _detect_dimension(query: str) -> Optional[Dimension]:
    for dimension, keywords in _DIMENSION_HINTS:
        if any(keyword in query for keyword in keywords):
            return dimension
    return None


def _group_by_for(intent: Intent, query: str) -> Optional[Dimension]:
    detected = _detect_dimension(query)
    if intent is Intent.time_trend:
        return Dimension.start_year
    if intent is Intent.geographic:
        return Dimension.country
    if intent is Intent.distribution:
        return detected or Dimension.phase
    if intent is Intent.ranking:
        return detected or Dimension.lead_sponsor_class
    if intent is Intent.comparison:
        return detected or Dimension.phase
    return detected


def _entities_from_request(request: QueryRequest) -> PlanEntities:
    return PlanEntities(
        condition=request.condition,
        intervention=request.drug_name,
        sponsor=request.sponsor,
        country=request.country,
        phase=_normalize_values("phase", request.phase),
        status=_normalize_values("overall_status", request.status),
        intervention_type=_normalize_values("intervention_type", request.intervention_type),
    )


def _normalize_values(
    field: str, values: Union[str, list, None]
) -> Optional[list[str]]:
    """Normalize to canonical enum values, dropping unmappable ones; None if empty."""
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    normalized = [c for c in (normalize_enum(field, v) for v in values) if c is not None]
    return normalized or None


def _year_range(request: QueryRequest, query: str) -> Optional[YearRange]:
    """Request years are ground truth; otherwise extract a range from the query."""
    if request.start_year is not None or request.end_year is not None:
        return YearRange(start=request.start_year, end=request.end_year)

    years = [int(y) for y in _YEAR_RE.findall(query)]
    if not years:
        return None
    if len(years) >= 2:
        return YearRange(start=min(years), end=max(years))

    year = years[0]
    if any(k in query for k in ("before", "until", "prior to", "up to", "by ")):
        return YearRange(start=None, end=year)
    return YearRange(start=year, end=None)
