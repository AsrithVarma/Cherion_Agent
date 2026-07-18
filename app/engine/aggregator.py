"""Aggregator: deterministic grouping of study records into buckets.

Given fetched study records and a dimension, count studies (or sum a numeric
measure) per bucket while preserving the contributing NCT ids for citation. All
values here come from the API records plus this Python — the LLM never touches
them.

Defensive by construction: a study missing the group value lands in the
``(unknown)`` bucket and increments ``missing_count``; multi-valued dimensions
(phase, intervention_type, country) contribute to every value's bucket. Nothing
here raises on messy/absent data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from app.dimensions.registry import Dimension, get_dimension

UNKNOWN_KEY = "(unknown)"

Study = dict[str, Any]


@dataclass
class Bucket:
    """One aggregated group."""

    key: str
    count: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class Aggregation:
    """Result of aggregating studies over one dimension."""

    dimension: str
    measure: str
    buckets: list[Bucket]
    missing_count: int
    total_studies: int


def _dig(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _nct_id(study: Study) -> Optional[str]:
    return _dig(study, "protocolSection", "identificationModule", "nctId")


def _to_keys(value: Any) -> list[str]:
    """Normalize an extractor result into a list of bucket keys (may be empty)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    if value == "":
        return []
    return [str(value)]


def _resolve_dimension(dimension: Union[Dimension, str]) -> Dimension:
    if isinstance(dimension, Dimension):
        return dimension
    resolved = get_dimension(dimension)
    if resolved is None:
        raise ValueError(f"Unknown dimension: {dimension!r}")
    return resolved


def aggregate(
    studies: list[Study],
    dimension: Union[Dimension, str],
    measure: str = "study_count",
) -> Aggregation:
    """Aggregate ``studies`` over ``dimension``.

    Args:
        studies: Fetched study records (each nesting ``protocolSection``).
        dimension: A registry :class:`Dimension` or its name.
        measure: ``"study_count"`` (default) counts studies per bucket;
            ``"enrollment_count"`` sums EnrollmentCount per bucket.

    Returns:
        An :class:`Aggregation`. Buckets are ordered by count descending, then
        key ascending, with ``(unknown)`` always last.
    """
    dim = _resolve_dimension(dimension)

    measure_extractor = None
    if measure == "enrollment_count":
        enrollment_dim = get_dimension("enrollment_count")
        if enrollment_dim is not None:
            measure_extractor = enrollment_dim.extractor

    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    members: dict[str, list[str]] = {}
    missing_count = 0

    for study in studies:
        nct = _nct_id(study)
        keys = _to_keys(dim.extractor(study))
        if not keys:
            missing_count += 1
            keys = [UNKNOWN_KEY]

        measure_value = None
        if measure_extractor is not None:
            measure_value = measure_extractor(study)

        for key in keys:
            counts[key] = counts.get(key, 0) + 1
            members.setdefault(key, [])
            if nct is not None:
                members[key].append(nct)
            if measure == "enrollment_count" and measure_value is not None:
                sums[key] = sums.get(key, 0) + measure_value

    buckets = [
        Bucket(
            key=key,
            count=int(sums.get(key, 0)) if measure == "enrollment_count" else counts[key],
            nct_ids=members.get(key, []),
        )
        for key in counts
    ]
    buckets.sort(key=_bucket_sort_key)

    return Aggregation(
        dimension=dim.name,
        measure=measure,
        buckets=buckets,
        missing_count=missing_count,
        total_studies=len(studies),
    )


def _bucket_sort_key(bucket: Bucket) -> tuple:
    """Sort by count desc, then key asc, with ``(unknown)`` forced last."""
    is_unknown = bucket.key == UNKNOWN_KEY
    return (is_unknown, -bucket.count, bucket.key)


# ============================================================================
# 1. Comparison merge — series-tagged, merged on the shared group_by
# ============================================================================


@dataclass
class SeriesBucket:
    """One (series, group_by key) cell of a comparison, zero-filled if absent."""

    series: str
    key: str
    count: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class ComparisonAggregation:
    """Comparison of several series over a shared group_by dimension."""

    dimension: str
    measure: str
    keys: list[str]  # union of group_by values across series ((unknown) last)
    series_labels: list[str]
    buckets: list[SeriesBucket]  # one per (series, key), aligned/zero-filled
    missing_count: int
    per_series: list[tuple[str, Aggregation]]


def aggregate_comparison(
    series_studies: list[tuple[str, list[Study]]],
    dimension: Union[Dimension, str],
    measure: str = "study_count",
) -> ComparisonAggregation:
    """Aggregate each labeled series and merge them on the shared group_by.

    Every series' studies are aggregated over the same dimension; the result is
    tagged with the series label and aligned to the union of group_by values so
    each series has an entry (zero-filled) for every key.
    """
    dim = _resolve_dimension(dimension)

    per_series: list[tuple[str, Aggregation]] = []
    key_order: list[str] = []
    seen: set[str] = set()
    missing_total = 0

    for label, studies in series_studies:
        agg = aggregate(studies, dim, measure)
        per_series.append((label, agg))
        missing_total += agg.missing_count
        for bucket in agg.buckets:
            if bucket.key not in seen:
                seen.add(bucket.key)
                key_order.append(bucket.key)

    totals = {key: 0 for key in key_order}
    for _, agg in per_series:
        for bucket in agg.buckets:
            totals[bucket.key] += bucket.count
    keys = sorted(key_order, key=lambda k: (k == UNKNOWN_KEY, -totals[k], k))

    buckets: list[SeriesBucket] = []
    for label, agg in per_series:
        by_key = {bucket.key: bucket for bucket in agg.buckets}
        for key in keys:
            found = by_key.get(key)
            buckets.append(
                SeriesBucket(
                    series=label,
                    key=key,
                    count=found.count if found else 0,
                    nct_ids=list(found.nct_ids) if found else [],
                )
            )

    return ComparisonAggregation(
        dimension=dim.name,
        measure=measure,
        keys=keys,
        series_labels=[label for label, _ in per_series],
        buckets=buckets,
        missing_count=missing_total,
        per_series=per_series,
    )


# ============================================================================
# 2. Histogram — bin a continuous numeric measure
# ============================================================================

# Sensible default enrollment bins (half-open [start, end); last is open-ended).
_ENROLLMENT_BINS: list[tuple[float, Optional[float]]] = [
    (0, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, None),
]


@dataclass
class HistogramBin:
    """One histogram bin."""

    bin: str
    bin_start: float
    bin_end: Optional[float]
    count: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class HistogramAggregation:
    """Result of binning a continuous measure across studies."""

    dimension: str
    bins: list[HistogramBin]
    total_values: int  # non-null values that were binned
    missing_count: int  # studies dropped for a missing/None value


def aggregate_histogram(
    studies: list[Study],
    dimension: Union[Dimension, str] = "enrollment_count",
    strategy: str = "auto",
    bin_width: Optional[float] = None,
) -> HistogramAggregation:
    """Bin a continuous numeric measure (e.g. enrollment) across studies.

    Args:
        studies: Fetched study records.
        dimension: A numeric registry dimension (default ``enrollment_count``).
        strategy: ``"auto"`` uses sensible enrollment defaults
            (0-49, 50-99, 100-249, 250-499, 500+); ``"fixed"`` uses uniform
            ``bin_width`` bins from 0 up past the max.
        bin_width: Width for the ``fixed`` strategy.

    Returns:
        A HistogramAggregation; ``None`` values are dropped and counted in
        ``missing_count``.
    """
    dim = _resolve_dimension(dimension)

    values: list[tuple[float, Optional[str]]] = []
    missing = 0
    for study in studies:
        value = dim.extractor(study)
        if value is None:
            missing += 1
            continue
        values.append((float(value), _nct_id(study)))

    edges = _bin_edges(strategy, bin_width, [v for v, _ in values])
    bins = [
        HistogramBin(
            bin=_bin_label(start, end),
            bin_start=float(start),
            bin_end=(float(end) if end is not None else None),
            count=0,
        )
        for start, end in edges
    ]

    binned = 0
    for value, nct in values:
        index = _find_bin(edges, value)
        if index is None:
            continue
        bins[index].count += 1
        if nct is not None:
            bins[index].nct_ids.append(nct)
        binned += 1

    return HistogramAggregation(
        dimension=dim.name, bins=bins, total_values=binned, missing_count=missing
    )


def _bin_edges(
    strategy: str, bin_width: Optional[float], values: list[float]
) -> list[tuple[float, Optional[float]]]:
    if strategy == "fixed" and bin_width and bin_width > 0:
        maximum = max(values) if values else 0.0
        edges: list[tuple[float, Optional[float]]] = []
        start = 0.0
        while start <= maximum:
            edges.append((start, start + bin_width))
            start += bin_width
        if not edges:
            edges.append((0.0, bin_width))
        # Make the final bin open-ended so the max value always lands.
        last_start, _ = edges[-1]
        edges[-1] = (last_start, None)
        return edges
    # auto strategy: sensible enrollment defaults
    return list(_ENROLLMENT_BINS)


def _find_bin(edges: list[tuple[float, Optional[float]]], value: float) -> Optional[int]:
    for index, (start, end) in enumerate(edges):
        if end is None:
            if value >= start:
                return index
        elif start <= value < end:
            return index
    return None


def _bin_label(start: float, end: Optional[float]) -> str:
    if end is None:
        return f"{int(start)}+"
    return f"{int(start)}-{int(end) - 1}"


# ============================================================================
# 3. Scatter — per-study (x, y) points for correlation
# ============================================================================


@dataclass
class ScatterPoint:
    """One study's (x, y) point, optionally colored by a categorical dimension."""

    nct_id: Optional[str]
    x: float
    y: float
    color: Optional[str] = None


@dataclass
class ScatterAggregation:
    """Result of extracting per-study (x, y) points for a correlation plot."""

    x_dimension: str
    y_dimension: str
    points: list[ScatterPoint]
    dropped_count: int  # studies missing either axis


def aggregate_scatter(
    studies: list[Study],
    x_dimension: Union[Dimension, str],
    y_dimension: Union[Dimension, str],
    color_dimension: Optional[Union[Dimension, str]] = "phase",
) -> ScatterAggregation:
    """Extract one (x, y) point per study; drop rows missing either axis.

    Args:
        studies: Fetched study records.
        x_dimension: Numeric registry dimension for the x axis.
        y_dimension: Numeric registry dimension for the y axis.
        color_dimension: Optional categorical dimension for point color
            (default ``phase``; multi-valued dimensions use the first value).
    """
    x_dim = _resolve_dimension(x_dimension)
    y_dim = _resolve_dimension(y_dimension)
    color_dim = _resolve_dimension(color_dimension) if color_dimension else None

    points: list[ScatterPoint] = []
    dropped = 0
    for study in studies:
        x_value = x_dim.extractor(study)
        y_value = y_dim.extractor(study)
        if x_value is None or y_value is None:
            dropped += 1
            continue

        color: Optional[str] = None
        if color_dim is not None:
            raw = color_dim.extractor(study)
            if isinstance(raw, list):
                color = raw[0] if raw else None
            else:
                color = raw

        points.append(
            ScatterPoint(nct_id=_nct_id(study), x=float(x_value), y=float(y_value), color=color)
        )

    return ScatterAggregation(
        x_dimension=x_dim.name,
        y_dimension=y_dim.name,
        points=points,
        dropped_count=dropped,
    )
