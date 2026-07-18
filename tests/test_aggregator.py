"""Aggregator tests (offline, injecting saved fixtures).

Ground-truth counts are derived directly from the JSON (reading the raw
``phases`` / ``interventions`` arrays) rather than via the registry, so these
tests exercise the aggregator's bucketing logic against an independent baseline.
"""

import json
from collections import Counter
from pathlib import Path

from app.engine.aggregator import (
    UNKNOWN_KEY,
    aggregate,
    aggregate_comparison,
    aggregate_histogram,
    aggregate_scatter,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(*names):
    studies = []
    for name in names:
        studies += json.loads((FIXTURES / name).read_text(encoding="utf-8"))["studies"]
    return studies


def _buckets_by_key(aggregation):
    return {b.key: b for b in aggregation.buckets}


def test_per_phase_counts_and_nct_ids_preserved():
    studies = _load("pembrolizumab_page1.json", "pembrolizumab_page2.json")

    # Independent ground truth from the raw fixture.
    expected_counts = Counter()
    expected_ncts = {}
    expected_missing = 0
    for study in studies:
        ps = study["protocolSection"]
        nct = ps["identificationModule"]["nctId"]
        phases = ps.get("designModule", {}).get("phases")
        if not phases:
            expected_missing += 1
            continue
        for phase in phases:
            expected_counts[phase] += 1
            expected_ncts.setdefault(phase, []).append(nct)

    result = aggregate(studies, "phase", measure="study_count")
    buckets = _buckets_by_key(result)

    # Counts per phase bucket match.
    for phase, count in expected_counts.items():
        assert buckets[phase].count == count, phase

    # NCT ids are preserved per bucket (same set of studies).
    for phase, ncts in expected_ncts.items():
        assert set(buckets[phase].nct_ids) == set(ncts), phase
        assert len(buckets[phase].nct_ids) == expected_counts[phase], phase

    # The two phase-less records land in (unknown) and count as missing.
    assert result.missing_count == expected_missing
    assert buckets[UNKNOWN_KEY].count == expected_missing
    assert result.total_studies == len(studies)


def test_sparse_fixture_unknown_bucket_and_missing_count_no_raise():
    studies = _load("sparse_studies.json")

    # NCT90000002 has an empty interventions array -> None -> (unknown).
    result = aggregate(studies, "intervention_type", measure="study_count")
    buckets = _buckets_by_key(result)

    assert UNKNOWN_KEY in buckets
    assert result.missing_count > 0
    assert buckets[UNKNOWN_KEY].nct_ids == ["NCT90000002"]
    # Real interventions still bucket correctly.
    assert buckets["DRUG"].nct_ids == ["NCT90000001"]
    assert buckets["BIOLOGICAL"].nct_ids == ["NCT90000003"]


# --- comparison merge -------------------------------------------------------


def _study_with_phases(nct, phases):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "designModule": {"phases": phases},
        }
    }


def test_comparison_merge_two_series_with_correct_counts():
    series_a = [
        _study_with_phases("A1", ["PHASE2"]),
        _study_with_phases("A2", ["PHASE2"]),
        _study_with_phases("A3", ["PHASE1"]),
    ]
    series_b = [_study_with_phases("B1", ["PHASE3"])]

    comp = aggregate_comparison([("A", series_a), ("B", series_b)], "phase")

    assert comp.series_labels == ["A", "B"]
    assert set(comp.keys) == {"PHASE2", "PHASE1", "PHASE3"}
    # Every series has an entry for every key (aligned + zero-filled).
    assert len(comp.buckets) == len(comp.keys) * 2

    def count(series, key):
        return next(sb.count for sb in comp.buckets if sb.series == series and sb.key == key)

    # Series A per-phase counts.
    assert count("A", "PHASE2") == 2
    assert count("A", "PHASE1") == 1
    assert count("A", "PHASE3") == 0  # zero-filled — A has no PHASE3
    # Series B per-phase counts.
    assert count("B", "PHASE3") == 1
    assert count("B", "PHASE2") == 0
    assert count("B", "PHASE1") == 0


# --- histogram binning ------------------------------------------------------


def test_histogram_bins_from_enrollment_fixture():
    studies = _load("enrollment_startdate.json")
    hist = aggregate_histogram(studies, "enrollment_count", strategy="auto")

    counts = {b.bin: b.count for b in hist.bins}
    assert counts == {"0-49": 13, "50-99": 7, "100-249": 3, "250-499": 2, "500+": 0}
    assert hist.missing_count == 0
    assert hist.total_values == 25

    first = next(b for b in hist.bins if b.bin == "0-49")
    assert (first.bin_start, first.bin_end) == (0.0, 50.0)
    last = next(b for b in hist.bins if b.bin == "500+")
    assert last.bin_end is None  # open-ended top bin


# --- scatter ----------------------------------------------------------------


def test_scatter_drops_rows_missing_either_axis():
    # sparse_studies: NCT90000001 has no enrollmentInfo -> missing x -> dropped.
    studies = _load("sparse_studies.json")
    scatter = aggregate_scatter(studies, "enrollment_count", "start_year_numeric")

    assert {p.nct_id for p in scatter.points} == {"NCT90000002", "NCT90000003"}
    assert scatter.dropped_count == 1
    # Both retained points carry finite x and y.
    for point in scatter.points:
        assert point.x is not None and point.y is not None
