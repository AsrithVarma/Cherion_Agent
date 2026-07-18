"""Tests for the dimension registry: defensive extraction + parse_year.

Offline — uses the hand-written tests/fixtures/sparse_studies.json.
"""

import json
from pathlib import Path

import pytest

from app.dimensions.registry import REGISTRY, get_dimension, parse_year

FIXTURES = Path(__file__).parent / "fixtures"


def _load_sparse():
    data = json.loads((FIXTURES / "sparse_studies.json").read_text(encoding="utf-8"))
    return {
        s["protocolSection"]["identificationModule"]["nctId"]: s
        for s in data["studies"]
    }


@pytest.fixture(scope="module")
def sparse():
    return _load_sparse()


def test_no_extractor_raises_on_sparse_records(sparse):
    # Every extractor must survive every (defective) record without raising.
    for study in sparse.values():
        for dim in REGISTRY.values():
            dim.extractor(study)  # must not raise


def test_missing_enrollment_count_returns_none(sparse):
    # NCT90000001 omits enrollmentInfo entirely.
    study = sparse["NCT90000001"]
    assert get_dimension("enrollment_count").extractor(study) is None


def test_empty_interventions_array_returns_none(sparse):
    # NCT90000002 has interventions: [].
    study = sparse["NCT90000002"]
    assert get_dimension("intervention_type").extractor(study) is None


def test_month_only_start_date_parses_year(sparse):
    # NCT90000003 has startDate "January 2024" (no day).
    study = sparse["NCT90000003"]
    assert get_dimension("start_year").extractor(study) == 2024
    assert get_dimension("start_year_numeric").extractor(study) == 2024


def test_extractors_return_none_on_totally_empty_record():
    empty = {"protocolSection": {}}
    for dim in REGISTRY.values():
        assert dim.extractor(empty) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024-01-15", 2024),        # ISO date
        ("January 2024", 2024),      # month + year
        ("2024", 2024),              # year only
        ("January 15, 2024", 2024),  # long form with day
    ],
)
def test_parse_year_handles_four_formats(raw, expected):
    assert parse_year(raw) == expected


@pytest.mark.parametrize("garbage", ["", "N/A", "no year here", None, {}, [], "abcd"])
def test_parse_year_returns_none_on_garbage(garbage):
    assert parse_year(garbage) is None
