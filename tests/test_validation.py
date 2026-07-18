"""Offline tests for enum normalization.

The cached introspection whitelist is mocked directly on the singleton so these
tests never hit the network.
"""

import pytest

from app.ctgov import introspection as intro_mod
from app.ctgov.introspection import normalize_enum


@pytest.fixture
def mock_whitelist():
    """Populate the introspection singleton with a small fixed whitelist."""
    intro = intro_mod.get_introspection()
    saved_values, saved_legacy = intro.enum_values, intro.enum_legacy
    intro.enum_values = {
        "Phase": {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"},
        "OverallStatus": {
            "RECRUITING",
            "COMPLETED",
            "TERMINATED",
            "WITHDRAWN",
            "ACTIVE_NOT_RECRUITING",
        },
    }
    intro.enum_legacy = {
        "Phase": {"phase 2": "PHASE2", "phase 3": "PHASE3"},
        "OverallStatus": {"active, not recruiting": "ACTIVE_NOT_RECRUITING"},
    }
    try:
        yield intro
    finally:
        intro.enum_values, intro.enum_legacy = saved_values, saved_legacy


@pytest.mark.parametrize(
    "field, raw, expected",
    [
        ("phase", "Phase II", "PHASE2"),
        ("phase", "phase 3", "PHASE3"),
        ("status", "Recruiting", "RECRUITING"),
        ("status", "completed", "COMPLETED"),
        ("phase", "not a real phase", None),
    ],
)
def test_normalize_enum(mock_whitelist, field, raw, expected):
    assert normalize_enum(field, raw) == expected
