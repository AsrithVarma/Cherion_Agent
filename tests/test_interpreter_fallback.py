"""Tests for the no-LLM rule-based interpreter (offline)."""

import pytest

from app.ctgov import introspection as intro_mod
from app.interpreter.rule_fallback import interpret_query_rule
from app.schemas.plan import Intent
from app.schemas.request import QueryRequest


@pytest.fixture(autouse=True)
def mock_whitelist():
    """Provide an offline enum whitelist so phase/status normalization works."""
    intro = intro_mod.get_introspection()
    saved_values, saved_legacy = intro.enum_values, intro.enum_legacy
    intro._load_fallback_enums()
    try:
        yield
    finally:
        intro.enum_values, intro.enum_legacy = saved_values, saved_legacy


@pytest.mark.parametrize(
    "query, expected_intent",
    [
        ("How have melanoma trials changed over time?", Intent.time_trend),
        ("Show the distribution of trials by phase", Intent.distribution),
        ("Compare pembrolizumab vs nivolumab", Intent.comparison),
        ("Build a co-occurrence network of interventions", Intent.network),
        ("Which countries run the most melanoma trials?", Intent.geographic),
        ("What is the relationship between phase and enrollment?", Intent.correlation),
        ("How large are these trials by enrollment size?", Intent.distribution_continuous),
        ("Top sponsors by number of trials", Intent.ranking),
    ],
)
def test_query_maps_to_expected_intent(query, expected_intent):
    plan = interpret_query_rule(QueryRequest(query=query))
    assert plan.intent is expected_intent


def test_compare_field_forces_comparison_and_builds_targets():
    request = QueryRequest(
        query="trends over time",  # would be time_trend, but compare wins
        compare=["Pembrolizumab", "Nivolumab"],
        condition="melanoma",
    )
    plan = interpret_query_rule(request)

    assert plan.intent is Intent.comparison
    assert plan.comparison_targets is not None
    assert [t.label for t in plan.comparison_targets] == ["Pembrolizumab", "Nivolumab"]
    # Each target keeps the shared condition and carries its own compared term.
    for target, name in zip(plan.comparison_targets, ["Pembrolizumab", "Nivolumab"]):
        assert target.entities.condition == "melanoma"
        assert target.entities.term == name


def test_no_compare_leaves_targets_empty():
    plan = interpret_query_rule(QueryRequest(query="distribution of trials by phase"))
    assert plan.comparison_targets is None
