"""Interpreter tests with a fully mocked Anthropic client (no real API call).

A fake client returns a canned tool_use block; we assert it parses into a valid
AnalysisPlan and that CT.gov enum values are repaired or rejected.
"""

import types

import pytest

from app.ctgov import introspection as intro_mod
from app.interpreter.llm_interpreter import InterpreterError, interpret_query
from app.interpreter.prompt import PLAN_TOOL_NAME
from app.schemas.plan import AnalysisPlan, Intent
from app.schemas.request import QueryRequest


@pytest.fixture
def mock_whitelist():
    """Load the hardcoded fallback enum whitelist onto the singleton (offline)."""
    intro = intro_mod.get_introspection()
    saved_values, saved_legacy = intro.enum_values, intro.enum_legacy
    intro._load_fallback_enums()
    try:
        yield intro
    finally:
        intro.enum_values, intro.enum_legacy = saved_values, saved_legacy


class _FakeToolUseBlock:
    def __init__(self, tool_input):
        self.type = "tool_use"
        self.name = PLAN_TOOL_NAME
        self.input = tool_input


def _fake_client(tool_input, *, stop_reason="tool_use"):
    """Build a fake AsyncAnthropic whose messages.create returns a canned reply."""
    if stop_reason == "refusal":
        message = types.SimpleNamespace(stop_reason="refusal", content=[])
    else:
        message = types.SimpleNamespace(
            stop_reason=stop_reason, content=[_FakeToolUseBlock(tool_input)]
        )

    async def create(**kwargs):
        # The interpreter must force our tool.
        assert kwargs["tool_choice"] == {"type": "tool", "name": PLAN_TOOL_NAME}
        assert kwargs["tools"][0]["name"] == PLAN_TOOL_NAME
        return message

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


async def test_parses_canned_tool_use_into_plan(mock_whitelist):
    tool_input = {
        "intent": "distribution",
        "entities": {"condition": "melanoma", "phase": ["PHASE2", "PHASE3"]},
        "group_by": "phase",
        "measure": "study_count",
    }
    plan = await interpret_query(
        QueryRequest(query="melanoma trials by phase"),
        client=_fake_client(tool_input),
    )
    assert isinstance(plan, AnalysisPlan)
    assert plan.intent is Intent.distribution
    assert plan.group_by.value == "phase"
    assert plan.entities.condition == "melanoma"
    assert plan.entities.phase == ["PHASE2", "PHASE3"]


async def test_invalid_enum_is_repaired(mock_whitelist):
    # Model emits messy, non-canonical phase labels; interpreter repairs them.
    tool_input = {
        "intent": "distribution",
        "entities": {"phase": ["Phase 2", "phase 3"], "status": ["Recruiting"]},
        "group_by": "phase",
    }
    plan = await interpret_query(
        QueryRequest(query="x"), client=_fake_client(tool_input)
    )
    assert plan.entities.phase == ["PHASE2", "PHASE3"]
    assert plan.entities.status == ["RECRUITING"]


async def test_unmappable_enum_is_rejected(mock_whitelist):
    tool_input = {
        "intent": "distribution",
        "entities": {"phase": ["PHASE9"]},
        "group_by": "phase",
    }
    with pytest.raises(ValueError):
        await interpret_query(QueryRequest(query="x"), client=_fake_client(tool_input))


async def test_refusal_raises(mock_whitelist):
    with pytest.raises(InterpreterError):
        await interpret_query(
            QueryRequest(query="x"),
            client=_fake_client({}, stop_reason="refusal"),
        )
