"""LLM interpreter: query -> validated AnalysisPlan via forced tool use.

The Anthropic model is called with a single tool (``emit_analysis_plan``) and
``tool_choice`` forcing it, so the response always carries one tool_use block.
Its input is validated into an :class:`AnalysisPlan`; every CT.gov enum value
(phase / status / intervention_type) is repaired through
:func:`~app.ctgov.introspection.normalize_enum` or the call raises.

Structured request fields are ground truth: any optional field supplied in the
:class:`QueryRequest` overrides the LLM's extraction for that slot, and a
non-empty ``compare`` forces ``intent=comparison`` with one comparison target
per named entity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.ctgov.introspection import normalize_enum
from app.interpreter.prompt import PLAN_TOOL, PLAN_TOOL_NAME, SYSTEM_PROMPT
from app.schemas.plan import AnalysisPlan, ComparisonTarget, Intent, PlanEntities, YearRange
from app.schemas.request import QueryRequest

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048

# entities key -> field name understood by normalize_enum
_ENTITY_ENUM_FIELDS = {
    "phase": "phase",
    "status": "overall_status",
    "intervention_type": "intervention_type",
}


class InterpreterError(RuntimeError):
    """Raised when the model fails to produce a usable plan."""


async def interpret_query(
    request: QueryRequest,
    *,
    client: Optional[AsyncAnthropic] = None,
    model: Optional[str] = None,
) -> AnalysisPlan:
    """Interpret a request into an :class:`AnalysisPlan`.

    Args:
        request: The validated API request (query + optional structured fields).
        client: Optional injected Anthropic client (for testing).
        model: Optional model override; defaults to settings.anthropic_model.

    Returns:
        A validated plan with request overrides applied.

    Raises:
        InterpreterError: If the model refuses or emits no plan.
        ValueError / ValidationError: If the plan fails enum/shape validation.
    """
    settings = get_settings()
    anthropic = client or AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    model_id = model or settings.anthropic_model

    message = await anthropic.messages.create(
        model=model_id,
        max_tokens=_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": PLAN_TOOL_NAME},
        messages=[{"role": "user", "content": request.query}],
    )

    if message.stop_reason == "refusal":
        raise InterpreterError("Model refused to interpret the query.")

    tool_input = _extract_tool_input(message)
    if tool_input is None:
        raise InterpreterError("Model did not emit an analysis plan.")

    plan = _build_plan(tool_input)
    _apply_request_overrides(plan, request)
    return plan


def _extract_tool_input(message: Any) -> Optional[dict[str, Any]]:
    """Return the emit_analysis_plan tool input from the response, or None."""
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == PLAN_TOOL_NAME:
            # block.input is already a parsed dict.
            return dict(block.input)
    return None


def _build_plan(raw: dict[str, Any]) -> AnalysisPlan:
    """Normalize CT.gov enum values, then validate into an AnalysisPlan."""
    _normalize_entity_enums(raw.get("entities"))
    for target in raw.get("comparison_targets") or []:
        if isinstance(target, dict):
            _normalize_entity_enums(target.get("entities"))
    # Pydantic validates the structural plan enums (intent, dimensions, measure);
    # a value it cannot coerce raises ValidationError — the "raise" path.
    return AnalysisPlan.model_validate(raw)


def _normalize_entity_enums(entities: Any) -> None:
    """Repair phase/status/intervention_type values in-place, or raise."""
    if not isinstance(entities, dict):
        return
    for key, field in _ENTITY_ENUM_FIELDS.items():
        if entities.get(key) is None:
            continue
        entities[key] = _normalize_values(field, entities[key])


def _normalize_values(field: str, values: Union[str, list]) -> list[str]:
    """Normalize a string or list of enum values to canonical form, or raise."""
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in values:
        canonical = normalize_enum(field, value)
        if canonical is None:
            raise ValueError(f"Unmappable {field} value: {value!r}")
        normalized.append(canonical)
    return normalized


def _apply_request_overrides(plan: AnalysisPlan, request: QueryRequest) -> None:
    """Overlay structured request fields (ground truth) onto the plan."""
    entities = plan.entities
    if request.condition is not None:
        entities.condition = request.condition
    if request.drug_name is not None:
        entities.intervention = request.drug_name
    if request.sponsor is not None:
        entities.sponsor = request.sponsor
    if request.country is not None:
        entities.country = request.country
    if request.phase is not None:
        entities.phase = _normalize_values("phase", request.phase)
    if request.status is not None:
        entities.status = _normalize_values("overall_status", request.status)
    if request.intervention_type is not None:
        entities.intervention_type = _normalize_values(
            "intervention_type", request.intervention_type
        )

    if request.start_year is not None or request.end_year is not None:
        plan.year_range = YearRange(start=request.start_year, end=request.end_year)

    if request.compare:
        # compare present => force a comparison, one target per named entity.
        # Each target keeps the shared plan entities and adds the compared name
        # as a free-text term (compare items are untyped, so term is the safe,
        # always-valid slot; the compiler maps it to query.term).
        plan.intent = Intent.comparison
        plan.comparison_targets = [
            ComparisonTarget(
                label=name,
                entities=entities.model_copy(update={"term": name}),
            )
            for name in request.compare
        ]
