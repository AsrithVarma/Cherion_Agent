"""System prompt and Anthropic tool schema for the interpreter.

The LLM's ONLY job is to translate a natural-language query into a validated
:class:`~app.schemas.plan.AnalysisPlan`. It must never produce a number, a
count, or an NCT ID — every value in the final output is computed by
deterministic Python from the ClinicalTrials.gov API. The tool schema below
mirrors ``AnalysisPlan``; its enum lists are derived from the Pydantic enums so
the two never drift.
"""

from __future__ import annotations

from typing import Any

from app.schemas.plan import (
    BinStrategy,
    Dimension,
    Intent,
    Measure,
    NetworkEdgeKind,
    NetworkNodeType,
    TimeGranularity,
)


def _values(enum_cls: Any) -> list[str]:
    return [member.value for member in enum_cls]


SYSTEM_PROMPT = """You are the query interpreter for a clinical-trials \
visualization service. Your SOLE task is to translate the user's \
natural-language question into a structured analysis plan by calling the \
`emit_analysis_plan` tool exactly once.

CRITICAL — anti-hallucination boundary (non-negotiable):
- You NEVER produce data. Do not invent, estimate, or state any count, \
percentage, enrollment number, year value, NCT ID, study title, or any other \
factual value about clinical trials.
- Every real value in the final visualization is computed by deterministic code \
from the live ClinicalTrials.gov API. You only describe HOW to query and shape \
the data — never WHAT the data is.
- If the query asks for a number you happen to "know", ignore that impulse. Your \
output is a plan, not an answer.

How to build the plan:
- Choose the single `intent` that best matches the question (distribution, \
time_trend, comparison, geographic, ranking, network, distribution_continuous, \
correlation, single_value).
- Extract entities (condition, intervention/drug, sponsor, country, free-text \
term) and any phase/status/intervention_type constraints the user mentions.
- Set `group_by` to the dimension the user wants broken down, and `measure` to \
what is being counted or summed (default study_count).
- Only populate `comparison_targets`, `binning`, `ranking`, `network`, \
`year_range`, `time_granularity`, or `measure_y` when the intent calls for them.
- Prefer canonical enum values (e.g. PHASE2, RECRUITING, DRUG) but do not worry \
about exact spelling — a downstream normalizer repairs messy values.
- Leave fields out when the query does not specify them. Do not guess filters \
the user did not ask for."""


_ENTITIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Entities extracted from the query; used to build API filters.",
    "properties": {
        "condition": {"type": "string", "description": "Medical condition of interest."},
        "intervention": {
            "type": "string",
            "description": "Intervention or drug name of interest.",
        },
        "sponsor": {"type": "string", "description": "Sponsor / organization of interest."},
        "phase": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Trial phases (canonical enum values like PHASE2; messy input is repaired downstream).",
        },
        "status": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Overall status values (canonical enum values like RECRUITING).",
        },
        "intervention_type": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Intervention type values (canonical enum values like DRUG).",
        },
        "country": {"type": "string", "description": "Location / country to filter on."},
        "term": {"type": "string", "description": "Free-text term for a broad query.term search."},
    },
    "additionalProperties": False,
}

PLAN_TOOL_NAME = "emit_analysis_plan"

PLAN_TOOL: dict[str, Any] = {
    "name": PLAN_TOOL_NAME,
    "description": (
        "Record the structured analysis plan extracted from the user's query. "
        "This is the ONLY output you produce — it contains no data values, only "
        "how the data should be queried and shaped."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": _values(Intent),
                "description": "The kind of analysis the query maps to.",
            },
            "entities": _ENTITIES_SCHEMA,
            "group_by": {
                "type": "string",
                "enum": _values(Dimension),
                "description": "Primary dimension to group results by.",
            },
            "split_by": {
                "type": "string",
                "enum": _values(Dimension),
                "description": "Secondary dimension to split series by (grouped charts).",
            },
            "measure": {
                "type": "string",
                "enum": _values(Measure),
                "description": "Quantity to compute per group (default study_count).",
            },
            "measure_y": {
                "type": "string",
                "enum": _values(Dimension),
                "description": "Second-axis dimension for scatter / correlation plots.",
            },
            "comparison_targets": {
                "type": "array",
                "description": "Labeled arms to contrast (comparison intent only).",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Human-readable arm label."},
                        "entities": _ENTITIES_SCHEMA,
                    },
                    "required": ["label", "entities"],
                    "additionalProperties": False,
                },
            },
            "binning": {
                "type": "object",
                "description": "Histogram binning config (continuous distributions only).",
                "properties": {
                    "strategy": {"type": "string", "enum": _values(BinStrategy)},
                    "bin_width": {
                        "type": "number",
                        "description": "Bin width when strategy is fixed.",
                    },
                },
                "required": ["strategy"],
                "additionalProperties": False,
            },
            "time_granularity": {
                "type": "string",
                "enum": _values(TimeGranularity),
                "description": "Bucketing granularity for time trends.",
            },
            "year_range": {
                "type": "object",
                "description": "Inclusive year bounds to constrain the query.",
                "properties": {
                    "start": {"type": "integer", "description": "Lower bound year (inclusive)."},
                    "end": {"type": "integer", "description": "Upper bound year (inclusive)."},
                },
                "additionalProperties": False,
            },
            "ranking": {
                "type": "object",
                "description": "Top-N ranking config (ranking intent only).",
                "properties": {
                    "top_n": {"type": "integer", "description": "Number of top groups to return."},
                },
                "required": ["top_n"],
                "additionalProperties": False,
            },
            "network": {
                "type": "object",
                "description": "Network-graph config (network intent only).",
                "properties": {
                    "node_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": _values(NetworkNodeType)},
                        "description": "Entity types that become nodes.",
                    },
                    "edge": {
                        "type": "string",
                        "enum": _values(NetworkEdgeKind),
                        "description": "Edge relationship (co_occurrence).",
                    },
                },
                "required": ["node_types"],
                "additionalProperties": False,
            },
            "viz_hint": {
                "type": "string",
                "description": "Advisory hint about the desired visualization; non-binding.",
            },
        },
        "required": ["intent"],
        "additionalProperties": False,
    },
}
