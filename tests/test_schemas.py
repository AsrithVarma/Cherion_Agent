"""Round-trip and validation tests for the request/plan/response schemas.

Offline and deterministic (no ``live`` marker) — pure Pydantic exercises.
"""

import pytest
from pydantic import ValidationError

from app.schemas.request import QueryRequest
from app.schemas.response import (
    Citation,
    Datum,
    Edge,
    Meta,
    Node,
    VisualizationSpec,
    VisualizeResponse,
)


def _roundtrip(model):
    """Serialize to JSON and parse back; assert the value survives unchanged."""
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model
    return restored


def test_query_request_roundtrips():
    req = QueryRequest(
        query="melanoma trials by phase",
        condition="melanoma",
        phase=["PHASE2", "PHASE3"],
        compare=["Pembrolizumab", "Nivolumab"],
        start_year=2015,
        include_citations=True,
    )
    _roundtrip(req)


def test_bar_chart_response_roundtrips():
    resp = VisualizeResponse(
        visualization=VisualizationSpec(
            type="bar_chart",
            title="Melanoma trials by phase",
            encoding={"x": "phase", "y": "trial_count"},
            data=[
                Datum(
                    phase="PHASE3",
                    trial_count=42,
                    citations=[Citation(nct_id="NCT00000001", field="Phase", excerpt="Phase 3")],
                    citation_count=1,
                ),
                Datum(phase="PHASE2", trial_count=17, citation_count=0),
            ],
        ),
        meta=Meta(
            interpretation={"intent": "distribution", "group_by": "phase"},
            filters_applied={"query.cond": "melanoma"},
            source="ClinicalTrials.gov",
            data_timestamp="2026-07-01",
            total_matched=59,
            truncated=False,
            measure="study_count",
            assumptions=["Missing phase treated as NA"],
        ),
    )
    restored = _roundtrip(resp)
    assert restored.visualization.type == "bar_chart"
    assert restored.visualization.data is not None
    assert restored.visualization.nodes is None


def test_network_graph_response_roundtrips():
    resp = VisualizeResponse(
        visualization=VisualizationSpec(
            type="network_graph",
            title="Co-occurring interventions in melanoma trials",
            encoding={"node": "id", "edge": "weight"},
            nodes=[
                Node(id="Pembrolizumab", type="drug", trial_count=30),
                Node(id="Nivolumab", type="drug", trial_count=25),
            ],
            edges=[Edge(source="Pembrolizumab", target="Nivolumab", weight=12)],
        ),
        meta=Meta(
            interpretation={"intent": "network"},
            filters_applied={"query.cond": "melanoma"},
            source="ClinicalTrials.gov",
            total_matched=200,
            truncated=True,
            measure="study_count",
        ),
    )
    restored = _roundtrip(resp)
    assert restored.visualization.type == "network_graph"
    assert restored.visualization.nodes is not None
    assert restored.visualization.edges is not None
    assert restored.visualization.data is None


@pytest.mark.parametrize("bad_phase", [123, [1, 2], {"phase": "PHASE2"}])
def test_invalid_phase_shape_rejected(bad_phase):
    # phase accepts str | list[str] | None; QueryRequest enforces that shape.
    # (Enum-value normalization is deferred to the pipeline, so bad *values*
    # like "PHASE9" are not rejected here — only bad shapes are.)
    with pytest.raises(ValidationError):
        QueryRequest(query="melanoma", phase=bad_phase)
