"""Citation wiring tests (offline, via the pipeline with mocked clients).

Confirms bar/bin datums carry real-NCT citations, citation_count equals the
bucket/bin count, and include_citations=false omits the excerpt lists.
"""

import json
import types
from pathlib import Path

import pytest

from app.ctgov import introspection as intro_mod
from app.interpreter.prompt import PLAN_TOOL_NAME
from app.schemas.request import QueryRequest
from app.services.pipeline import run_visualization

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def mock_whitelist():
    intro = intro_mod.get_introspection()
    saved = (intro.enum_values, intro.enum_legacy)
    intro._load_fallback_enums()
    try:
        yield
    finally:
        intro.enum_values, intro.enum_legacy = saved


def _page(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _nct_ids(*names):
    ids = set()
    for name in names:
        for s in _page(name)["studies"]:
            ids.add(s["protocolSection"]["identificationModule"]["nctId"])
    return ids


def _fake_anthropic(plan_input):
    block = types.SimpleNamespace(type="tool_use", name=PLAN_TOOL_NAME, input=plan_input)
    message = types.SimpleNamespace(stop_reason="tool_use", content=[block])

    async def create(**kwargs):
        return message

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


class _FakeCTGov:
    def __init__(self, pages):
        self._pages = pages

    async def paginate(self, params, max_pages=None):
        for page in self._pages:
            yield page

    async def aclose(self):
        pass


async def _run(plan_input, page_names, include=True):
    return await run_visualization(
        QueryRequest(query="q", include_citations=include),
        anthropic_client=_fake_anthropic(plan_input),
        ctgov_client=_FakeCTGov([_page(n) for n in page_names]),
    )


async def test_bar_datums_carry_real_nct_citations():
    pages = ["pembrolizumab_page1.json", "pembrolizumab_page2.json"]
    valid_ncts = _nct_ids(*pages)
    resp = await _run(
        {"intent": "distribution", "entities": {"condition": "melanoma"},
         "group_by": "phase", "measure": "study_count"},
        pages,
    )
    data = resp.visualization.data
    assert data, "expected phase buckets"
    real_buckets = [d for d in data if d.phase != "(unknown)"]
    assert real_buckets, "expected at least one real phase bucket"
    for datum in data:
        # citation_count is always the true bucket total; excerpts are capped at 5.
        assert datum.citation_count == datum.trial_count
        for c in datum.citations:
            assert c.nct_id in valid_ncts
            assert c.field == "Phase"
            assert c.excerpt == datum.phase  # exact phase value for this bucket

    # Real phase buckets carry the full (capped) set of citations.
    for datum in real_buckets:
        assert len(datum.citations) == min(5, datum.citation_count)

    # The (unknown) bucket has no Phase value to quote, so it carries none —
    # citations are never fabricated for a field that's absent.
    for datum in data:
        if datum.phase == "(unknown)":
            assert datum.citations == []


async def test_histogram_bins_carry_real_nct_citations():
    pages = ["enrollment_startdate.json"]
    valid_ncts = _nct_ids(*pages)
    resp = await _run(
        {"intent": "distribution_continuous", "measure": "enrollment_count",
         "binning": {"strategy": "auto"}},
        pages,
    )
    bins = resp.visualization.data
    assert bins, "expected histogram bins"
    for b in bins:
        assert b.citation_count == b.trial_count
        assert len(b.citations) == min(5, b.citation_count)
        for c in b.citations:
            assert c.nct_id in valid_ncts
            assert c.field == "EnrollmentCount"
            assert c.excerpt.isdigit()  # exact enrollment value


async def test_include_citations_false_omits_excerpts():
    pages = ["pembrolizumab_page1.json"]
    resp = await _run(
        {"intent": "distribution", "entities": {"condition": "melanoma"},
         "group_by": "phase", "measure": "study_count"},
        pages,
        include=False,
    )
    for datum in resp.visualization.data:
        assert datum.citations == []          # excerpts omitted
        assert datum.citation_count == datum.trial_count  # count retained
