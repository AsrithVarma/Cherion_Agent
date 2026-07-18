"""Live smoke test against the ClinicalTrials.gov v2 API.

Marked ``live`` so it can be deselected offline with ``-m "not live"``.
"""

import pytest

from app.ctgov.client import CTGovClient

pytestmark = pytest.mark.live


async def test_search_studies_melanoma_live():
    """Querying a common condition returns a positive total and a record."""
    async with CTGovClient() as client:
        data = await client.search_studies(
            {
                "query.cond": "melanoma",
                "pageSize": 1,
                "countTotal": "true",
            }
        )

    # totalCount reflects the full result set, not just this page.
    assert data.get("totalCount", 0) > 0

    # A study record should come back and be shaped as expected.
    studies = data.get("studies")
    assert studies, "expected at least one study record"
    assert "protocolSection" in studies[0]
