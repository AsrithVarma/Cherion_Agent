"""Capture live ClinicalTrials.gov responses as JSON fixtures for offline tests.

Run from the repo root:

    python scripts/capture_fixtures.py

Saves into ``tests/fixtures/``:
  - pembrolizumab_page1.json / pembrolizumab_page2.json
        pembrolizumab studies, fields NCTId,Phase,BriefTitle (2 pages)
  - melanoma_studies.json
        melanoma studies, fields NCTId,InterventionName,LeadSponsorName,Condition,BriefTitle
  - enrollment_startdate.json
        records including EnrollmentCount and StartDate

Requests are spaced ~1s apart to stay polite. This script hits the network; the
saved fixtures are what the offline test-suite uses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/capture_fixtures.py`) by
# putting the repo root on the path ahead of the script's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ctgov.client import CTGovClient  # noqa: E402 — after sys.path bootstrap

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# Be polite: ~1 request/second.
_REQUEST_SPACING_SECONDS = 1.0


def _save(name: str, payload: dict) -> Path:
    """Write ``payload`` as pretty JSON into the fixtures dir; return the path."""
    path = FIXTURES_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    studies = payload.get("studies", [])
    print(f"  saved {name} ({len(studies)} studies)")
    return path


async def capture(client: CTGovClient) -> None:
    """Fetch and save all three fixture sets."""
    # (a) pembrolizumab studies, 2 pages.
    print("(a) pembrolizumab (NCTId,Phase,BriefTitle), 2 pages")
    params_a = {
        "query.intr": "pembrolizumab",
        "fields": "NCTId,Phase,BriefTitle",
        "pageSize": 20,
        "countTotal": "true",
    }
    page_num = 0
    async for page in client.paginate(params_a, max_pages=2):
        page_num += 1
        _save(f"pembrolizumab_page{page_num}.json", page)
        await asyncio.sleep(_REQUEST_SPACING_SECONDS)

    # (b) melanoma studies with intervention / sponsor / condition fields.
    print("(b) melanoma (NCTId,InterventionName,LeadSponsorName,Condition,BriefTitle)")
    page_b = await client.search_studies(
        {
            "query.cond": "melanoma",
            "fields": "NCTId,InterventionName,LeadSponsorName,Condition,BriefTitle",
            "pageSize": 25,
            "countTotal": "true",
        }
    )
    _save("melanoma_studies.json", page_b)
    await asyncio.sleep(_REQUEST_SPACING_SECONDS)

    # (c) records carrying EnrollmentCount and StartDate.
    print("(c) melanoma (NCTId,BriefTitle,EnrollmentCount,StartDate)")
    page_c = await client.search_studies(
        {
            "query.cond": "melanoma",
            "fields": "NCTId,BriefTitle,EnrollmentCount,StartDate",
            "pageSize": 25,
            "countTotal": "true",
        }
    )
    _save("enrollment_startdate.json", page_c)


async def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capturing fixtures into {FIXTURES_DIR}")
    async with CTGovClient() as client:
        await capture(client)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
