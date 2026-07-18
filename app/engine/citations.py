"""Citation builders.

Each chart datum, histogram bin, network node/edge, and scatter point can carry
up to ``CITATION_LIMIT`` citations — ``{nct_id, field, excerpt}`` where the
excerpt is the EXACT field value drawn from a contributing record (a Phase
string, an EnrollmentCount value, an InterventionName, a sponsor name, or a
BriefTitle). ``citation_count`` (set by the caller) remains the true total,
independent of how many excerpts are attached.

Every value here comes from the API records + this Python; nothing is invented.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from app.dimensions.registry import Dimension
from app.schemas.response import Citation

CITATION_LIMIT = 5

Study = dict[str, Any]


def _dig(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _nct_id(study: Study) -> Optional[str]:
    return _dig(study, "protocolSection", "identificationModule", "nctId")


def brief_title(study: Study) -> Optional[str]:
    return _dig(study, "protocolSection", "identificationModule", "briefTitle")


def build_index(studies: list[Study]) -> dict[str, Study]:
    """Index studies by NCT id (first occurrence wins)."""
    index: dict[str, Study] = {}
    for study in studies:
        nct = _nct_id(study)
        if nct and nct not in index:
            index[nct] = study
    return index


def _excerpt_for_dimension(
    dimension: Dimension, record: Study, match_key: Optional[str]
) -> Optional[str]:
    """Exact field value for ``dimension`` from ``record``, matching a bucket key."""
    value = dimension.extractor(record)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if match_key is not None:
            for item in value:
                if str(item) == str(match_key):
                    return str(item)
        return str(value[0])
    return str(value)


def _collect(
    nct_ids: list[str],
    index: dict[str, Study],
    field: str,
    excerpt_fn: Callable[[Study], Optional[Any]],
    limit: int = CITATION_LIMIT,
) -> list[Citation]:
    """Build up to ``limit`` citations from contributing records."""
    out: list[Citation] = []
    for nct in nct_ids:
        if len(out) >= limit:
            break
        record = index.get(nct)
        if record is None:
            continue
        excerpt = excerpt_fn(record)
        if excerpt is None:
            continue
        out.append(Citation(nct_id=nct, field=field, excerpt=str(excerpt)))
    return out


def dimension_citations(
    nct_ids: list[str],
    index: dict[str, Study],
    dimension: Dimension,
    match_key: Optional[str] = None,
    limit: int = CITATION_LIMIT,
) -> list[Citation]:
    """Cite a dimension's exact value (field = the dimension's piece name)."""
    return _collect(
        nct_ids,
        index,
        field=dimension.piece,
        excerpt_fn=lambda rec: _excerpt_for_dimension(dimension, rec, match_key),
        limit=limit,
    )


def value_citations(
    nct_ids: list[str],
    index: dict[str, Study],
    field: str,
    value: Any,
    limit: int = CITATION_LIMIT,
) -> list[Citation]:
    """Cite a known exact value (e.g. a node's own sponsor/drug name)."""
    return _collect(nct_ids, index, field=field, excerpt_fn=lambda _rec: value, limit=limit)


def title_citations(
    nct_ids: list[str],
    index: dict[str, Study],
    limit: int = CITATION_LIMIT,
) -> list[Citation]:
    """Cite each study's BriefTitle."""
    return _collect(nct_ids, index, field="BriefTitle", excerpt_fn=brief_title, limit=limit)
