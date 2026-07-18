"""Dimension registry — the single source of truth for extractable fields.

Each :class:`Dimension` knows how to project a field from the API (``api_field``
/ ``piece``), how to pull a value out of a study record (``extractor``), whether
it is categorical or numeric, whether it is enum-constrained, and where a
citation for that value lives (``citation_field_path``).

Defensive extraction is non-negotiable: ClinicalTrials.gov records routinely
omit whole modules or carry empty arrays. Every extractor returns ``None`` on
missing/empty data and never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

# A study record is the raw JSON dict (nesting data under ``protocolSection``).
Study = dict[str, Any]
# Extractors may return a scalar, a list (multi-valued fields), or None.
ExtractResult = Union[str, int, float, list, None]


@dataclass(frozen=True)
class Dimension:
    """Describes one groupable/measurable field of a study."""

    name: str
    api_field: str
    piece: str
    extractor: Callable[[Study], ExtractResult]
    kind: str  # "categorical" | "numeric"
    is_enum: bool
    citation_field_path: str


# --- shared helpers ---------------------------------------------------------


def _dig(obj: Any, *path: str) -> Any:
    """Walk nested dict keys, returning None if any level is missing/not a dict."""
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def parse_year(raw: Any) -> Optional[int]:
    """Extract a 4-digit year from a messy date string; None if unparseable.

    Handles the common ClinicalTrials.gov shapes:
    ``"2024-01-15"``, ``"2024-03"``, ``"2024"``, ``"January 2024"``,
    ``"January 15, 2024"``. Never raises.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 1000 <= raw <= 9999 else None

    text = str(raw).strip()
    if not text:
        return None

    # Prefer a leading year ("2024", "2024-01-15", "2024-03").
    leading = re.match(r"^(\d{4})\b", text)
    if leading:
        return int(leading.group(1))

    # Otherwise the first 4-digit run ("January 2024", "January 15, 2024").
    anywhere = re.search(r"\b(\d{4})\b", text)
    if anywhere:
        return int(anywhere.group(1))

    return None


def _dedupe(values: list) -> list:
    """Return values with duplicates removed, order preserved."""
    seen: set = set()
    out: list = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --- extractors -------------------------------------------------------------


def _extract_phase(study: Study) -> Optional[list]:
    phases = _dig(study, "protocolSection", "designModule", "phases")
    if isinstance(phases, list):
        values = [p for p in phases if p]
        return values or None
    return None


def _extract_start_year(study: Study) -> Optional[int]:
    date = _dig(study, "protocolSection", "statusModule", "startDateStruct", "date")
    return parse_year(date)


def _extract_overall_status(study: Study) -> Optional[str]:
    return _dig(study, "protocolSection", "statusModule", "overallStatus") or None


def _extract_study_type(study: Study) -> Optional[str]:
    return _dig(study, "protocolSection", "designModule", "studyType") or None


def _extract_intervention_type(study: Study) -> Optional[list]:
    interventions = _dig(
        study, "protocolSection", "armsInterventionsModule", "interventions"
    )
    if isinstance(interventions, list):
        types = [
            item.get("type")
            for item in interventions
            if isinstance(item, dict) and item.get("type")
        ]
        return _dedupe(types) or None
    return None


def _extract_lead_sponsor_class(study: Study) -> Optional[str]:
    return (
        _dig(
            study,
            "protocolSection",
            "sponsorCollaboratorsModule",
            "leadSponsor",
            "class",
        )
        or None
    )


def _extract_lead_sponsor_name(study: Study) -> Optional[str]:
    return (
        _dig(
            study,
            "protocolSection",
            "sponsorCollaboratorsModule",
            "leadSponsor",
            "name",
        )
        or None
    )


def _extract_condition(study: Study) -> Optional[list]:
    conditions = _dig(study, "protocolSection", "conditionsModule", "conditions")
    if isinstance(conditions, list):
        values = [c for c in conditions if c]
        return _dedupe(values) or None
    return None


def _extract_country(study: Study) -> Optional[list]:
    locations = _dig(
        study, "protocolSection", "contactsLocationsModule", "locations"
    )
    if isinstance(locations, list):
        countries = [
            loc.get("country")
            for loc in locations
            if isinstance(loc, dict) and loc.get("country")
        ]
        return _dedupe(countries) or None
    return None


def _extract_enrollment_count(study: Study) -> Optional[int]:
    count = _dig(study, "protocolSection", "designModule", "enrollmentInfo", "count")
    if isinstance(count, bool):
        return None
    if isinstance(count, int):
        return count
    if isinstance(count, str):
        try:
            return int(count.strip())
        except ValueError:
            return None
    return None


# --- registry ---------------------------------------------------------------

REGISTRY: dict[str, Dimension] = {
    "phase": Dimension(
        name="phase",
        api_field="Phase",
        piece="Phase",
        extractor=_extract_phase,
        kind="categorical",
        is_enum=True,
        citation_field_path="protocolSection.designModule.phases",
    ),
    "start_year": Dimension(
        name="start_year",
        api_field="StartDate",
        piece="StartDate",
        extractor=_extract_start_year,
        kind="categorical",
        is_enum=False,
        citation_field_path="protocolSection.statusModule.startDateStruct.date",
    ),
    "overall_status": Dimension(
        name="overall_status",
        api_field="OverallStatus",
        piece="OverallStatus",
        extractor=_extract_overall_status,
        kind="categorical",
        is_enum=True,
        citation_field_path="protocolSection.statusModule.overallStatus",
    ),
    "study_type": Dimension(
        name="study_type",
        api_field="StudyType",
        piece="StudyType",
        extractor=_extract_study_type,
        kind="categorical",
        is_enum=True,
        citation_field_path="protocolSection.designModule.studyType",
    ),
    "intervention_type": Dimension(
        name="intervention_type",
        api_field="InterventionType",
        piece="InterventionType",
        extractor=_extract_intervention_type,
        kind="categorical",
        is_enum=True,
        citation_field_path="protocolSection.armsInterventionsModule.interventions.type",
    ),
    "lead_sponsor_class": Dimension(
        name="lead_sponsor_class",
        api_field="LeadSponsorClass",
        piece="LeadSponsorClass",
        extractor=_extract_lead_sponsor_class,
        kind="categorical",
        is_enum=True,
        citation_field_path="protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    ),
    "lead_sponsor_name": Dimension(
        name="lead_sponsor_name",
        api_field="LeadSponsorName",
        piece="LeadSponsorName",
        extractor=_extract_lead_sponsor_name,
        kind="categorical",
        is_enum=False,
        citation_field_path="LeadSponsorName",
    ),
    "condition": Dimension(
        name="condition",
        api_field="Condition",
        piece="Condition",
        extractor=_extract_condition,
        kind="categorical",
        is_enum=False,
        citation_field_path="protocolSection.conditionsModule.conditions",
    ),
    "country": Dimension(
        name="country",
        api_field="LocationCountry",
        piece="LocationCountry",
        extractor=_extract_country,
        kind="categorical",
        is_enum=False,
        citation_field_path="protocolSection.contactsLocationsModule.locations.country",
    ),
    "enrollment_count": Dimension(
        name="enrollment_count",
        api_field="EnrollmentCount",
        piece="EnrollmentCount",
        extractor=_extract_enrollment_count,
        kind="numeric",
        is_enum=False,
        citation_field_path="protocolSection.designModule.enrollmentInfo.count",
    ),
    "start_year_numeric": Dimension(
        name="start_year_numeric",
        api_field="StartDate",
        piece="StartDate",
        extractor=_extract_start_year,
        kind="numeric",
        is_enum=False,
        citation_field_path="protocolSection.statusModule.startDateStruct.date",
    ),
}


def get_dimension(name: str) -> Optional[Dimension]:
    """Return the registered dimension for ``name``, or None."""
    return REGISTRY.get(name)


def all_dimensions() -> list[Dimension]:
    """Return all registered dimensions."""
    return list(REGISTRY.values())
