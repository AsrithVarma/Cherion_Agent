"""Startup-only introspection of the ClinicalTrials.gov v2 API.

At boot we call the four metadata endpoints once and cache the results in a
module-level singleton:

- ``/studies/enums``       -> ``{field -> set(valid_values)}`` plus a
                              ``legacyValue -> value`` map (so "Recruiting"
                              resolves to ``RECRUITING``).
- ``/studies/metadata``    -> ``{piece_name -> json_path}`` (dotted path of the
                              ``name`` fields from the record root to the piece).
- ``/studies/search-areas``-> the set of valid ``AREA[...]`` names.
- ``/version``             -> the API version and ``dataTimestamp``.

If a call fails, we log which path was used and, for enums specifically, fall
back to the hardcoded lists from the project brief so downstream normalization
still works offline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --- Hardcoded fallback enum whitelists (from the project brief) -----------
# Keyed by the canonical CT.gov piece/type name. Used only if the live
# /studies/enums call fails; live values are preferred when available.
FALLBACK_ENUMS: dict[str, set[str]] = {
    "Phase": {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"},
    "OverallStatus": {
        "RECRUITING",
        "NOT_YET_RECRUITING",
        "ENROLLING_BY_INVITATION",
        "ACTIVE_NOT_RECRUITING",
        "SUSPENDED",
        "TERMINATED",
        "COMPLETED",
        "WITHDRAWN",
        "UNKNOWN",
    },
    "StudyType": {"INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"},
    "LeadSponsorClass": {"NIH", "FED", "INDUSTRY", "NETWORK", "OTHER"},
    "InterventionType": {
        "DRUG",
        "DEVICE",
        "BIOLOGICAL",
        "PROCEDURE",
        "RADIATION",
        "BEHAVIORAL",
        "GENETIC",
        "DIETARY_SUPPLEMENT",
        "COMBINATION_PRODUCT",
        "DIAGNOSTIC_TEST",
        "OTHER",
    },
}

# Aliases mapping user/dimension-style field names to canonical enum keys.
_FIELD_ALIASES: dict[str, str] = {
    "phase": "Phase",
    "status": "OverallStatus",
    "overallstatus": "OverallStatus",
    "overall_status": "OverallStatus",
    "last_known_status": "LastKnownStatus",
    "studytype": "StudyType",
    "study_type": "StudyType",
    "leadsponsorclass": "LeadSponsorClass",
    "lead_sponsor_class": "LeadSponsorClass",
    "sponsor_class": "LeadSponsorClass",
    "interventiontype": "InterventionType",
    "intervention_type": "InterventionType",
}

# Roman numerals used in legacy phase labels ("Phase II" -> 2).
_ROMAN_TO_ARABIC = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}


class Introspection:
    """Cached, parsed view of the CT.gov API metadata (startup-only)."""

    def __init__(self) -> None:
        self.enum_values: dict[str, set[str]] = {}
        self.enum_legacy: dict[str, dict[str, str]] = {}
        self.metadata_paths: dict[str, str] = {}
        self.area_names: set[str] = set()
        self.area_params: set[str] = set()
        self.api_version: Optional[str] = None
        self.data_timestamp: Optional[str] = None
        # Which path each component came from: "live" or "fallback".
        self.sources: dict[str, str] = {}
        self.loaded: bool = False

    # -- loading -----------------------------------------------------------

    async def load(self, client: Any) -> None:
        """Populate the cache from the live API, falling back where needed.

        Each endpoint is fetched independently so one failure does not abort the
        rest. Enums fall back to :data:`FALLBACK_ENUMS`; the other components
        simply stay empty (and are marked ``fallback``) if they fail.
        """
        await self._load_version(client)
        await self._load_enums(client)
        await self._load_metadata(client)
        await self._load_search_areas(client)

        self.loaded = True
        logger.info("Introspection loaded. sources=%s", self.sources)

    async def _load_version(self, client: Any) -> None:
        try:
            payload = await client.get("/version")
            self.api_version = payload.get("apiVersion")
            self.data_timestamp = payload.get("dataTimestamp")
            self.sources["version"] = "live"
            logger.info(
                "Introspection /version live: api=%s dataTimestamp=%s",
                self.api_version,
                self.data_timestamp,
            )
        except Exception as exc:  # noqa: BLE001 — startup must be resilient
            self.sources["version"] = "fallback"
            logger.warning("Introspection /version failed (%s); no data timestamp.", exc)

    async def _load_enums(self, client: Any) -> None:
        try:
            payload = await client.get("/studies/enums")
            self._parse_enums(payload)
            self._merge_missing_fallback_enums()
            self.sources["enums"] = "live"
            logger.info("Introspection /studies/enums live: %d fields.", len(self.enum_values))
        except Exception as exc:  # noqa: BLE001
            self._load_fallback_enums()
            self.sources["enums"] = "fallback"
            logger.warning(
                "Introspection /studies/enums failed (%s); using hardcoded fallback enums.",
                exc,
            )

    async def _load_metadata(self, client: Any) -> None:
        try:
            payload = await client.get("/studies/metadata")
            _walk_metadata(payload, "", self.metadata_paths)
            self.sources["metadata"] = "live"
            logger.info(
                "Introspection /studies/metadata live: %d pieces mapped.",
                len(self.metadata_paths),
            )
        except Exception as exc:  # noqa: BLE001
            self.sources["metadata"] = "fallback"
            logger.warning(
                "Introspection /studies/metadata failed (%s); piece paths unavailable.", exc
            )

    async def _load_search_areas(self, client: Any) -> None:
        try:
            payload = await client.get("/studies/search-areas")
            self._parse_search_areas(payload)
            self.sources["search-areas"] = "live"
            logger.info(
                "Introspection /studies/search-areas live: %d area names.", len(self.area_names)
            )
        except Exception as exc:  # noqa: BLE001
            self.sources["search-areas"] = "fallback"
            logger.warning(
                "Introspection /studies/search-areas failed (%s); AREA names unavailable.", exc
            )

    # -- parsing -----------------------------------------------------------

    def _parse_enums(self, payload: list[dict[str, Any]]) -> None:
        """Parse the /studies/enums array into value sets and a legacy map.

        Each enum is indexed both by its ``type`` (e.g. "Status") and by every
        piece that uses it (e.g. "OverallStatus", "LastKnownStatus").
        """
        for entry in payload or []:
            values: set[str] = set()
            legacy: dict[str, str] = {}
            for item in entry.get("values", []) or []:
                canonical = item.get("value")
                if not canonical:
                    continue
                values.add(canonical)
                legacy_label = item.get("legacyValue")
                if legacy_label:
                    legacy[legacy_label.strip().lower()] = canonical

            keys = list(entry.get("pieces", []) or [])
            enum_type = entry.get("type")
            if enum_type:
                keys.append(enum_type)
            for key in keys:
                self.enum_values.setdefault(key, set()).update(values)
                self.enum_legacy.setdefault(key, {}).update(legacy)

    def _merge_missing_fallback_enums(self) -> None:
        """Ensure the five brief-critical fields exist even if live omits one."""
        for field, values in FALLBACK_ENUMS.items():
            if not self.enum_values.get(field):
                self.enum_values.setdefault(field, set()).update(values)

    def _load_fallback_enums(self) -> None:
        for field, values in FALLBACK_ENUMS.items():
            self.enum_values[field] = set(values)
            self.enum_legacy.setdefault(field, {})

    def _parse_search_areas(self, payload: list[dict[str, Any]]) -> None:
        for group in payload or []:
            for area in group.get("areas", []) or []:
                name = area.get("name")
                if name:
                    self.area_names.add(name)
                param = area.get("param")
                if param:
                    self.area_params.add(param)

    # -- accessors ---------------------------------------------------------

    def valid_values(self, field: str) -> set[str]:
        """Return the whitelist of canonical values for ``field`` (may be empty)."""
        key = self.resolve_field(field)
        return set(self.enum_values.get(key, set())) if key else set()

    def json_path(self, piece: str) -> Optional[str]:
        """Return the dotted JSON path for a metadata piece, or None."""
        return self.metadata_paths.get(piece)

    def valid_area_names(self) -> set[str]:
        """Return the set of valid ``AREA[...]`` names from search-areas."""
        return set(self.area_names)

    def resolve_field(self, field: str) -> Optional[str]:
        """Resolve a user/dimension field name to a canonical enum key."""
        if field in self.enum_values:
            return field
        normalized = field.lower().replace(" ", "_")
        alias = _FIELD_ALIASES.get(normalized)
        if alias and alias in self.enum_values:
            return alias
        for key in self.enum_values:
            if key.lower() == field.lower():
                return key
        # Alias to a canonical name even if not present (fallback guarantees it).
        return alias if alias in self.enum_values else None


# --- module-level singleton -------------------------------------------------

_introspection = Introspection()


def get_introspection() -> Introspection:
    """Return the process-wide introspection singleton."""
    return _introspection


async def load_introspection(client: Any) -> Introspection:
    """Load the singleton from the given CT.gov client and return it."""
    await _introspection.load(client)
    return _introspection


# --- enum normalization -----------------------------------------------------


def normalize_enum(field: str, raw_value: Any) -> Optional[str]:
    """Map a user-supplied value to its canonical enum value, or None.

    Resolution order, using the cached whitelist + legacy map:

    1. Exact canonical match (already ``PHASE3``).
    2. Case-insensitive value match ("recruiting" -> ``RECRUITING``).
    3. Legacy label map ("Active, not recruiting" -> ``ACTIVE_NOT_RECRUITING``).
    4. Alphanumeric-only comparison ("phase 3" -> ``PHASE3``).
    5. Phase-specific heuristic incl. roman numerals ("Phase II" -> ``PHASE2``).

    Returns None if the field is unknown or the value cannot be mapped.
    """
    intro = get_introspection()
    key = intro.resolve_field(field)
    if key is None:
        return None
    whitelist = intro.enum_values.get(key)
    if not whitelist:
        return None

    raw = str(raw_value).strip()
    if not raw:
        return None
    lower = raw.lower()

    # 1. exact canonical
    if raw in whitelist:
        return raw

    # 2. case-insensitive value match
    for value in whitelist:
        if value.lower() == lower:
            return value

    # 3. legacy label map
    legacy = intro.enum_legacy.get(key, {})
    if lower in legacy:
        return legacy[lower]

    # 4. alphanumeric-only comparison (drops spaces, commas, hyphens)
    squashed = re.sub(r"[^a-z0-9]", "", lower)
    for value in whitelist:
        if re.sub(r"[^a-z0-9]", "", value.lower()) == squashed:
            return value

    # 5. phase heuristic
    if key == "Phase":
        candidate = _canonical_phase(lower)
        if candidate and candidate in whitelist:
            return candidate

    return None


def _canonical_phase(lower_value: str) -> Optional[str]:
    """Best-effort mapping of a free-form phase label to a ``PHASE*`` value."""
    text = lower_value.replace("-", " ").replace("/", " ").strip()
    if "early" in text:
        return "EARLY_PHASE1"
    if "not applicable" in text or text in {"na", "n/a"}:
        return "NA"

    match = re.search(r"phase\s*([0-9]+|iv|iii|ii|i)\b", text)
    if not match:
        match = re.fullmatch(r"\s*([0-9]+|iv|iii|ii|i)\s*", text)
    if not match:
        return None

    token = match.group(1)
    token = _ROMAN_TO_ARABIC.get(token, token)
    return f"PHASE{token}"


# --- metadata walk ----------------------------------------------------------


def _walk_metadata(
    nodes: list[dict[str, Any]], prefix: str, out: dict[str, str]
) -> None:
    """Recursively map each node's ``piece`` to its dotted ``name`` path."""
    for node in nodes or []:
        name = node.get("name")
        if not name:
            continue
        path = f"{prefix}.{name}" if prefix else name
        piece = node.get("piece")
        if piece:
            out[piece] = path
        children = node.get("children")
        if children:
            _walk_metadata(children, path, out)
