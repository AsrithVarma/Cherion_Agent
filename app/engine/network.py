"""Network builder for intent=network.

Builds co-occurrence / bipartite graphs from study records using defensive
extraction (mirroring the registry style — nothing here raises on missing data).

Node types (``sponsor``, ``drug``, ``condition``) drive the graph shape:

- two distinct types (e.g. ``[sponsor, drug]``, ``[drug, condition]``) ->
  a **bipartite** graph: within each study, every value of the first type is
  linked to every value of the second type. The first type is the edge source.
- a single type (e.g. ``[drug, drug]`` or ``[condition]``) -> a **co-occurrence**
  graph: an edge for every pair of that type's values co-occurring in a study
  (pairs deduped per study; needs at least two distinct values).

A study that can't contribute an edge (bipartite: missing a value on either side;
co-occurrence: fewer than two distinct values) contributes nothing and increments
``skipped_count``. Node weight = number of studies containing it; edge weight =
number of studies supporting the pair; supporting NCT ids are kept per node and
edge. Results are capped to the top-weighted N nodes (config), flagging
truncation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Optional

from app.dimensions.registry import get_dimension

Study = dict[str, Any]

_DRUG_TYPES = {"DRUG", "BIOLOGICAL"}


@dataclass
class NetworkNode:
    id: str
    type: str
    trial_count: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class NetworkEdge:
    source: str
    target: str
    weight: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class NetworkResult:
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    skipped_count: int
    truncated: bool


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _drug_names(study: Study) -> list[str]:
    """DRUG/BIOLOGICAL intervention names for a study (deduped); [] if none."""
    interventions = _dig(study, "protocolSection", "armsInterventionsModule", "interventions")
    if not isinstance(interventions, list):
        return []
    names = [
        item.get("name")
        for item in interventions
        if isinstance(item, dict) and item.get("type") in _DRUG_TYPES and item.get("name")
    ]
    return _dedupe(names)


def _sponsor_names(study: Study) -> list[str]:
    """The lead sponsor name as a 0/1-length list (uniform with multi-valued types)."""
    name = get_dimension("lead_sponsor_name").extractor(study)
    return [name] if name else []


def _condition_names(study: Study) -> list[str]:
    """Condition names for a study (deduped by the registry extractor); [] if none."""
    conditions = get_dimension("condition").extractor(study)
    return conditions if isinstance(conditions, list) else []


# Per-node-type extractor: each returns the study's values for that type as a
# (possibly empty) deduped list of names.
_NODE_EXTRACTORS: dict[str, Callable[[Study], list[str]]] = {
    "sponsor": _sponsor_names,
    "drug": _drug_names,
    "condition": _condition_names,
}


def _resolve_node_types(node_types: list[str]) -> tuple[str, str, bool]:
    """Normalize requested node types into (source_type, target_type, bipartite).

    Unknown types are dropped; an empty/unknown request falls back to drug-drug.
    Two distinct known types => bipartite (first is the edge source); one type
    (or a repeated type like ``[drug, drug]``) => co-occurrence within that type.
    """
    ordered = [t for t in _dedupe([str(t) for t in node_types]) if t in _NODE_EXTRACTORS]
    if not ordered:
        ordered = ["drug"]
    if len(ordered) >= 2:
        return ordered[0], ordered[1], True
    return ordered[0], ordered[0], False


def build_network(
    studies: list[Study],
    node_types: list[str],
    max_nodes: int,
) -> NetworkResult:
    """Build a bipartite or co-occurrence network over the requested node types."""
    source_type, target_type, bipartite = _resolve_node_types(node_types)
    source_extractor = _NODE_EXTRACTORS[source_type]
    target_extractor = _NODE_EXTRACTORS[target_type]

    node_type: dict[str, str] = {}
    node_count: Counter[str] = Counter()
    node_ncts: dict[str, set[str]] = defaultdict(set)
    edge_count: Counter[tuple[str, str]] = Counter()
    edge_ncts: dict[tuple[str, str], set[str]] = defaultdict(set)
    skipped = 0

    def add_node(node_id: str, kind: str, nct: Optional[str]) -> None:
        node_type.setdefault(node_id, kind)
        node_count[node_id] += 1
        if nct:
            node_ncts[node_id].add(nct)

    def add_edge(key: tuple[str, str], nct: Optional[str]) -> None:
        edge_count[key] += 1
        if nct:
            edge_ncts[key].add(nct)

    for study in studies:
        nct = _nct_id(study)

        if bipartite:
            sources = source_extractor(study)
            targets = target_extractor(study)
            if not sources or not targets:
                skipped += 1
                continue
            for src in sources:
                add_node(src, source_type, nct)
            for tgt in targets:
                add_node(tgt, target_type, nct)
            for src in sources:
                for tgt in targets:
                    add_edge((src, tgt), nct)
        else:
            # co-occurrence: need at least two distinct values of the one type.
            values = source_extractor(study)
            if len(values) < 2:
                skipped += 1
                continue
            for value in values:
                add_node(value, source_type, nct)
            for a, b in combinations(sorted(values), 2):
                add_edge((a, b), nct)

    # Cap to the top-weighted N nodes; drop edges that lose an endpoint.
    all_ids = sorted(node_type, key=lambda i: (-node_count[i], i))
    truncated = len(all_ids) > max_nodes
    kept_ids = set(all_ids[:max_nodes])

    nodes = [
        NetworkNode(
            id=node_id,
            type=node_type[node_id],
            trial_count=node_count[node_id],
            nct_ids=sorted(node_ncts[node_id]),
        )
        for node_id in all_ids
        if node_id in kept_ids
    ]
    edges = [
        NetworkEdge(
            source=source,
            target=target,
            weight=edge_count[(source, target)],
            nct_ids=sorted(edge_ncts[(source, target)]),
        )
        for (source, target) in edge_count
        if source in kept_ids and target in kept_ids
    ]
    edges.sort(key=lambda e: -e.weight)

    return NetworkResult(nodes=nodes, edges=edges, skipped_count=skipped, truncated=truncated)
