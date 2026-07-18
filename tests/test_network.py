"""Network builder tests (offline).

Uses a melanoma-themed network fixture with typed interventions, a
duplicate-drug record (NCTMEL004), and a record missing its sponsor block
(NCTMEL005). NOTE: the captured melanoma_studies.json can't be used here — it
was projected name-only (no InterventionType) and every record has a sponsor,
so it can't exercise type-filtered drug edges or the missing-sponsor skip.
"""

import json
from pathlib import Path

from app.engine.compiler import compile_plan
from app.engine.network import build_network
from app.schemas.plan import AnalysisPlan, Intent, Network, NetworkNodeType

FIXTURE = Path(__file__).parent / "fixtures" / "network_melanoma.json"


def _studies():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["studies"]


def _edge(result, source, target):
    """Find an undirected drug-drug edge regardless of endpoint order."""
    pair = {source, target}
    return next(e for e in result.edges if {e.source, e.target} == pair)


def test_edges_weighted_by_supporting_study_count():
    net = build_network(_studies(), ["drug", "drug"], max_nodes=50)

    # Pembrolizumab+Nivolumab co-occur in MEL001, MEL004, MEL005 -> weight 3.
    pn = _edge(net, "Pembrolizumab", "Nivolumab")
    assert pn.weight == 3
    assert set(pn.nct_ids) == {"NCTMEL001", "NCTMEL004", "NCTMEL005"}

    # Each pair that appears in exactly one study has weight 1.
    assert _edge(net, "Pembrolizumab", "Ipilimumab").weight == 1
    assert _edge(net, "Nivolumab", "Ipilimumab").weight == 1

    # Edge weight equals the number of supporting studies it cites.
    for edge in net.edges:
        assert edge.weight == len(edge.nct_ids)

    # NCTMEL006 has no drugs -> contributes no edges, counted as skipped.
    assert net.skipped_count == 1


def test_duplicate_drug_in_one_study_does_not_inflate_weight():
    net = build_network(_studies(), ["drug", "drug"], max_nodes=50)
    nodes = {n.id: n for n in net.nodes}

    # NCTMEL004 lists Pembrolizumab twice; it must count as ONE supporting study.
    # Pembrolizumab appears in MEL001, MEL002, MEL004, MEL005 -> 4 (not 5).
    assert nodes["Pembrolizumab"].trial_count == 4
    assert set(nodes["Pembrolizumab"].nct_ids) == {
        "NCTMEL001", "NCTMEL002", "NCTMEL004", "NCTMEL005",
    }
    # And the Pembro-Nivo pair from MEL004 is counted once, not twice.
    assert _edge(net, "Pembrolizumab", "Nivolumab").weight == 3


def test_record_missing_sponsor_is_skipped_and_counted_not_crashed():
    # Bipartite sponsor->drug: MEL005 has no sponsor, MEL006 has no drugs.
    net = build_network(_studies(), ["sponsor", "drug"], max_nodes=50)

    assert net.skipped_count == 2  # MEL005 (no sponsor) + MEL006 (no drugs)

    # The sponsor-less study contributes nothing — its NCT appears nowhere.
    all_ncts = {nct for n in net.nodes for nct in n.nct_ids}
    all_ncts |= {nct for e in net.edges for nct in e.nct_ids}
    assert "NCTMEL005" not in all_ncts

    # A real bipartite edge is still built and weighted by supporting studies.
    alpha_pembro = next(
        e for e in net.edges if e.source == "Alpha Oncology" and e.target == "Pembrolizumab"
    )
    assert alpha_pembro.weight == 2  # MEL001 + MEL002


# --- condition node type -----------------------------------------------------


def _study(nct, *, conditions=None, sponsor=None, drugs=None):
    """Build a minimal study record with the modules the network builder reads."""
    protocol = {"identificationModule": {"nctId": nct, "briefTitle": f"{nct} trial"}}
    if conditions is not None:
        protocol["conditionsModule"] = {"conditions": conditions}
    if sponsor is not None:
        protocol["sponsorCollaboratorsModule"] = {"leadSponsor": {"name": sponsor}}
    if drugs is not None:
        protocol["armsInterventionsModule"] = {
            "interventions": [{"type": "DRUG", "name": d} for d in drugs]
        }
    return {"protocolSection": protocol}


def _condition_studies():
    # C3 has a single condition (skipped by co-occurrence); C4 has none at all.
    return [
        _study("NCTC001", conditions=["Melanoma", "Skin Cancer"], sponsor="Alpha Oncology", drugs=["Pembrolizumab"]),
        _study("NCTC002", conditions=["Melanoma", "Lung Cancer"], sponsor="Beta Pharma", drugs=["Nivolumab"]),
        _study("NCTC003", conditions=["Melanoma"], sponsor="Alpha Oncology", drugs=["Pembrolizumab"]),
        _study("NCTC004", conditions=[], sponsor="Gamma Institute", drugs=["Ipilimumab"]),
    ]


def test_condition_cooccurrence_pairs_and_skips():
    net = build_network(_condition_studies(), ["condition", "condition"], max_nodes=50)

    nodes = {n.id: n for n in net.nodes}
    assert nodes["Melanoma"].type == "condition"
    # Melanoma co-occurs with another condition only in C001 and C002 (C003 has one
    # condition -> skipped; C004 has none -> skipped).
    assert nodes["Melanoma"].trial_count == 2
    assert set(nodes["Melanoma"].nct_ids) == {"NCTC001", "NCTC002"}
    assert net.skipped_count == 2

    assert _edge(net, "Melanoma", "Skin Cancer").weight == 1
    assert _edge(net, "Melanoma", "Lung Cancer").weight == 1
    for edge in net.edges:
        assert edge.weight == len(edge.nct_ids)


def test_sponsor_condition_bipartite_edges_directed_and_weighted():
    net = build_network(_condition_studies(), ["sponsor", "condition"], max_nodes=50)

    nodes = {n.id: n for n in net.nodes}
    assert nodes["Alpha Oncology"].type == "sponsor"
    assert nodes["Melanoma"].type == "condition"

    # C004 has no conditions -> nothing to link -> skipped.
    assert net.skipped_count == 1

    # First requested type (sponsor) is the edge source.
    edge = next(
        e for e in net.edges if e.source == "Alpha Oncology" and e.target == "Melanoma"
    )
    assert edge.weight == 2  # C001 + C003
    assert set(edge.nct_ids) == {"NCTC001", "NCTC003"}


def test_drug_condition_bipartite_links_drugs_to_conditions():
    net = build_network(_condition_studies(), ["drug", "condition"], max_nodes=50)

    nodes = {n.id: n for n in net.nodes}
    assert nodes["Pembrolizumab"].type == "drug"
    assert nodes["Melanoma"].type == "condition"
    assert net.skipped_count == 1  # C004 has no conditions

    edge = next(
        e for e in net.edges if e.source == "Pembrolizumab" and e.target == "Melanoma"
    )
    assert edge.weight == 2  # C001 + C003


def test_unknown_node_types_fall_back_to_drug_drug():
    # An unrecognized type set degrades to drug-drug co-occurrence rather than crashing.
    net = build_network(_studies(), ["bogus"], max_nodes=50)
    assert {n.type for n in net.nodes} == {"drug"}


def test_network_intent_projects_condition_field():
    # A condition-node network needs the Condition field to reach the extractor.
    plan = AnalysisPlan(
        intent=Intent.network,
        network=Network(node_types=[NetworkNodeType.drug, NetworkNodeType.condition]),
    )
    params = compile_plan(plan)[0]
    projected = params["fields"].split(",")
    assert "Condition" in projected
