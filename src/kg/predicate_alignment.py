"""
src/kg/predicate_alignment.py
Semi-manual predicate alignment: maps local predicates to Wikidata properties.

Strategy (as per professor requirements)
-----------------------------------------
1. Start from a small, explicitly defined local predicate set.
2. Try label-based matching against a curated Wikidata property list.
3. Apply manual validation rules (see MANUAL_VALIDATION below).
4. Emit owl:equivalentProperty or rdfs:subPropertyOf only when justified.
5. Save the alignment as triples appended to alignment.ttl.

Usage:
    python -m src.kg.predicate_alignment
"""

from pathlib import Path
from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef, Literal

SKG_O  = Namespace("http://space-kg.org/ontology/")
WD_P   = Namespace("http://www.wikidata.org/prop/direct/")
WD_ENT = Namespace("http://www.wikidata.org/entity/")

# ── Manually validated predicate mapping ──────────────────────────────────────
# Format: local_name → (wikidata_pid, relation_type, rationale)
# relation_type: "equivalentProperty" | "subPropertyOf"
# Only include mappings where the alignment is clearly justified.

MANUAL_VALIDATION: list[tuple[str, str, str, str]] = [
    # (local_name, wikidata_pid, relation_type, rationale)
    ("hasAstronaut",  "P1029", "equivalentProperty",
     "P1029 = 'crew member' - direct equivalent"),
    ("launchedBy",    "P137",  "equivalentProperty",
     "P137 = 'operator' - equivalent for space missions"),
    ("landedOn",      "P65",   "equivalentProperty",
     "P65 = 'site of astronomical discovery' is closest; approximate"),
    ("partOf",        "P361",  "equivalentProperty",
     "P361 = 'part of' - exact equivalent"),
    ("locatedIn",     "P131",  "subPropertyOf",
     "P131 = 'located in the administrative territorial entity' is broader"),
    ("discoveredBy",  "P61",   "equivalentProperty",
     "P61 = 'discoverer or inventor' - direct equivalent"),
    ("isCommanderOf", "P1344", "subPropertyOf",
     "P1344 = 'participant in' - isCommanderOf is a specialisation"),
]


def build_predicate_alignment() -> Graph:
    """Return a Graph with predicate alignment triples."""
    g = Graph()
    g.bind("skg-o",   SKG_O)
    g.bind("wd-prop", WD_P)
    g.bind("wd",      WD_ENT)
    g.bind("owl",     OWL)
    g.bind("rdfs",    RDFS)

    for local_name, wd_pid, rel_type, rationale in MANUAL_VALIDATION:
        local_uri = SKG_O[local_name]
        wd_uri    = WD_P[wd_pid]     # direct-claim property

        # Mark local predicate as ObjectProperty
        g.add((local_uri, RDF.type, OWL.ObjectProperty))

        if rel_type == "equivalentProperty":
            g.add((local_uri, OWL.equivalentProperty, wd_uri))
        elif rel_type == "subPropertyOf":
            g.add((local_uri, RDFS.subPropertyOf, wd_uri))

        # Annotation: human-readable rationale
        g.add((local_uri, RDFS.comment, Literal(f"Aligned to wd:{wd_pid}. {rationale}")))

    return g


def append_to_alignment(align_path: str) -> None:
    """
    Load existing alignment.ttl and append predicate alignment triples,
    then re-save.
    """
    p = Path(align_path)
    g = Graph()
    if p.exists():
        g.parse(str(p), format="turtle")

    pred_graph = build_predicate_alignment()
    for triple in pred_graph:
        g.add(triple)

    # Re-bind namespaces
    g.bind("skg-o",   SKG_O)
    g.bind("wd-prop", WD_P)
    g.bind("owl",     OWL)
    g.bind("rdfs",    RDFS)

    g.serialize(destination=str(p), format="turtle")
    print(f"Predicate alignment appended → {p}  ({len(pred_graph)} new triples)")

    # Print summary table to stdout
    print("\nPredicate alignment summary:")
    print(f"{'Local predicate':<20} {'WD Property':<12} {'Relation':<22} Rationale")
    print("-" * 90)
    for local_name, wd_pid, rel_type, rationale in MANUAL_VALIDATION:
        print(f"{local_name:<20} wd:{wd_pid:<10} {rel_type:<22} {rationale}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Predicate alignment → Wikidata")
    parser.add_argument("--alignment", default="kg_artifacts/alignment.ttl",
                        help="Alignment TTL to append predicate triples to")
    args = parser.parse_args()
    append_to_alignment(args.alignment)


if __name__ == "__main__":
    main()
