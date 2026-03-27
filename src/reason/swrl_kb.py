"""
src/reason/swrl_kb.py
One SWRL rule on the project's own ontology.

Rule (simple Horn clause with 2 conditions):
    Astronaut(?x) ∧ isCommanderOf(?x, ?m) → MissionCommander(?x)

This infers that any Astronaut who commands a mission belongs to
the MissionCommander class (a subclass of Astronaut in the ontology).

Usage:
    python -m src.reason.swrl_kb
    python -m src.reason.swrl_kb --ontology kg_artifacts/ontology.ttl \
                                   --graph kg_artifacts/initial_graph.ttl
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPACE_NS = "http://space-kg.org/ontology/"
SKG_R_NS = "http://space-kg.org/resource/"


def run_swrl_kb(ontology_path: str, graph_path: str) -> None:
    """
    Load the project ontology + initial graph, add the SWRL rule,
    run reasoning, and print inferred MissionCommander individuals.
    """
    try:
        from owlready2 import get_ontology, Thing, ObjectProperty, Imp, sync_reasoner_pellet
    except ImportError:
        logger.error("owlready2 is not installed.")
        return

    onto_file = Path(ontology_path).resolve()
    graph_file = Path(graph_path).resolve()

    if not onto_file.exists():
        logger.warning(
            "Ontology not found at %s. "
            "Run: python -m src.kg.ontology_builder first.\n"
            "Falling back to built-in demo …", onto_file
        )
        _run_demo_mode()
        return

    import tempfile
    import rdflib

    # Owlready2 doesn't natively support Turtle syntax, only RDF/XML and N-Triples.
    # So we use rdflib to load the turtle files and save them as a temporary .nt file.
    logger.info("Converting Turtle to N-Triples for Owlready2…")
    g = rdflib.Graph()
    g.parse(str(onto_file), format="turtle")
    if graph_file.exists():
        g.parse(str(graph_file), format="turtle")

    with tempfile.NamedTemporaryFile(suffix=".nt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    g.serialize(destination=str(tmp_path), format="nt")

    onto = get_ontology(f"file://{tmp_path.as_posix()}").load()

    with onto:
        # Ensure required classes/properties exist
        Astronaut_cls       = onto.search_one(iri=SPACE_NS + "Astronaut")
        MissionCommander_cls = onto.search_one(iri=SPACE_NS + "MissionCommander")
        isCommanderOf_prop  = onto.search_one(iri=SPACE_NS + "isCommanderOf")

        if None in (Astronaut_cls, MissionCommander_cls, isCommanderOf_prop):
            logger.warning(
                "Required ontology terms not found. "
                "Run python -m src.kg.ontology_builder first.\n"
                "Running demo mode …"
            )
            _run_demo_mode()
            return

        # Add SWRL rule
        # Astronaut(?x) ∧ isCommanderOf(?x, ?m) → MissionCommander(?x)
        
        # Add the classes as attributes to the ontology namespace
        onto.Astronaut = Astronaut_cls
        onto.MissionCommander = MissionCommander_cls
        onto.isCommanderOf = isCommanderOf_prop
            
        rule = Imp()
        rule.set_as_rule(
            "Astronaut(?x), isCommanderOf(?x, ?m) -> MissionCommander(?x)",
            namespaces=[onto],
        )
        logger.info(
            "SWRL rule added: Astronaut(?x) ∧ isCommanderOf(?x, ?m) → MissionCommander(?x)"
        )

        # Add test individual
        neil = Astronaut_cls("NeilArmstrong_test", namespace=onto)
        apollo11 = Thing("Apollo11_mission", namespace=onto)
        neil.isCommanderOf = [apollo11]

    # Run reasoner
    logger.info("Running Pellet reasoner …")
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True)
    except Exception as exc:
        logger.warning("Pellet failed: %s - showing manual check", exc)
        print("\n[Manual check] isCommanderOf associations:")
        print(f"  NeilArmstrong_test isCommanderOf Apollo11_mission")
        print(f"  → Would be inferred as MissionCommander by SWRL rule")
        return

    commanders = list(MissionCommander_cls.instances())
    print(f"\n=== SWRL Inference: MissionCommander ===")
    print(f"Rule: Astronaut(?x) ∧ isCommanderOf(?x, ?m) → MissionCommander(?x)")
    print(f"Inferred MissionCommander instances ({len(commanders)}):")
    for c in commanders:
        print(f"  {c.name}")
    print("=========================================\n")


def _run_demo_mode() -> None:
    """
    Pure-Python demo that shows the SWRL rule logic without OWL reasoning.
    Used as a fallback when the ontology hasn't been built yet.
    """
    print("\n=== SWRL Rule Demo (logic only, no reasoner) ===")
    print("Rule: Astronaut(?x) ∧ isCommanderOf(?x, ?m) → MissionCommander(?x)\n")

    # Simulated knowledge base
    astronauts = {"Neil Armstrong", "Pete Conrad", "Alan Shepard", "John Young"}
    commanders_of = {
        "Neil Armstrong": "Apollo 11",
        "Pete Conrad":    "Apollo 12",
        "Alan Shepard":   "Apollo 14",
    }
    # non-commanders
    crew_only = {"Buzz Aldrin", "Michael Collins"}

    print("Input facts:")
    for astro in astronauts:
        print(f"  Astronaut({astro})")
    for astro, mission in commanders_of.items():
        print(f"  isCommanderOf({astro}, {mission})")

    print("\nApplying rule …")
    inferred = [
        astro for astro in astronauts
        if astro in commanders_of
    ]
    print("\nInferred MissionCommander:")
    for astro in inferred:
        print(f"  MissionCommander({astro})  [commands → {commanders_of[astro]}]")
    print("================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SWRL rule on project KB")
    parser.add_argument("--ontology", default="kg_artifacts/ontology.ttl")
    parser.add_argument("--graph",    default="kg_artifacts/initial_graph.ttl")
    args = parser.parse_args()
    run_swrl_kb(args.ontology, args.graph)


if __name__ == "__main__":
    main()