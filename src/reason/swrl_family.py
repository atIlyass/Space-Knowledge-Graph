"""
src/reason/swrl_family.py
Demonstrates OWLReady2 SWRL reasoning on family.owl.

SWRL Rule implemented:
    Person(?x) ∧ hasAge(?x, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?x)

Prerequisites
-------------
- owlready2 installed
- data/family.owl present (standard family ontology)
- Java installed + JAVA_HOME set (for HermiT reasoner via owlready2)

Usage:
    python -m src.reason.swrl_family
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_family_swrl(owl_path: str = "data/family.owl") -> None:
    """
    Load family.owl, add the SWRL 'oldPerson' rule, run HermiT,
    and print all inferred oldPerson individuals.
    """
    try:
        from owlready2 import get_ontology, sync_reasoner_pellet
        import owlready2
    except ImportError as e:
        logger.error("owlready2 import failed: %s. Run: pip install owlready2", e)
        return

    owl_file = Path(owl_path).resolve()
    if not owl_file.exists():
        logger.error(
            "family.owl not found at %s. "
            "Please place family.owl in data/. See README for download instructions.",
            owl_file,
        )
        return

    onto = get_ontology(f"file://{owl_file.as_posix()}").load()

    logger.info("Loaded ontology: %s (%d axioms)", owl_file.name, len(list(onto.classes())))

    # ── Check required vocabulary ─────────────────────────────────────────────
    with onto:
        from owlready2 import Thing, DataProperty, ObjectProperty, AllDisjoint
        from owlready2 import Imp  # for SWRL rules

        # Ensure classes/properties exist (family.owl may already define them)
        if not onto.search_one(iri="*Person"):
            logger.warning("No Person class found; using Thing as fallback")

        person_cls = onto.search_one(iri="*#Person") or onto.search_one(iri="*Person")
        old_person_cls = onto.search_one(iri="*#oldPerson")

        if old_person_cls is None:
            # Create oldPerson class if absent
            with onto:
                class oldPerson(Thing):
                    namespace = onto
            old_person_cls = onto.oldPerson
            logger.info("Created class: oldPerson")

        has_age_prop = onto.search_one(iri="*#hasAge") or onto.search_one(iri="*hasAge")
        if has_age_prop is None:
            with onto:
                class hasAge(DataProperty):
                    namespace = onto
                    domain = [person_cls] if person_cls else []
                    range = [int]
            has_age_prop = onto.hasAge
            logger.info("Created data property: hasAge")

        # ── Add SWRL Rule ─────────────────────────────────────────────────────
        # Person(?x) ∧ hasAge(?x, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?x)
        # In owlready2, mathematical built-ins map to Python operators or require the swrlb namespace.
        rule = Imp()
        rule.set_as_rule(
            """
            Person(?x),
            hasAge(?x, ?a),
            greaterThan(?a, 60)
            ->
            oldPerson(?x)
            """,
            namespaces=[onto],
        )
        logger.info("SWRL rule added: Person(?x) ∧ hasAge(?x, ?a) ∧ greaterThan(?a,60) → oldPerson(?x)")

        # ── Add test individuals ──────────────────────────────────────────────
        class Person(Thing):
            namespace = onto

        alice = Person("Alice", namespace=onto)
        alice.hasAge = [72]

        bob = Person("Bob", namespace=onto)
        bob.hasAge = [45]

        carol = Person("Carol", namespace=onto)
        carol.hasAge = [63]

    # ── Run reasoner ─────────────────────────────────────────────────────────
    logger.info("Running Pellet reasoner …")
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    except Exception as exc:
        logger.warning(
            "Pellet reasoning failed (%s). "
            "Trying HermiT …", exc
        )
        try:
            from owlready2 import sync_reasoner
            with onto:
                sync_reasoner()
        except Exception as exc2:
            logger.error("HermiT also failed: %s", exc2)
            logger.info("Printing individuals added manually (no inference).")
            print("\n[Manual check] hasAge values:")
            for ind in onto.individuals():
                age = getattr(ind, "hasAge", None)
                print(f"  {ind.name}: age={age}")
            return

    # ── Print results ─────────────────────────────────────────────────────────
    old_persons = list(old_person_cls.instances())
    print(f"\n=== SWRL Inference: oldPerson ===")
    print(f"Rule: Person(?x) ∧ hasAge(?x,?a) ∧ greaterThan(?a,60) → oldPerson(?x)")
    print(f"Inferred oldPerson individuals ({len(old_persons)}):")
    for p in old_persons:
        age = getattr(p, "hasAge", ["?"])
        print(f"  {p.name}  (age: {age})")
    print("=================================\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SWRL reasoning on family.owl")
    parser.add_argument("--owl", default="data/family.owl",
                        help="Path to family.owl")
    args = parser.parse_args()
    run_family_swrl(args.owl)


if __name__ == "__main__":
    main()
