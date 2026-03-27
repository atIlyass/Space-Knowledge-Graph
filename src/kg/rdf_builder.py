"""
src/kg/rdf_builder.py
Converts extracted_knowledge.csv into an initial RDF graph (initial_graph.ttl).

Entity URIs  : http://space-kg.org/resource/<slugified_name>
Predicate URIs: http://space-kg.org/ontology/<predicate_name>
Class URIs    : http://space-kg.org/ontology/<EntityLabel>

Usage:
    python -m src.kg.rdf_builder
    python -m src.kg.rdf_builder --input data/processed/extracted_knowledge.csv \
                                  --output kg_artifacts/initial_graph.ttl
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SKG_R = Namespace("http://space-kg.org/resource/")
SKG_O = Namespace("http://space-kg.org/ontology/")
PROV  = Namespace("http://www.w3.org/ns/prov#")

# Map spaCy NER labels to ontology classes
LABEL_TO_CLASS = {
    "PERSON":  SKG_O["Person"],
    "ORG":     SKG_O["SpaceAgency"],   # most ORGs in this domain are agencies
    "GPE":     SKG_O["CelestialBody"], # most GPEs are planets / countries
    "LOC":     SKG_O["CelestialBody"],
    "DATE":    SKG_O["TimePoint"],
    "FAC":     SKG_O["Spacecraft"],
    "PRODUCT": SKG_O["Spacecraft"],
    "EVENT":   SKG_O["SpaceMission"],
    "Article": SKG_O["Document"],
}


def slugify(text: str) -> str:
    """Convert a string to a valid URI local name."""
    t = re.sub(r"[^\w\s-]", "", text.strip())
    t = re.sub(r"[\s-]+", "_", t)
    return t[:80]  # cap length


def build_initial_graph(csv_path: str, onto_path: str | None = None) -> Graph:
    """
    Build and return the initial RDF graph from extracted_knowledge.csv.
    Optionally imports the ontology.
    """
    g = Graph()
    g.bind("skg-r", SKG_R)
    g.bind("skg-o", SKG_O)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("prov", PROV)

    # Optionally import ontology
    if onto_path and Path(onto_path).exists():
        g.parse(onto_path, format="turtle")
        logger.info("Imported ontology from %s", onto_path)

    df = pd.read_csv(csv_path, keep_default_na=False)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    added = 0
    for _, row in df.iterrows():
        subj_slug  = slugify(str(row["subject"]))
        obj_slug   = slugify(str(row["object"]))
        pred_name  = slugify(str(row["predicate"]))
        subj_label = str(row.get("subject_label", ""))
        obj_label  = str(row.get("object_label", ""))
        source_url = str(row.get("source_url", ""))

        if not subj_slug or not obj_slug or not pred_name:
            continue

        subj_uri = SKG_R[subj_slug]
        pred_uri = SKG_O[pred_name]
        obj_uri  = SKG_R[obj_slug]

        # Triple
        g.add((subj_uri, pred_uri, obj_uri))

        # Labels
        g.add((subj_uri, RDFS.label, Literal(str(row["subject"]))))
        g.add((obj_uri,  RDFS.label, Literal(str(row["object"]))))

        # Type assertions from NER labels
        if subj_label in LABEL_TO_CLASS:
            g.add((subj_uri, RDF.type, LABEL_TO_CLASS[subj_label]))
        if obj_label in LABEL_TO_CLASS:
            g.add((obj_uri, RDF.type, LABEL_TO_CLASS[obj_label]))

        # Provenance
        if source_url:
            g.add((subj_uri, PROV.wasDerivedFrom, URIRef(source_url)))

        added += 1

    logger.info("Added %d triples to initial graph", added)
    return g


def save_graph(g: Graph, out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_file), format="turtle")
    logger.info("Saved initial graph → %s  (%d triples)", out_file, len(g))


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV → initial_graph.ttl")
    parser.add_argument("--input",  default="data/processed/extracted_knowledge.csv")
    parser.add_argument("--output", default="kg_artifacts/initial_graph.ttl")
    parser.add_argument("--ontology", default="kg_artifacts/ontology.ttl",
                        help="Ontology Turtle to import (optional)")
    args = parser.parse_args()
    g = build_initial_graph(args.input, args.ontology)
    save_graph(g, args.output)


if __name__ == "__main__":
    main()
