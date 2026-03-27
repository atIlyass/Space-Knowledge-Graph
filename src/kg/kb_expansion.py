"""
src/kg/kb_expansion.py
Expands the KB by fetching 1-hop Wikidata triples for aligned entities.

Strategy
--------
1. Parse alignment.ttl; collect entities with owl:sameAs (high confidence).
2. For each Wikidata QID, run a SPARQL query on the Wikidata endpoint to
   fetch 1-hop outgoing triples filtered to a predicate whitelist.
3. Translate Wikidata URIs back to local URIs where possible, otherwise keep
   Wikidata URIs.
4. Deduplicate, clean, and write NT format to kg_artifacts/expanded.nt.

Predicate whitelist
-------------------
Maps Wikidata property PIDs → local predicate URIs where applicable.
Only properties that align with our ontology are kept.

Usage:
    python -m src.kg.kb_expansion
    python -m src.kg.kb_expansion --alignment kg_artifacts/alignment.ttl \
                                   --initial   kg_artifacts/initial_graph.ttl \
                                   --output    kg_artifacts/expanded.nt
"""

import argparse
import logging
import time
from pathlib import Path

import requests
from rdflib import Graph, Namespace, OWL, RDFS, RDF, URIRef, Literal
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SKG_R = Namespace("http://space-kg.org/resource/")
SKG_O = Namespace("http://space-kg.org/ontology/")
WD    = Namespace("http://www.wikidata.org/entity/")
WDT   = Namespace("http://www.wikidata.org/prop/direct/")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
REQUEST_DELAY   = 1.2   # seconds between SPARQL requests
MAX_HOPS        = 200   # triples per entity (avoid unbounded expansion)

# Predicate whitelist: Wikidata PID → (local_pred_name, include)
# Only these Wikidata properties will be retained
PREDICATE_WHITELIST: dict[str, str | None] = {
    "P1029": "hasAstronaut",       # crew member
    "P137":  "launchedBy",         # operator
    "P61":   "discoveredBy",       # discoverer
    "P361":  "partOf",             # part of
    "P131":  "locatedIn",          # located in
    "P18":   None,                 # image skip literals
    "P571":  "foundedDate",        # inception date
    "P576":  "dissolvedDate",      # dissolved
    "P17":   "country",            # country
    "P131":  "locatedIn",
    "P277":  None,                 # programming language skip
    "P856":  None,                 # official website skip
    "P569":  "birthDate",          # date of birth
    "P570":  "deathDate",          # date of death
    "P106":  "occupation",         # occupation
    "P19":   "birthPlace",         # place of birth
    "P27":   "citizenship",        # country of citizenship
    "P21":   "gender",             # sex/gender
    "P22":   "father",             # father
    "P25":   "mother",             # mother
    "P159":  "headquarters",       # headquarters location
    "P355":  "subsidiary",         # has subsidiary
    "P452":  "industry",           # industry
    "P31":   "instanceOf",         # instance of
    "P279":  "subclassOf",         # subclass of
}


def get_aligned_entities(alignment_path: str) -> dict[URIRef, str]:
    """Return {local_uri: wikidata_qid} for owl:sameAs aligned entities."""
    g = Graph()
    g.parse(alignment_path, format="turtle")
    aligned: dict[URIRef, str] = {}
    for s, _, o in g.triples((None, OWL.sameAs, None)):
        wd_str = str(o)
        if "wikidata.org/entity/Q" in wd_str:
            qid = wd_str.split("/")[-1]
            aligned[s] = qid
    logger.info("Found %d sameAs-aligned entities", len(aligned))
    return aligned


def fetch_1hop_wikidata(qid: str) -> list[tuple[str, str, str]]:
    """
    Fetch 1-hop triples from Wikidata for a QID.
    Returns list of (subject_qid, property_pid, object_qid_or_literal).
    """
    query = f"""
    SELECT ?prop ?propLabel ?obj ?objLabel WHERE {{
      wd:{qid} ?prop ?obj .
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
      FILTER(!isLiteral(?obj) || lang(?obj) = "en" || lang(?obj) = "")
    }}
    LIMIT {MAX_HOPS}
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "SpaceKGBot/1.0 (academic research)",
    }
    try:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        bindings = resp.json()["results"]["bindings"]
    except Exception as exc:
        logger.warning("SPARQL error for %s: %s", qid, exc)
        return []

    triples = []
    for b in bindings:
        prop_uri = b["prop"]["value"]
        pid = prop_uri.split("/")[-1]
        if pid not in PREDICATE_WHITELIST:
            continue
        pred_name = PREDICATE_WHITELIST[pid]
        if pred_name is None:
            continue
        obj_val = b["obj"]["value"]
        obj_type = b["obj"]["type"]
        triples.append((qid, pred_name, obj_val, obj_type))

    return triples


def build_expanded_graph(
    alignment_path: str,
    initial_path:   str,
    out_path:       str,
) -> Graph:
    """
    Main expansion routine. Returns the merged expanded graph.
    """
    # Load initial graph
    g = Graph()
    g.parse(initial_path, format="turtle")
    initial_size = len(g)
    logger.info("Initial graph: %d triples", initial_size)

    # Get aligned entities
    aligned = get_aligned_entities(alignment_path)

    # Reverse map: wikidata QID → local URI
    qid_to_local: dict[str, URIRef] = {v: k for k, v in aligned.items()}

    added = 0
    for local_uri, qid in tqdm(aligned.items(), desc="Expanding"):
        raw_triples = fetch_1hop_wikidata(qid)
        for _, pred_name, obj_val, obj_type in raw_triples:
            pred_uri = SKG_O[pred_name]

            # Object: try to map QID to local URI, otherwise keep WD URI / Literal
            if obj_type == "uri" and "wikidata.org/entity/Q" in obj_val:
                obj_qid = obj_val.split("/")[-1]
                obj_node = qid_to_local.get(obj_qid, WD[obj_qid])
            elif obj_type == "literal":
                obj_node = Literal(obj_val)
            else:
                obj_node = URIRef(obj_val)

            g.add((local_uri, pred_uri, obj_node))
            added += 1

        time.sleep(REQUEST_DELAY)

    # Deduplication happens automatically in rdflib Graph (set semantics)
    logger.info(
        "Expansion done. Added ~%d triples. Total: %d (was %d)",
        added, len(g), initial_size
    )

    # Write NT
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_file), format="nt")
    logger.info("Saved expanded KB → %s", out_file)
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="KB expansion via Wikidata 1-hop SPARQL")
    parser.add_argument("--alignment", default="kg_artifacts/alignment.ttl")
    parser.add_argument("--initial",   default="kg_artifacts/initial_graph.ttl")
    parser.add_argument("--output",    default="kg_artifacts/expanded.nt")
    args = parser.parse_args()
    build_expanded_graph(args.alignment, args.initial, args.output)


if __name__ == "__main__":
    main()
