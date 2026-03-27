"""
src/kg/kb_stats.py
Compute statistics for the expanded knowledge base and save kb_stats.json.

Usage:
    python -m src.kg.kb_stats
    python -m src.kg.kb_stats --input kg_artifacts/expanded.nt \
                               --output kg_artifacts/kb_stats.json
"""

import argparse
import json
import logging
from pathlib import Path

from rdflib import Graph, URIRef

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(graph_path: str) -> dict:
    """Load a graph and compute KB statistics."""
    logger.info("Loading graph from %s …", graph_path)
    g = Graph()

    # Auto-detect format by extension
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g.parse(graph_path, format=fmt)

    triples = list(g)
    num_triples = len(triples)

    entities: set[str] = set()
    relations: set[str] = set()

    for s, p, o in triples:
        if isinstance(s, URIRef):
            entities.add(str(s))
        if isinstance(p, URIRef):
            relations.add(str(p))
        if isinstance(o, URIRef):
            entities.add(str(o))

    # Subject types
    types_query = """
    SELECT DISTINCT ?type (COUNT(?s) AS ?count)
    WHERE { ?s a ?type }
    GROUP BY ?type
    ORDER BY DESC(?count)
    """
    type_counts = {}
    try:
        for row in g.query(types_query):
            type_counts[str(row.type)] = int(row["count"])
    except Exception:
        pass

    stats = {
        "num_triples":  num_triples,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "top_types": dict(list(type_counts.items())[:10]),
        "source_file": graph_path,
    }
    return stats


def save_stats(stats: dict, out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("KB stats saved → %s", out_file)


def print_stats(stats: dict) -> None:
    print("\n=== Knowledge Base Statistics ===")
    print(f"  Triples  : {stats['num_triples']:,}")
    print(f"  Entities : {stats['num_entities']:,}")
    print(f"  Relations: {stats['num_relations']:,}")
    if stats.get("top_types"):
        print("  Top types:")
        for t, c in list(stats["top_types"].items())[:5]:
            short = t.split("/")[-1]
            print(f"    {short:<30} {c:>6}")
    print("=================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="KB statistics")
    parser.add_argument("--input",  default="kg_artifacts/expanded.nt")
    parser.add_argument("--output", default="kg_artifacts/kb_stats.json")
    args = parser.parse_args()
    stats = compute_stats(args.input)
    print_stats(stats)
    save_stats(stats, args.output)


if __name__ == "__main__":
    main()
