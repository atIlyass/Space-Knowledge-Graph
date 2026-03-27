"""
src/rag/schema_summary.py
Extract schema information from the expanded RDF graph.

Produces a concise text summary with:
  - Namespace prefixes
  - Classes (with instance counts)
  - Predicates (with occurrence counts)

This summary is injected into NL→SPARQL prompts.

Usage:
    from src.rag.schema_summary import build_schema_summary
    schema = build_schema_summary("kg_artifacts/expanded.nt")

    python -m src.rag.schema_summary
"""

import argparse
import json
from pathlib import Path

from rdflib import Graph, RDF, RDFS, OWL, Namespace, URIRef

SKG_O = "http://space-kg.org/ontology/"
SKG_R = "http://space-kg.org/resource/"
WD    = "http://www.wikidata.org/entity/"


def _short(uri: str) -> str:
    """Return compact form of a URI."""
    for prefix, ns in [("skg-o:", SKG_O), ("skg-r:", SKG_R), ("wd:", WD)]:
        if uri.startswith(ns):
            return prefix + uri[len(ns):]
    return f"<{uri}>"


def build_schema_summary(graph_path: str, top_n: int = 20) -> str:
    """
    Parse the RDF graph and return a compact schema string suitable for LLM prompts.

    Parameters
    ----------
    graph_path : path to NT or Turtle file
    top_n      : max number of predicates/classes to include

    Returns
    -------
    Multi-line string with prefix declarations, classes, and predicates.
    """
    g = Graph()
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g.parse(graph_path, format=fmt)

    # ── Prefixes ─────────────────────────────────────────────────────────────
    lines = ["## Schema Summary", ""]
    lines.append("### Prefixes")
    lines.append(f"  PREFIX skg-r: <{SKG_R}>")
    lines.append(f"  PREFIX skg-o: <{SKG_O}>")
    lines.append(f"  PREFIX wd:    <{WD}>")
    lines.append(f"  PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append(f"  PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>")
    lines.append("")

    # ── Classes and instance counts ──────────────────────────────────────────
    class_counts: dict[str, int] = {}
    for _, _, cls in g.triples((None, RDF.type, None)):
        key = _short(str(cls))
        class_counts[key] = class_counts.get(key, 0) + 1

    lines.append("### Classes (instance count)")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:top_n]:
        lines.append(f"  {cls:<45} {cnt:>6} instances")
    lines.append("")

    # ── Predicates and triple counts ─────────────────────────────────────────
    pred_counts: dict[str, int] = {}
    for _, p, _ in g:
        key = _short(str(p))
        pred_counts[key] = pred_counts.get(key, 0) + 1

    lines.append("### Predicates (occurrence count)")
    for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])[:top_n]:
        lines.append(f"  {pred:<45} {cnt:>6} triples")
    lines.append("")

    # ── Sample entities ───────────────────────────────────────────────────────
    sample_subjects: set[str] = set()
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef):
            sample_subjects.add(f"{_short(str(s))} rdfs:label \"{str(o)}\"")
        if len(sample_subjects) >= 10:
            break

    if sample_subjects:
        lines.append("### Sample entities")
        for ex in list(sample_subjects)[:10]:
            lines.append(f"  {ex}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RDF schema summary for LLM")
    parser.add_argument("--graph",   default="kg_artifacts/expanded.nt")
    parser.add_argument("--top-n",   type=int, default=20)
    parser.add_argument("--out",     default=None, help="Save to file (optional)")
    args = parser.parse_args()

    summary = build_schema_summary(args.graph, args.top_n)
    print(summary)

    if args.out:
        Path(args.out).write_text(summary, encoding="utf-8")
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
