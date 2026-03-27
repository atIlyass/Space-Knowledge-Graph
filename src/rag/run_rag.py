"""
src/rag/run_rag.py
Non-interactive pipeline runner for the RAG system.

Accepts one question, runs the full RAG pipeline, prints result.
Useful for batch processing or testing from scripts.

Usage:
    python -m src.rag.run_rag --question "Who were the astronauts of Apollo 11?"
"""

import argparse
import logging

from rdflib import Graph

from src.rag.schema_summary import build_schema_summary
from src.rag.self_repair import rag_answer
from src.rag.baseline import ask_baseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    question: str,
    graph_path: str,
    model: str,
    compare_baseline: bool = False,
) -> None:
    """Load graph, run RAG, optionally compare with baseline."""
    logger.info("Loading knowledge graph …")
    g = Graph()
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g.parse(graph_path, format=fmt)

    schema = build_schema_summary(graph_path)

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    answer, sparql, retries = rag_answer(question, schema, g, model=model)

    print(f"\n[RAG Answer]\n{answer}")
    print(f"\n[Generated SPARQL]\n{sparql}")
    if retries:
        print(f"(self-repair retries: {retries})")

    if compare_baseline:
        bl = ask_baseline(question, model=model)
        print(f"\n[Baseline (no RAG)]\n{bl}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline runner (non-interactive)")
    parser.add_argument("--question", required=True, help="Natural-language question")
    parser.add_argument("--graph",    default="kg_artifacts/expanded.nt")
    parser.add_argument("--model",    default="gemma2:2b")
    parser.add_argument("--baseline", action="store_true",
                        help="Also show baseline (direct LLM) answer for comparison")
    args = parser.parse_args()

    run_pipeline(args.question, args.graph, args.model, args.baseline)


if __name__ == "__main__":
    main()
