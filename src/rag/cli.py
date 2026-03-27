"""
src/rag/cli.py
Interactive CLI demo for the RAG system.

Presents a prompt loop where the user types questions and receives:
  - A grounded answer retrieved from the RDF knowledge graph via SPARQL
  - The generated SPARQL query
  - Optionally the baseline (direct LLM) answer

Commands:
  q / quit / exit  → quit
  baseline on/off  → toggle baseline comparison mode
  schema           → print schema summary

Usage:
    python -m src.rag.cli
    python -m src.rag.cli --graph kg_artifacts/expanded.nt --model gemma2:2b
"""

import argparse
import logging
import sys

from rdflib import Graph

from src.rag.schema_summary import build_schema_summary
from src.rag.self_repair import rag_answer
from src.rag.baseline import ask_baseline

logging.basicConfig(level=logging.WARNING)  # quiet during interactive session


def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  Space Knowledge Graph - RAG CLI Demo")
    print("  Type a question, or 'help' for commands")
    print("=" * 60 + "\n")


def print_help() -> None:
    print("""
Commands:
  <question>    → Run RAG pipeline and show answer + SPARQL
  baseline      → Toggle baseline (direct LLM) comparison on/off
  schema        → Print schema summary
  help          → Show this help
  q / quit      → Exit
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG CLI demo")
    parser.add_argument("--graph", default="kg_artifacts/expanded.nt",
                        help="Path to expanded NT or TTL graph")
    parser.add_argument("--model", default="gemma2:2b",
                        help="Ollama model name")
    args = parser.parse_args()

    print_banner()
    print(f"Loading knowledge graph from: {args.graph}")
    try:
        g = Graph()
        fmt = "nt" if args.graph.endswith(".nt") else "turtle"
        g.parse(args.graph, format=fmt)
        print(f"Graph loaded: {len(g):,} triples\n")
    except Exception as exc:
        print(f"Error loading graph: {exc}")
        print("Please run the full pipeline first (see README).")
        sys.exit(1)

    print("Building schema summary …")
    schema = build_schema_summary(args.graph)

    show_baseline = False

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("q", "quit", "exit"):
            print("Bye!")
            break

        elif low == "help":
            print_help()

        elif low == "schema":
            print("\n" + schema + "\n")

        elif low == "baseline":
            show_baseline = not show_baseline
            print(f"Baseline mode: {'ON' if show_baseline else 'OFF'}\n")

        else:
            # Run RAG
            print(f"\n[Thinking …]")
            answer, sparql, retries = rag_answer(user_input, schema, g, model=args.model)

            print(f"\n{'─'*60}")
            print(f"[RAG Answer]")
            print(answer)
            print(f"\n[SPARQL used]")
            print(sparql)
            if retries:
                print(f"(self-repair retries: {retries})")

            if show_baseline:
                bl = ask_baseline(user_input, model=args.model)
                print(f"\n[Baseline (no RAG)]")
                print(bl)

            print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
