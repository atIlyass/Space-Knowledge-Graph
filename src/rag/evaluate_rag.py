"""
src/rag/evaluate_rag.py
Compare RAG (SPARQL-grounded) vs Baseline (direct LLM) on 5 preset questions.

Saves results to reports/rag_evaluation.md.

Usage:
    python -m src.rag.evaluate_rag
    python -m src.rag.evaluate_rag --graph kg_artifacts/expanded.nt --model gemma2:2b
"""

import argparse
import logging
from pathlib import Path

from rdflib import Graph

from src.rag.schema_summary import build_schema_summary
from src.rag.self_repair import rag_answer
from src.rag.baseline import ask_baseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 5 evaluation questions ────────────────────────────────────────────────────
# Questions are tuned to predicates verified to exist in the graph:
#   skg-o:hasAstronaut - connects missions to astronauts
#   skg-o:partOf       - connects telescopes/missions to programs
#   skg-o:launchedBy   - connects spacecraft to launch agencies
#   skg-o:instanceOf   - connects entities to their types (from Wikidata P31)
#   skg-o:country      - connects entities to their countries (from Wikidata P17)
EVAL_QUESTIONS = [
    "Who were the astronauts of the Apollo 11 mission?",
    "What is the Hubble Space Telescope part of?",
    "Which agency launched the Hubble Space Telescope?",
    "What is the Hubble Space Telescope an instance of?",
    "What countries is SpaceX associated with?",
]


def run_evaluation(graph_path: str, model: str) -> list[dict]:
    """Run all 5 questions and return list of result dicts."""
    logger.info("Loading graph …")
    g = Graph()
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g.parse(graph_path, format=fmt)

    schema = build_schema_summary(graph_path)
    results = []

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        logger.info("[%d/5] %s", i, q)

        # Baseline
        bl_answer = ask_baseline(q, model=model)

        # RAG
        rag_ans, sparql_used, n_retries = rag_answer(q, schema, g, model=model)

        results.append({
            "question":  q,
            "baseline":  bl_answer,
            "rag":       rag_ans,
            "sparql":    sparql_used,
            "retries":   n_retries,
        })

    return results


def format_report(results: list[dict]) -> str:
    """Format evaluation results as a Markdown report."""
    lines = [
        "# RAG Evaluation - Baseline vs SPARQL-grounded RAG",
        "",
        "| # | Question | Baseline Answer | RAG Answer | Retries |",
        "|---|----------|----------------|------------|---------|",
    ]
    for i, r in enumerate(results, 1):
        q   = r["question"]
        bl  = r["baseline"].replace("\n", " ").replace("|", "/")[:120]
        rag = r["rag"].replace("\n", " ").replace("|", "/")[:120]
        ret = r["retries"]
        lines.append(f"| {i} | {q} | {bl} | {rag} | {ret} |")

    lines += [
        "",
        "## Detailed Results",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"### Q{i}: {r['question']}",
            "",
            "**Baseline (no RAG):**",
            f"> {r['baseline']}",
            "",
            "**RAG answer (SPARQL-grounded):**",
            f"> {r['rag']}",
            "",
            "**Generated SPARQL:**",
            "```sparql",
            r["sparql"],
            "```",
            f"*(Repair retries: {r['retries']})*",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG vs Baseline")
    parser.add_argument("--graph", default="kg_artifacts/expanded.nt")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--out",   default="reports/rag_evaluation.md")
    args = parser.parse_args()

    results = run_evaluation(args.graph, args.model)
    report = format_report(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
