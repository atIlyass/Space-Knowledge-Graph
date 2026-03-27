"""
src/rag/baseline.py
Baseline: ask the LLM a question directly, without any RAG or SPARQL.

Used to compare against the RAG pipeline.

Usage:
    from src.rag.baseline import ask_baseline
    answer = ask_baseline("Who are the astronauts of Apollo 11?")

    python -m src.rag.baseline --question "Who are the astronauts of Apollo 11?"
"""

import argparse
import requests

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma2:2b"

BASELINE_TEMPLATE = """\
You are a knowledgeable assistant. Answer the following question concisely and accurately,
based on your general knowledge. Do not generate SPARQL or any code.

Question: {question}

Answer:
"""


def ask_baseline(
    question: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
) -> str:
    """
    Query Ollama with the question alone (no schema, no SPARQL).
    Returns the model's text answer.
    """
    prompt = BASELINE_TEMPLATE.format(question=question)
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"(Ollama error: {exc})"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM baseline (no RAG)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    args = parser.parse_args()

    answer = ask_baseline(args.question, model=args.model)
    print(f"\n[Baseline] {answer}")


if __name__ == "__main__":
    main()
