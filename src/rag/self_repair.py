"""
src/rag/self_repair.py
Self-repair loop for NL→SPARQL RAG.

If SPARQL execution fails or returns empty results, this module:
  1. Captures the error / empty-result signal
  2. Sends error + original question + schema back to Ollama
  3. Asks for a corrected SPARQL query
  4. Retries up to MAX_RETRIES times

Usage:
    from src.rag.self_repair import rag_answer
    answer, sparql_used, n_retries = rag_answer(question, schema, graph)
"""

import logging
from rdflib import Graph

from src.rag.sparql_generator import generate_sparql, call_ollama, extract_sparql

logger = logging.getLogger(__name__)

MAX_RETRIES     = 3
DEFAULT_MODEL   = "gemma2:2b"

REPAIR_TEMPLATE = """\
You are a SPARQL expert.
The following SPARQL query was generated for the question below, but it failed or returned no results.

Question: {question}

Attempted SPARQL:
```sparql
{sparql}
```

Error / issue: {error}

Schema summary:
{schema}

Please write a corrected SPARQL SELECT query. Rules:
1. Only use prefixes and predicates listed in the schema.
2. Return only the raw SPARQL between ```sparql and ``` tags.
3. Keep the query simple.
4. CRITICAL: Every triple MUST have SUBJECT PREDICATE OBJECT - never just subject and predicate.
5. Use FILTER(CONTAINS(LCASE(STR(?x)), "keyword")) to match entities. Apply the filter to the entity you are searching for - NOT the result variable.

VERIFIED WORKING EXAMPLES:

User question: Who were the astronauts of the Apollo 11 mission?
SPARQL query:
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?astronaut WHERE {{
  ?mission skg-o:hasAstronaut ?astronaut .
  FILTER(CONTAINS(LCASE(STR(?mission)), "apollo"))
}}
```

User question: What is the Hubble Space Telescope part of?
SPARQL query:
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?group WHERE {{
  ?telescope skg-o:partOf ?group .
  FILTER(CONTAINS(LCASE(STR(?telescope)), "hubble"))
}}
```
"""


def execute_sparql(sparql: str, g: Graph) -> list[dict]:
    """
    Execute a SPARQL SELECT query on graph g.
    Returns list of result row dicts, or raises on parse error.
    """
    results = g.query(sparql)
    rows = []
    for row in results:
        rows.append({str(var): str(val) for var, val in zip(results.vars, row)})
    return rows


def format_results(rows: list[dict]) -> str:
    """Convert query results to a readable answer string."""
    if not rows:
        return "(No results found)"
    lines = []
    for row in rows[:10]:  # cap at 10 rows
        lines.append("  " + " | ".join(f"{k}: {v}" for k, v in row.items()))
    if len(rows) > 10:
        lines.append(f"  … and {len(rows) - 10} more results")
    return "\n".join(lines)


def rag_answer(
    question: str,
    schema: str,
    g: Graph,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str, int]:
    """
    Full RAG pipeline with self-repair loop.

    Returns
    -------
    (answer_text, final_sparql_used, number_of_retries)
    """
    sparql = generate_sparql(question, schema, model=model)
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            rows = execute_sparql(sparql, g)
            if rows:
                return format_results(rows), sparql, attempt
            # Empty result - try repair
            last_error = "The query executed successfully but returned 0 results."
        except Exception as exc:
            last_error = str(exc)[:500]
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, last_error)

        if attempt < MAX_RETRIES:
            # Ask model to repair
            repair_prompt = REPAIR_TEMPLATE.format(
                question=question,
                sparql=sparql,
                error=last_error,
                schema=schema,
            )
            raw = call_ollama(repair_prompt, model=model)
            sparql = extract_sparql(raw)
            logger.info("Repaired SPARQL (attempt %d):\n%s", attempt + 1, sparql[:200])

    return f"(Could not retrieve an answer after {MAX_RETRIES} retries. Last error: {last_error})", sparql, MAX_RETRIES


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("self_repair.py loaded OK - use rag_answer() from your pipeline scripts.")
