"""
src/rag/sparql_generator.py
Generate SPARQL queries from natural-language questions using Ollama.

Pipeline:
  1. Build a schema-aware prompt (schema + question)
  2. Call Ollama HTTP API  (/api/generate)
  3. Extract the SPARQL block from the response

Model: gemma2:2b (configurable via --model)
Endpoint: http://localhost:11434 (Ollama default)

Usage:
    from src.rag.sparql_generator import generate_sparql
    sparql = generate_sparql(question, schema_summary)

    python -m src.rag.sparql_generator --question "Who were the astronauts of Apollo 11?"
"""

import argparse
import re
import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma2:2b"

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert in SPARQL and RDF knowledge graphs.
Below is the schema of a space-domain knowledge graph.

{schema}

Your task: Convert the user's question into a valid SPARQL SELECT query.

Rules:
1. Only use prefixes and predicates listed in the schema above.
2. Use PREFIX declarations at the top of your query.
3. Return ONLY the raw SPARQL query between ```sparql and ``` tags.
4. CRITICAL: Every triple MUST have SUBJECT PREDICATE OBJECT - never just subject and predicate.
5. Use FILTER(CONTAINS(LCASE(STR(?x)), "keyword")) to match entity names by string.
6. The FILTER applies to the entity variable you are searching for - NOT to the result variable.

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

User question: Which agency launched the James Webb Space Telescope?
SPARQL query:
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?agency WHERE {{
  ?telescope skg-o:launchedBy ?agency .
  FILTER(CONTAINS(LCASE(STR(?telescope)), "webb"))
}}
```

User question: {question}

SPARQL query:
"""


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 60) -> str:
    """
    Call the Ollama HTTP API and return the full response text.
    Raises requests.RequestException on network errors.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


def extract_sparql(text: str) -> str:
    """
    Extract SPARQL from a markdown code fence if present,
    otherwise return the full text stripped.
    """
    # Try ```sparql ... ```
    res = ""
    m = re.search(r"```(?:sparql)?\s*(SELECT.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        res = m.group(1).strip()
    else:
        # Fallback: first SELECT … block
        m2 = re.search(r"(SELECT\s+.*)", text, re.DOTALL | re.IGNORECASE)
        if m2:
            res = m2.group(1).strip()
        else:
            res = text.strip()
            
    # Clean trailing LLM hallucinations and markdown by breaking at the last '}'
    if "}" in res:
        res = res[:res.rfind("}") + 1]

    # Clean stray backticks at the end just in case
    res = res.rstrip("`").strip()
            
    # Auto-inject prefixes if not present
    if "PREFIX" not in res.upper():
        prefixes = (
            "PREFIX skg-o: <http://space-kg.org/ontology/>\n"
            "PREFIX skg-r: <http://space-kg.org/resource/>\n"
        )
        res = prefixes + res
        
    return res


def generate_sparql(
    question: str,
    schema: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a SPARQL query for the given question using the provided schema.

    Returns the extracted SPARQL string.
    Raises an exception if Ollama is unavailable.
    """
    prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
    raw = call_ollama(prompt, model=model)
    sparql = extract_sparql(raw)
    return sparql


def main() -> None:
    from src.rag.schema_summary import build_schema_summary

    parser = argparse.ArgumentParser(description="NL → SPARQL via Ollama")
    parser.add_argument("--question", required=True, help="Natural-language question")
    parser.add_argument("--graph",    default="kg_artifacts/expanded.nt")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    args = parser.parse_args()

    schema = build_schema_summary(args.graph)
    sparql = generate_sparql(args.question, schema, model=args.model)
    print(f"\nGenerated SPARQL:\n{sparql}")


if __name__ == "__main__":
    main()
