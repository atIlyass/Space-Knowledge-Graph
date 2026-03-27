"""
src/ie/run_ie.py
Main IE pipeline runner.

Reads data/processed/crawler_output.jsonl
→ runs spaCy NER + dependency-based relation extraction
→ saves data/processed/extracted_knowledge.csv

Usage:
    python -m src.ie.run_ie
    python -m src.ie.run_ie --input data/processed/crawler_output.jsonl \
                             --output data/processed/extracted_knowledge.csv
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ie.ner import extract_entities
from src.ie.relation_extractor import extract_relations

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_jsonl(jsonl_path: str, csv_path: str) -> pd.DataFrame:
    """
    Read crawler JSONL, run NER + RE on each page, save combined CSV.

    CSV columns:
        subject, subject_label, predicate, object, object_label,
        source_url, sentence
    """
    records_re: list[dict] = []
    records_ner: list[dict] = []

    jsonl_file = Path(jsonl_path)
    pages = [
        json.loads(line)
        for line in jsonl_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    logger.info("Processing %d pages …", len(pages))

    for page in tqdm(pages, desc="IE"):
        url = page["url"]
        text = page["text"]

        # ── Relation extraction ───────────────────────────────────────────
        for triple in extract_relations(text):
            records_re.append(
                {
                    "subject": triple.subject,
                    "subject_label": triple.subject_label,
                    "predicate": triple.predicate,
                    "object": triple.obj,
                    "object_label": triple.object_label,
                    "source_url": url,
                    "sentence": triple.sentence[:300],  # truncate for CSV
                    "extraction_type": "relation",
                }
            )

        # ── NER-only records (entities without a paired object) ───────────
        for ent in extract_entities(text):
            records_ner.append(
                {
                    "subject": ent.text,
                    "subject_label": ent.label,
                    "predicate": "mentionedIn",
                    "object": page.get("title", url),
                    "object_label": "Article",
                    "source_url": url,
                    "sentence": ent.sentence[:300],
                    "extraction_type": "ner",
                }
            )

    df = pd.DataFrame(records_re + records_ner)

    # Normalise text
    for col in ("subject", "predicate", "object"):
        df[col] = df[col].str.strip()

    # Drop obvious duplicates
    df = df.drop_duplicates(subset=["subject", "predicate", "object"])

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(
        "Saved %d triples (%d relation, %d NER) → %s",
        len(df),
        len(records_re),
        len(records_ner),
        out_path,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="IE pipeline: JSONL → CSV")
    parser.add_argument(
        "--input",
        default="data/processed/crawler_output.jsonl",
        help="Input crawler JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/extracted_knowledge.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()
    process_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
