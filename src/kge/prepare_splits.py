"""
src/kge/prepare_splits.py
Clean the expanded KB and split into train/valid/test sets.

Split strategy (as per professor requirements)
----------------------------------------------
- Random 80/10/10 split on all triples
- Enforce: every entity and relation in valid/test also appears in train
- Remove orphan entities (appear only in valid/test) by moving them to train
- No leakage: valid and test sets do not overlap

Input  : kg_artifacts/expanded.nt  (or initial_graph.ttl as fallback)
Outputs: data/processed/train.txt / valid.txt / test.txt
         Format: <subject> <predicate> <object> (tab-separated, no angle brackets)

Usage:
    python -m src.kge.prepare_splits
    python -m src.kge.prepare_splits --input kg_artifacts/expanded.nt \
                                      --outdir data/processed
"""

import argparse
import logging
import random
from pathlib import Path

from rdflib import Graph, URIRef

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def load_uri_triples(graph_path: str) -> list[tuple[str, str, str]]:
    """
    Load an RDF graph and return only triples where all three terms are URIs.
    Literals are excluded (not suitable for KGE without special handling).
    """
    g = Graph()
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g.parse(graph_path, format=fmt)
    logger.info("Loaded graph: %d triples", len(g))

    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))

    # Deduplicate
    triples = list(set(triples))
    logger.info("URI-only triples after dedup: %d", len(triples))
    return triples


def enforce_coverage(
    train: list, valid: list, test: list
) -> tuple[list, list, list]:
    """
    Move triples from valid/test to train if any entity or predicate
    appears *only* in valid/test (orphan check).
    """
    def entity_set(triples):
        ents = set()
        for s, p, o in triples:
            ents.add(s)
            ents.add(o)
        return ents

    def pred_set(triples):
        return {p for _, p, _ in triples}

    train_ents  = entity_set(train)
    train_preds = pred_set(train)

    # Filter valid
    new_valid, rescued = [], []
    for t in valid:
        s, p, o = t
        if s in train_ents and o in train_ents and p in train_preds:
            new_valid.append(t)
        else:
            rescued.append(t)
    train = train + rescued

    # Refresh
    train_ents  = entity_set(train)
    train_preds = pred_set(train)

    # Filter test
    new_test, rescued2 = [], []
    for t in test:
        s, p, o = t
        if s in train_ents and o in train_ents and p in train_preds:
            new_test.append(t)
        else:
            rescued2.append(t)
    train = train + rescued2

    if rescued or rescued2:
        logger.info(
            "Coverage fix: moved %d triples from valid+test → train",
            len(rescued) + len(rescued2),
        )

    return train, new_valid, new_test


def split_triples(
    triples: list, train_ratio=0.8, valid_ratio=0.1, seed=SEED
) -> tuple[list, list, list]:
    """Random 80/10/10 split with orphan-coverage enforcement."""
    random.seed(seed)
    shuffled = triples[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test  = shuffled[n_train + n_valid :]

    # Enforce coverage
    train, valid, test = enforce_coverage(train, valid, test)
    return train, valid, test


def write_split(triples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s, p, o in triples:
            fh.write(f"{s}\t{p}\t{o}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KGE train/valid/test splits")
    parser.add_argument("--input",  default="kg_artifacts/expanded.nt",
                        help="Input NT or TTL file")
    parser.add_argument("--outdir", default="data/processed",
                        help="Output directory")
    args = parser.parse_args()

    triples = load_uri_triples(args.input)
    if len(triples) < 100:
        logger.warning(
            "Very few triples (%d). "
            "Run kb_expansion.py first to populate expanded.nt.", len(triples)
        )

    train, valid, test = split_triples(triples)

    out = Path(args.outdir)
    write_split(train, out / "train.txt")
    write_split(valid, out / "valid.txt")
    write_split(test,  out / "test.txt")

    print(f"\n=== Split summary ===")
    print(f"  Total  : {len(triples):,}")
    print(f"  Train  : {len(train):,}  ({len(train)/len(triples)*100:.1f}%)")
    print(f"  Valid  : {len(valid):,}  ({len(valid)/len(triples)*100:.1f}%)")
    print(f"  Test   : {len(test):,}  ({len(test)/len(triples)*100:.1f}%)")

    # Check no orphan entities in valid/test
    train_ents = {s for s, p, o in train} | {o for s, p, o in train}
    orphan_v = sum(1 for s, p, o in valid if s not in train_ents or o not in train_ents)
    orphan_t = sum(1 for s, p, o in test  if s not in train_ents or o not in train_ents)
    print(f"  Orphan entities in valid: {orphan_v}  (should be 0)")
    print(f"  Orphan entities in test : {orphan_t}  (should be 0)")
    print(f"====================\n")


if __name__ == "__main__":
    main()
