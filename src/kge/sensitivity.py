"""
src/kge/sensitivity.py
KB size sensitivity experiments: 20k / 50k / full triple subsets.

For each subset:
  1. Sample N triples from train.txt (keeping valid/test fixed)
  2. Train TransE + ComplEx for a reduced number of epochs (50)
  3. Evaluate and record MRR + Hits@10

Saves comparison table to reports/sensitivity_results.json.

Usage:
    python -m src.kge.sensitivity
    python -m src.kge.sensitivity --sizes 20000 50000 full --epochs 50
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
REPORTS_DIR = Path("reports")


def sample_triples(train_path: str, n: int) -> list[str]:
    """Return n randomly sampled lines from train_path."""
    lines = Path(train_path).read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    random.seed(SEED)
    if n >= len(lines):
        return lines
    return random.sample(lines, n)


def run_experiment(
    model_name: str,
    train_lines: list[str],
    valid_path: str,
    test_path: str,
    dim: int,
    lr: float,
    batch: int,
    epochs: int,
    subset_label: str,
) -> dict:
    """Train model on given train_lines, return metric dict."""
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.error("PyKEEN not installed.")
        return {}

    # Write temp train file
    tmp_train = Path("data/processed/_tmp_sensitivity_train.txt")
    tmp_train.write_text("\n".join(train_lines), encoding="utf-8")

    training   = TriplesFactory.from_path(str(tmp_train))
    validation = TriplesFactory.from_path(valid_path,
                                           entity_to_id=training.entity_to_id,
                                           relation_to_id=training.relation_to_id)
    testing    = TriplesFactory.from_path(test_path,
                                           entity_to_id=training.entity_to_id,
                                           relation_to_id=training.relation_to_id)

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs={"embedding_dim": dim},
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        training_kwargs={"num_epochs": epochs, "batch_size": batch},
        negative_sampler="basic",
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        random_seed=SEED,
        use_tqdm=False,
    )

    tmp_train.unlink(missing_ok=True)

    m = result.metric_results.to_flat_dict()
    return {
        "model":   model_name,
        "subset":  subset_label,
        "n_train": len(train_lines),
        "mrr":     m.get("both.realistic.inverse_harmonic_mean_rank", None),
        "hits_10": m.get("both.realistic.hits_at_10", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KB size sensitivity experiments")
    parser.add_argument("--train",  default="data/processed/train.txt")
    parser.add_argument("--valid",  default="data/processed/valid.txt")
    parser.add_argument("--test",   default="data/processed/test.txt")
    parser.add_argument("--sizes",  nargs="+", default=["20000", "50000", "full"],
                        help="Subset sizes (integers or 'full')")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for each experiment (default 50)")
    parser.add_argument("--dim",    type=int, default=128)
    parser.add_argument("--lr",     type=float, default=0.01)
    parser.add_argument("--batch",  type=int, default=512)
    parser.add_argument("--out",    default="reports/sensitivity_results.json")
    args = parser.parse_args()

    all_train = Path(args.train).read_text(encoding="utf-8").splitlines()
    all_train = [l for l in all_train if l.strip()]
    full_size = len(all_train)

    results = []
    for size_str in args.sizes:
        n = full_size if size_str == "full" else int(size_str)
        label = size_str if size_str == "full" else f"{n:,}"
        train_lines = sample_triples(args.train, n)
        logger.info("Subset %s: %d triples", label, len(train_lines))

        for model in ("TransE", "ComplEx"):
            logger.info("  Training %s …", model)
            r = run_experiment(
                model_name=model,
                train_lines=train_lines,
                valid_path=args.valid,
                test_path=args.test,
                dim=args.dim,
                lr=args.lr,
                batch=args.batch,
                epochs=args.epochs,
                subset_label=label,
            )
            if r:
                results.append(r)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  KB Size Sensitivity Results (epochs={args.epochs})")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Subset':>10} {'N_train':>10} {'MRR':>8} {'H@10':>8}")
    print("-" * 70)
    for r in results:
        mrr = f"{r['mrr']:.4f}" if isinstance(r.get('mrr'), float) else "n/a"
        h10 = f"{r['hits_10']:.4f}" if isinstance(r.get('hits_10'), float) else "n/a"
        print(f"{r['model']:<12} {r['subset']:>10} {r['n_train']:>10,} {mrr:>8} {h10:>8}")
    print(f"{'='*70}\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved → {args.out}")


if __name__ == "__main__":
    main()
