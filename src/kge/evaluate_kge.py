"""
src/kge/evaluate_kge.py
Load trained KGE models and evaluate with filtered MRR, Hits@1/3/10.

Loads results saved by train_kge.py from kg_artifacts/kge/<model>/
Reports both head and tail prediction metrics.
Saves combined table to reports/kge_metrics.json.

Usage:
    python -m src.kge.evaluate_kge
    python -m src.kge.evaluate_kge --models transe complex
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("kg_artifacts/kge")
REPORTS_DIR    = Path("reports")


def load_and_report(model_name: str) -> dict | None:
    """Load PyKEEN pipeline result and extract metrics."""
    try:
        from pykeen.pipeline import PipelineResult
    except ImportError:
        logger.error("PyKEEN not installed.")
        return None

    model_dir = CHECKPOINT_DIR / model_name
    # PyKEEN stores metrics in slightly different locations depending on version
    candidates = [
        model_dir / "results.json",
        model_dir / "results" / "results.json",
        model_dir / "metric_results.json"
    ]
    
    metrics_file = None
    for cand in candidates:
        if cand.exists():
            metrics_file = cand
            break
            
    if not metrics_file:
        logger.warning("No results file found for %s. Train first.", model_name)
        return None

    data = json.loads(metrics_file.read_text())
    metrics = data.get("metrics", data)

    def get(domain, type_, metric):
        try:
            val = metrics[domain][type_][metric]
            # some older PyKEEN versions or raw files might float-format differently or nest
            return val
        except KeyError:
            # Fallback to flattened lookup just in case
            flat_key = f"{domain}.{type_}.{metric}"
            return metrics.get(flat_key, "n/a")

    report = {
        "model": model_name,
        "mrr_both":   get("both", "realistic", "inverse_harmonic_mean_rank"),
        "mrr_head":   get("head", "realistic", "inverse_harmonic_mean_rank"),
        "mrr_tail":   get("tail", "realistic", "inverse_harmonic_mean_rank"),
        "hits_at_1":  get("both", "realistic", "hits_at_1"),
        "hits_at_3":  get("both", "realistic", "hits_at_3"),
        "hits_at_10": get("both", "realistic", "hits_at_10"),
    }
    return report


def print_comparison(reports: list[dict]) -> None:
    header = f"{'Model':<12} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}"
    print("\n" + "=" * 56)
    print("  Filtered KGE Evaluation (head + tail)")
    print("=" * 56)
    print(header)
    print("-" * 56)
    for r in reports:
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        print(
            f"{r['model']:<12} "
            f"{fmt(r['mrr_both']):>8} "
            f"{fmt(r['hits_at_1']):>8} "
            f"{fmt(r['hits_at_3']):>8} "
            f"{fmt(r['hits_at_10']):>8}"
        )
    print("=" * 56 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KGE models")
    parser.add_argument(
        "--models", nargs="+", default=["transe", "complex"],
        help="Model names to evaluate (must be trained first)"
    )
    parser.add_argument("--out", default="reports/kge_metrics.json")
    args = parser.parse_args()

    reports = []
    for model_name in args.models:
        r = load_and_report(model_name)
        if r:
            reports.append(r)

    if not reports:
        logger.error("No models could be loaded. Run train_kge.py first.")
        return

    print_comparison(reports)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
