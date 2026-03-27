"""
src/kge/train_kge.py
Train TransE and ComplEx KGE models using PyKEEN.

Configuration (defaults, all configurable via CLI flags)
---------------------------------------------------------
  --dim        128       embedding dimension
  --lr         0.01      learning rate
  --batch      512       batch size
  --epochs     100       training epochs  (default 100 per project plan)
  --model      both      'transe' | 'complex' | 'both'
  --neg-sampler basic    PyKEEN negative sampler

Usage:
    python -m src.kge.train_kge
    python -m src.kge.train_kge --epochs 50 --model transe
    python -m src.kge.train_kge --epochs 100 --dim 128 --model both
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("kg_artifacts/kge")


def train_model(
    model_name: str,
    train_path: str,
    valid_path: str,
    test_path: str,
    dim: int,
    lr: float,
    batch: int,
    epochs: int,
    neg_sampler: str,
) -> None:
    """Train a single PyKEEN model and save results."""
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.error("PyKEEN is not installed. Run: pip install pykeen")
        return

    logger.info("Loading triples from %s …", train_path)
    training   = TriplesFactory.from_path(train_path)
    validation = TriplesFactory.from_path(valid_path,
                                           entity_to_id=training.entity_to_id,
                                           relation_to_id=training.relation_to_id)
    testing    = TriplesFactory.from_path(test_path,
                                           entity_to_id=training.entity_to_id,
                                           relation_to_id=training.relation_to_id)

    logger.info(
        "Dataset: %d train | %d valid | %d test | %d entities | %d relations",
        training.num_triples, validation.num_triples, testing.num_triples,
        training.num_entities, training.num_relations,
    )

    model_map = {
        "transe":  "TransE",
        "complex": "ComplEx",
        "distmult": "DistMult",
    }
    pykeen_model = model_map.get(model_name.lower(), model_name)

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=pykeen_model,
        model_kwargs={
            "embedding_dim": dim,
        },
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch,
        },
        negative_sampler=neg_sampler,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        random_seed=42,
        use_tqdm=True,
    )

    # Save model
    out_dir = CHECKPOINT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(out_dir))

    # Print quick metrics
    metrics = result.metric_results.to_flat_dict()
    mrr    = metrics.get("both.realistic.inverse_harmonic_mean_rank", "n/a")
    h1     = metrics.get("both.realistic.hits_at_1",  "n/a")
    h3     = metrics.get("both.realistic.hits_at_3",  "n/a")
    h10    = metrics.get("both.realistic.hits_at_10", "n/a")

    print(f"\n{'='*50}")
    print(f"  Model : {pykeen_model}  (dim={dim}, lr={lr}, epochs={epochs})")
    print(f"  MRR   : {mrr}")
    print(f"  H@1   : {h1}")
    print(f"  H@3   : {h3}")
    print(f"  H@10  : {h10}")
    print(f"  Saved → {out_dir}")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KGE models (TransE + ComplEx)")
    parser.add_argument("--train",  default="data/processed/train.txt")
    parser.add_argument("--valid",  default="data/processed/valid.txt")
    parser.add_argument("--test",   default="data/processed/test.txt")
    parser.add_argument("--dim",    type=int,   default=128)
    parser.add_argument("--lr",     type=float, default=0.01)
    parser.add_argument("--batch",  type=int,   default=512)
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--model",  default="both",
                        choices=["transe", "complex", "both"],
                        help="Which model(s) to train")
    parser.add_argument("--neg-sampler", default="basic",
                        dest="neg_sampler")
    args = parser.parse_args()

    models_to_train = (
        ["transe", "complex"] if args.model == "both" else [args.model]
    )

    for m in models_to_train:
        logger.info("Training %s …", m)
        train_model(
            model_name=m,
            train_path=args.train,
            valid_path=args.valid,
            test_path =args.test,
            dim=args.dim,
            lr=args.lr,
            batch=args.batch,
            epochs=args.epochs,
            neg_sampler=args.neg_sampler,
        )


if __name__ == "__main__":
    main()
