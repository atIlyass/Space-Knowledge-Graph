"""
src/kge/visualize.py
Visualize KGE entity embeddings with t-SNE and nearest-neighbor analysis.

Outputs:
  reports/figures/tsne.png         - 2D t-SNE, colored by entity class
  reports/figures/tsne_labels.png  - same figure with entity labels
  Console: top-5 nearest neighbors for 3 selected entities

Usage:
    python -m src.kge.visualize
    python -m src.kge.visualize --model transe --model-dir kg_artifacts/kge/transe
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")

# Entities to spotlight for nearest-neighbor analysis
SPOTLIGHT_ENTITIES = [
    "http://space-kg.org/resource/NASA",
    "http://space-kg.org/resource/Apollo_11",
    "http://space-kg.org/resource/Mars",
]

# Class → colour mapping for t-SNE
CLASS_COLORS = {
    "SpaceAgency":   "#4A90D9",
    "SpaceMission":  "#E67E22",
    "CelestialBody": "#27AE60",
    "Astronaut":     "#E74C3C",
    "Spacecraft":    "#9B59B6",
    "Telescope":     "#F39C12",
    "Other":         "#95A5A6",
}

SKG_O = "http://space-kg.org/ontology/"
SKG_R = "http://space-kg.org/resource/"


def load_entity_embeddings(model_dir: str) -> tuple[np.ndarray, dict, dict] | None:
    """
    Load entity embedding matrix and id→label/class mappings from PyKEEN result.

    Returns (embedding_matrix, id_to_label, id_to_class) or None on failure.
    """
    try:
        import torch
        from pykeen.pipeline import PipelineResult
    except ImportError:
        logger.error("PyKEEN / torch not installed.")
        return None

    model_path = Path(model_dir) / "trained_model.pkl"
    if not model_path.exists():
        logger.error("No trained_model.pkl found in %s. Train first.", model_dir)
        return None

    # In PyTorch 2.6, weights_only=True is the default but PyKEEN saves custom objects.
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    emb = model.entity_representations[0](indices=None).detach().cpu().numpy()
    logger.info("Loaded embeddings: %s", emb.shape)

    # Load entity-to-id mapping
    meta_path = Path(model_dir) / "training_triples" / "entity_to_id.tsv.gz"
    if not meta_path.exists():
        meta_path = Path(model_dir) / "entity_to_id.tsv"
    if not meta_path.exists():
        logger.warning("entity_to_id mapping not found; using integer ids")
        id_to_label = {i: str(i) for i in range(emb.shape[0])}
    else:
        import pandas as pd
        df = pd.read_csv(str(meta_path), sep="\t",
                         header=0 if str(meta_path).endswith(".tsv") else None,
                         compression="gzip" if str(meta_path).endswith(".gz") else None,
                         names=["entity", "id"])
        id_to_label = {row["id"]: row["entity"] for _, row in df.iterrows()}

    # Assign class by URI pattern
    id_to_class = {}
    class_keywords = {
        "Astronaut":   ["Astronaut", "astronaut", "Person", "Armstrong", "Shepard",
                         "Conrad", "Glenn", "Collins"],
        "SpaceMission": ["Apollo", "Mission", "mission", "Artemis", "Gemini", "Mercury",
                          "Soyuz", "Shuttle"],
        "SpaceAgency": ["NASA", "ESA", "SpaceX", "Roscosmos", "JAXA", "Agency"],
        "CelestialBody": ["Mars", "Moon", "Jupiter", "Saturn", "Earth", "Venus",
                           "Mercury_planet", "Pluto", "asteroid"],
        "Telescope":   ["Telescope", "Webb", "Hubble", "Spitzer"],
        "Spacecraft":  ["Spacecraft", "Dragon", "Falcon", "Orion", "Soyuz_craft",
                         "ISS", "Station"],
    }
    for idx, label in id_to_label.items():
        assigned = "Other"
        for cls, keywords in class_keywords.items():
            if any(k.lower() in label.lower() for k in keywords):
                assigned = cls
                break
        id_to_class[idx] = assigned

    return emb, id_to_label, id_to_class


def run_tsne(emb: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    """Reduce embeddings to 2D with t-SNE."""
    from sklearn.manifold import TSNE
    n = emb.shape[0]
    perp = min(perplexity, max(5, n // 3))
    logger.info("Running t-SNE on %d x %d matrix (perplexity=%d) …", n, emb.shape[1], perp)
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(emb)


def plot_tsne(
    coords_2d: np.ndarray,
    id_to_label: dict,
    id_to_class: dict,
    out_path: Path,
    add_labels: bool = False,
) -> None:
    """Save t-SNE scatter plot colored by class."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")

    colors = [CLASS_COLORS.get(id_to_class.get(i, "Other"), "#95A5A6")
              for i in range(len(coords_2d))]
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    ax.scatter(x, y, c=colors, s=12, alpha=0.7, linewidths=0)

    if add_labels:
        for i, (xi, yi) in enumerate(zip(x, y)):
            label = id_to_label.get(i, str(i)).split("/")[-1][:20]
            ax.annotate(label, (xi, yi), fontsize=4, alpha=0.6, color="white")

    # Legend
    patches = [
        mpatches.Patch(color=col, label=cls)
        for cls, col in CLASS_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              facecolor="#2a2a3e", labelcolor="white", edgecolor="#444")

    ax.set_title("Entity Embeddings - t-SNE (2D)", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def nearest_neighbors(
    emb: np.ndarray,
    id_to_label: dict,
    spotlight: list[str],
    top_k: int = 5,
) -> None:
    """Print top-k nearest neighbors for spotlight entities."""
    from sklearn.metrics.pairwise import cosine_similarity

    label_to_id = {v: k for k, v in id_to_label.items()}
    sim_matrix = cosine_similarity(emb)

    print("\n=== Nearest Neighbor Analysis ===")
    for target_uri in spotlight:
        target_id = label_to_id.get(target_uri)
        if target_id is None:
            # Try partial match
            for lbl, idx in label_to_id.items():
                if target_uri.split("/")[-1].lower() in lbl.lower():
                    target_id = idx
                    target_uri = lbl
                    break

        if target_id is None:
            print(f"\n  [{target_uri}] - not found in embedding index")
            continue

        sims = sim_matrix[target_id]
        neighbor_ids = np.argsort(-sims)[1 : top_k + 1]
        print(f"\n  Entity: {target_uri.split('/')[-1]}")
        print(f"  Top-{top_k} nearest neighbors (cosine similarity):")
        for rank, nid in enumerate(neighbor_ids, 1):
            nb_label = id_to_label.get(nid, str(nid)).split("/")[-1]
            print(f"    {rank}. {nb_label:<40}  sim={sims[nid]:.4f}")
    print("=================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE + nearest-neighbor visualization")
    parser.add_argument("--model",     default="transe",  choices=["transe", "complex"])
    parser.add_argument("--model-dir", default=None,
                        help="Override model directory path")
    parser.add_argument("--spotlight", nargs="+", default=SPOTLIGHT_ENTITIES,
                        help="Entity URIs for nearest-neighbor analysis")
    args = parser.parse_args()

    model_dir = args.model_dir or f"kg_artifacts/kge/{args.model}"

    result = load_entity_embeddings(model_dir)
    if result is None:
        return

    emb, id_to_label, id_to_class = result

    # t-SNE
    coords = run_tsne(emb)
    plot_tsne(coords, id_to_label, id_to_class,
              out_path=FIGURES_DIR / "tsne.png", add_labels=False)
    plot_tsne(coords, id_to_label, id_to_class,
              out_path=FIGURES_DIR / "tsne_labels.png", add_labels=True)

    # Nearest neighbors
    nearest_neighbors(emb, id_to_label, args.spotlight)


if __name__ == "__main__":
    main()
