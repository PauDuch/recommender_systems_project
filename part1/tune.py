import argparse
import random
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set fixed random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part1.baselines import ItemKNNRecommender


def tune_item_knn(train_df: pd.DataFrame, val_df: pd.DataFrame, catalog_items: list[int], output_dir: Path):
    """Sweeps neighborhood sizes k and similarity measures on the Validation set only."""
    users_to_evaluate = sorted(val_df["user_id"].unique())
    catalog_set = set(catalog_items)

    # Search space defined for Part 1 (Lecture 1.2, slides 5-10)
    k_values = [10, 20, 50, 100]
    metrics = ["cosine", "adjusted_cosine"]

    results = []
    print("\n--- Running Item-based kNN Hyperparameter Tuning (Validation Set) ---")
    print(f"Search space: k in {k_values}, similarity in {metrics}")
    print(f"Total configurations: {len(k_values) * len(metrics)}")

    for metric in metrics:
        # Build matrix and compute full similarity once per metric
        print(f"\nBuilding Item-kNN model with similarity='{metric}'...")
        knn = ItemKNNRecommender(metric=metric)
        knn.fit(train_df, catalog_items)

        for k in k_values:
            knn.set_k(k)
            recs = {}
            for u in tqdm(users_to_evaluate, desc=f"Val Eval [sim={metric}, k={k}]"):
                recs[u] = knn.recommend_for_user(u, top_k=10)

            eval_res = evaluate_model(recs, val_df, catalog_set, k=10)
            eval_res["Similarity"] = metric
            eval_res["k"] = k
            results.append(eval_res)

    results_df = pd.DataFrame(results)[
        ["Similarity", "k", "Precision@10", "Recall@10", "NDCG@10", "Coverage (%)"]
    ]

    # Sort by NDCG@10 descending to identify the best hyperparameter setting
    results_df = results_df.sort_values(by="NDCG@10", ascending=False).reset_index(drop=True)

    print("\n==================== TUNING RESULTS (Validation Set) ====================")
    print(results_df.to_string(index=False))
   
    best_row = results_df.iloc[0]
    print(f"Best Configuration: Similarity='{best_row['Similarity']}', k={best_row['k']} (Validation NDCG@10: {best_row['NDCG@10']:.4f})")

    # Save to CSV in outputs
    out_path = output_dir / "tuning_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Tuning results saved to: {out_path.resolve()}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Part 1 Hyperparameter Tuning for Item-based kNN")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outputs_dir = Path("part1") / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=== Recommender Systems: Item-kNN Hyperparameter Tuning ===")
    ratings, movies, users = load_all(data_dir)
    train_df, val_df, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)

    catalog_items = sorted(movies["item_id"].unique())
    tune_item_knn(train_df, val_df, catalog_items, outputs_dir)


if __name__ == "__main__":
    main()
