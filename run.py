import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from part1.data_loader import load_all
from part1.data_analysis import run_data_analysis
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part1.baselines import MostPopularRecommender, ItemKNNRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="Recommender Systems Project - MovieLens 1M Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml-1m",
        help="Path to directory containing MovieLens 1M .dat files (default: ml-1m)",
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Project part to execute (default: 1)",
    )
    return parser.parse_args()


def run_baselines_evaluation(train_df: pd.DataFrame, eval_df: pd.DataFrame, catalog_items: list[int], eval_name: str = "Test"):
    """Evaluates Popularity variants and Item-kNN on eval_df with progress bars."""
    users_to_evaluate = sorted(eval_df["user_id"].unique())
    catalog_set = set(catalog_items)
    results = []

    print(f"\n--- Evaluating Baselines on {eval_name} Set ({len(users_to_evaluate):,} users) ---")

    # 1. Most Popular Variants
    popular_variants = [
        ("MostPopular (All Interactions)", "all"),
        ("MostPopular (Positive Only)", "positive_only"),
        ("MostPopular (Damped Average)", "average_rating"),
    ]

    for model_name, variant in popular_variants:
        pop_model = MostPopularRecommender(variant=variant)
        pop_model.fit(train_df, catalog_items)

        recs = {}
        for u in tqdm(users_to_evaluate, desc=f"Predicting: {model_name}"):
            recs[u] = pop_model.recommend_for_user(u, top_k=10)

        metrics = evaluate_model(recs, eval_df, catalog_set, k=10)
        metrics["Model"] = model_name
        results.append(metrics)

    # 2. Item-based kNN (Cosine)
    item_knn = ItemKNNRecommender(k_neighbors=20, metric="cosine")
    print("\nFitting Item-based kNN (building sparse matrix & computing similarities)...")
    item_knn.fit(train_df, catalog_items)

    knn_recs = {}
    for u in tqdm(users_to_evaluate, desc="Predicting: Item-kNN (k=20, Cosine)"):
        knn_recs[u] = item_knn.recommend_for_user(u, top_k=10)

    knn_metrics = evaluate_model(knn_recs, eval_df, catalog_set, k=10)
    knn_metrics["Model"] = "Item-based kNN (k=20, Cosine)"
    results.append(knn_metrics)

    # Format into a clean summary table
    results_df = pd.DataFrame(results)[["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage (%)"]]
    print(f"\n==================== BASELINE RESULTS ({eval_name} Set) ====================")
    print(results_df.to_string(index=False))
    print("=======================================================================\n")

    return results_df


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"=== Running Recommender Systems Pipeline: Part {args.part} ===")
    print(f"Dataset path: {data_dir.resolve()}\n")

    # Step 1: Ingestion
    ratings, movies, users = load_all(data_dir)

    # Ensure output directory exists for Part 1
    outputs_dir = Path("part1") / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.part == 1:
        # Step 2: Data characterization and plots
        print("\nPart 1: Running data analysis...")
        run_data_analysis(ratings, movies, users, outputs_dir)

        # Step 3: Filtering & Temporal Split
        print("\nPart 1: Running filtering & temporal split...")
        train_df, val_df, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)

        # Step 4: Run Baselines & Evaluation
        catalog_items = sorted(movies["item_id"].unique())
        run_baselines_evaluation(train_df, test_df, catalog_items, eval_name="Test")


if __name__ == "__main__":
    main()
