import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Set fixed random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from part1.data_loader import load_all
from part1.data_analysis import run_data_analysis
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part1.baselines import MostPopularRecommender, ItemKNNRecommender
from part2.evaluation import evaluate_part2_models
from part2.disaggregation import run_disaggregated_evaluation
from part2.sparsity import run_sparsity_ablation
from part2.plots import generate_part2_plots
from part3.evaluate_track_a import run_track_a_evaluation
from part3.plots import generate_part3_plots


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
        default=None,
        help="Project part to execute (omit to run all implemented parts)",
    )
    return parser.parse_args()


def run_baselines_evaluation(train_df: pd.DataFrame, eval_df: pd.DataFrame, catalog_items: list[int], output_dir: Path, eval_name: str = "Test"):
    """Evaluates Popularity variants and Item-kNN on eval_df and exports results matching official schema."""
    users_to_evaluate = sorted(eval_df["user_id"].unique())
    catalog_set = set(catalog_items)
    results = []

    print(f"\n--- Evaluating Baselines on {eval_name} Set ({len(users_to_evaluate):,} users) ---")

    # 1. Most Popular Variants
    popular_variants = [
        ("Most Popular", "All interactions", "all"),
        ("Most Popular", "Positive only (>= 4.0)", "positive_only"),
        ("Most Popular", "Damped average (m=50)", "average_rating"),
    ]

    for model_name, setting_desc, variant in popular_variants:
        pop_model = MostPopularRecommender(variant=variant)
        pop_model.fit(train_df, catalog_items)

        recs = {}
        for u in tqdm(users_to_evaluate, desc=f"Predicting: {model_name} ({setting_desc})"):
            recs[u] = pop_model.recommend_for_user(u, top_k=10)

        metrics = evaluate_model(recs, eval_df, catalog_set, k=10)
        results.append({
            "Model": model_name,
            "Variant / settings": setting_desc,
            "Similarity": "-",
            "k": "-",
            "Precision@10": round(metrics["Precision@10"], 4),
            "Recall@10": round(metrics["Recall@10"], 4),
            "NDCG@10": round(metrics["NDCG@10"], 4),
            "Catalogue coverage": f"{metrics['Coverage (%)']:.2f}%",
        })

    # 2. Item-based kNN (Cosine, optimal k=100 from validation tuning)
    item_knn = ItemKNNRecommender(k_neighbors=100, metric="cosine")
    print("\nFitting Item-based kNN (building sparse matrix & computing similarities)...")
    item_knn.fit(train_df, catalog_items)

    knn_recs = {}
    for u in tqdm(users_to_evaluate, desc="Predicting: Item-kNN (k=100, Cosine)"):
        knn_recs[u] = item_knn.recommend_for_user(u, top_k=10)

    knn_metrics = evaluate_model(knn_recs, eval_df, catalog_set, k=10)
    results.append({
        "Model": "Item-based kNN",
        "Variant / settings": "Neighborhood-based CF",
        "Similarity": "Cosine",
        "k": 100,
        "Precision@10": round(knn_metrics["Precision@10"], 4),
        "Recall@10": round(knn_metrics["Recall@10"], 4),
        "NDCG@10": round(knn_metrics["NDCG@10"], 4),
        "Catalogue coverage": f"{knn_metrics['Coverage (%)']:.2f}%",
    })

    # Format into DataFrame matching official 'Part 1 - Baselines' schema
    results_df = pd.DataFrame(results)[
        ["Model", "Variant / settings", "Similarity", "k", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]
    ]

    print(f"\n==================== BASELINE RESULTS ({eval_name} Set) ====================")
    print(results_df.to_string(index=False))
    print("=======================================================================\n")

    # Export machine-readable CSV to part1/outputs/
    out_csv = output_dir / "results_part1.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Results successfully saved to: {out_csv.resolve()}\n")

    return results_df


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    parts_to_run = [1, 2, 3] if args.part is None else [args.part]

    label = "All Parts (1, 2 & 3)" if args.part is None else f"Part {args.part}"
    print(f"=== Running Recommender Systems Pipeline: {label} ===")
    print(f"Dataset path: {data_dir.resolve()}\n")

    # Step 1: Ingestion
    ratings, movies, users = load_all(data_dir)

    if 1 in parts_to_run:
        outputs_dir = Path("part1") / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Data characterization and plots
        print("\nPart 1: Running data analysis...")
        run_data_analysis(ratings, movies, users, outputs_dir)

        # Step 3: Filtering & Temporal Split
        print("\nPart 1: Running filtering & temporal split...")
        train_df, val_df, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)

        # Step 4: Run Baselines & Evaluation
        catalog_items = sorted(movies["item_id"].unique())
        run_baselines_evaluation(train_df, test_df, catalog_items, outputs_dir, eval_name="Test")

    if 2 in parts_to_run:
        part2_out = Path("part2") / "outputs"
        part2_out.mkdir(parents=True, exist_ok=True)

        print("\nPart 2: Running filtering & temporal split...")
        train_df, _, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)

        print("\nPart 2: Running multi-seed test evaluation & paired tests...")
        evaluate_part2_models(train_df, test_df, movies, part2_out)

        print("\nPart 2: Running disaggregated evaluation...")
        run_disaggregated_evaluation(train_df, test_df, movies, part2_out)

        print("\nPart 2: Running sparsity ablation...")
        run_sparsity_ablation(train_df, test_df, movies, part2_out)

        print("\nPart 2: Generating report plots...")
        generate_part2_plots(part2_out)

    if 3 in parts_to_run:
        part3_out = Path("part3") / "outputs"
        part3_out.mkdir(parents=True, exist_ok=True)

        print("\nPart 3 (Track A): Running filtering & temporal split...")
        train_df, _, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)

        print("\nPart 3 (Track A): Running MMR diversity & popularity bias sweep...")
        run_track_a_evaluation(train_df, test_df, movies, part3_out)

        print("\nPart 3 (Track A): Generating accuracy-diversity frontier plot...")
        generate_part3_plots(part3_out)


if __name__ == "__main__":
    main()
