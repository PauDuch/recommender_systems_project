import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Disaggregated Evaluation (User Activity & Item Popularity)

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part2.evaluation import get_model_specs

DEFAULT_SEED = 42


def build_user_activity_groups(train_df: pd.DataFrame) -> list[tuple[str, str, set[int]]]:
    """Splits users into Low, Medium, and High activity tertiles based on training interactions"""
    counts = train_df.groupby("user_id").size()
    q1 = int(np.quantile(counts.values, 1 / 3))
    q2 = int(np.quantile(counts.values, 2 / 3))

    low_users = set(counts[counts <= q1].index)
    mid_users = set(counts[(counts > q1) & (counts <= q2)].index)
    high_users = set(counts[counts > q2].index)

    return [
        ("Low Activity", f"<= {q1} train ratings ({len(low_users)} users)", low_users),
        ("Medium Activity", f"{q1 + 1}-{q2} train ratings ({len(mid_users)} users)", mid_users),
        ("High Activity", f"> {q2} train ratings ({len(high_users)} users)", high_users),
    ]


def build_item_popularity_groups(train_df: pd.DataFrame, catalog_items: list[int]) -> list[tuple[str, str, set[int]]]:
    """Splits catalog items into Popular (Top 20%), Mid-Tier (Next 30%), and Long-Tail (Bottom 50%)."""
    counts = train_df.groupby("item_id").size().reindex(catalog_items, fill_value=0)
    sorted_items = counts.sort_values(ascending=False)
    n_items = len(sorted_items)

    top_cut = int(0.20 * n_items)
    mid_cut = int(0.50 * n_items)

    pop_series = sorted_items.iloc[:top_cut]
    mid_series = sorted_items.iloc[top_cut:mid_cut]
    tail_series = sorted_items.iloc[mid_cut:]

    return [
        ("Popular (Top 20%)", f">= {int(pop_series.min())} train ratings ({len(pop_series)} items)", set(pop_series.index)),
        ("Mid-Tier (20-50%)", f"{int(mid_series.min())}-{int(mid_series.max())} train ratings ({len(mid_series)} items)", set(mid_series.index)),
        ("Long-Tail (Bottom 50%)", f"<= {int(tail_series.max())} train ratings ({len(tail_series)} items)", set(tail_series.index)),
    ]


def run_disaggregated_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Evaluates all baselines and models across user activity and item popularity groups."""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    test_users = sorted(test_df["user_id"].unique())

    user_groups = build_user_activity_groups(train_df)
    item_groups = build_item_popularity_groups(train_df, catalog_items)
    model_specs = get_model_specs(movies_df)

    rows = []
    for name, _, builder in tqdm(model_specs, desc="Disaggregated Eval", unit="model"):
        model = builder(seed)
        model.fit(train_df, catalog_items)
        recs = {u: model.recommend_for_user(u, top_k=10) for u in test_users}

        # Dimension 1: User Activity
        for group_name, definition, user_set in user_groups:
            group_recs = {u: r for u, r in recs.items() if u in user_set}
            group_test = test_df[test_df["user_id"].isin(user_set)]
            metrics = evaluate_model(group_recs, group_test, catalog_set, k=10)
            rows.append({
                "Model": name,
                "Breakdown dimension": "User Activity",
                "Group": group_name,
                "Definition / threshold": definition,
                "Precision@10": round(metrics["Precision@10"], 4),
                "Recall@10": round(metrics["Recall@10"], 4),
                "NDCG@10": round(metrics["NDCG@10"], 4),
                "Catalogue coverage": round(metrics["Coverage (%)"], 2),
                "Users Evaluated": metrics["Users Evaluated"],
                "Seed(s)": seed,
            })

        # Dimension 2: Item Popularity
        for group_name, definition, item_set in item_groups:
            group_test = test_df[test_df["item_id"].isin(item_set)]
            metrics = evaluate_model(recs, group_test, item_set, k=10)
            rec_items_in_group = {i for top in recs.values() for i in top[:10] if i in item_set}
            group_cov = (len(rec_items_in_group) / len(item_set)) * 100.0 if item_set else 0.0
            rows.append({
                "Model": name,
                "Breakdown dimension": "Item Popularity",
                "Group": group_name,
                "Definition / threshold": definition,
                "Precision@10": round(metrics["Precision@10"], 4),
                "Recall@10": round(metrics["Recall@10"], 4),
                "NDCG@10": round(metrics["NDCG@10"], 4),
                "Catalogue coverage": round(group_cov, 2),
                "Users Evaluated": metrics["Users Evaluated"],
                "Seed(s)": seed,
            })

    results_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "disaggregation_part2.csv"
    results_df.to_csv(out_path, index=False)

    print("\n--- PART 2 DISAGGREGATED RESULTS: USER ACTIVITY ---")
    u_df = results_df[results_df["Breakdown dimension"] == "User Activity"]
    print(u_df[["Model", "Group", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))

    print("\n--- PART 2 DISAGGREGATED RESULTS: ITEM POPULARITY ---")
    i_df = results_df[results_df["Breakdown dimension"] == "Item Popularity"]
    print(i_df[["Model", "Group", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))

    print(f"\nSaved disaggregated results to: {out_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 Disaggregated Evaluation on Test Set")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M folder")
    parser.add_argument("--output-dir", type=str, default="part2/outputs", help="Directory to save CSV outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir) if Path(args.data_dir).is_absolute() else root_dir / args.data_dir
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir

    ratings, movies, _ = load_all(str(data_path))
    train_df, _, test_df = run_filter_split(ratings)
    run_disaggregated_evaluation(train_df, test_df, movies, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
