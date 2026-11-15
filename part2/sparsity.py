import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Sparsity Ablation (100%, 50%, 20% Training Interactions)

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part2.evaluation import get_model_specs

DEFAULT_SEED = 42
DEFAULT_FRACTIONS = [1.0, 0.5, 0.2]

def run_sparsity_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
    fractions: list[float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """ Creates subsamples of train_df given the fractions and evaluates all models on test_df"""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    test_users = sorted(test_df["user_id"].unique())
    model_specs = get_model_specs(movies_df)

    rows = []
    for frac in fractions:
        pct = int(round(frac * 100))
        # Per-user stratified subsampling ensures every user retains >= 1 training interaction
        sub_train = (
            train_df
            if frac >= 1.0
            else train_df.groupby("user_id", group_keys=False).sample(frac=frac, random_state=seed)
        )
        fallback_items = sub_train["item_id"].value_counts().index.tolist()[:10]

        frac_rows = []
        for name, _, builder in tqdm(model_specs, desc=f"Sparsity {pct}%", unit="model"):
            model = builder(seed)
            model.fit(sub_train, catalog_items)
            recs = {}
            for u in test_users:
                r = model.recommend_for_user(u, top_k=10)
                if len(r) < 10:
                    seen = set(r)
                    r = list(r) + [i for i in fallback_items if i not in seen][: 10 - len(r)]
                recs[u] = r
            metrics = evaluate_model(recs, test_df, catalog_set, k=10)

            frac_rows.append({
                "Model": name,
                "Training interactions (%)": f"{pct}%",
                "Train rows": len(sub_train),
                "Precision@10": round(metrics["Precision@10"], 4),
                "Recall@10": round(metrics["Recall@10"], 4),
                "NDCG@10": round(metrics["NDCG@10"], 4),
                "Catalogue coverage": round(metrics["Coverage (%)"], 2),
                "Seed(s)": seed,
            })

        # Add within-sparsity NDCG@10 rank to highlight ranking shifts
        frac_df = pd.DataFrame(frac_rows).sort_values(by="NDCG@10", ascending=False).reset_index(drop=True)
        frac_df["Rank (NDCG@10)"] = frac_df.index + 1
        rows.extend(frac_df.to_dict(orient="records"))

    results_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sparsity_part2.csv"
    results_df.to_csv(out_path, index=False)

    print("\n--- PART 2 SPARSITY ABLATION RESULTS ---")
    display_cols = ["Training interactions (%)", "Rank (NDCG@10)", "Model", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]
    print(results_df[display_cols].to_string(index=False))
    print(f"\nSaved sparsity results to: {out_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 Sparsity Ablation on Test Set")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M folder")
    parser.add_argument("--output-dir", type=str, default="part2/outputs", help="Directory to save CSV outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir) if Path(args.data_dir).is_absolute() else root_dir / args.data_dir
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir

    ratings, movies, _ = load_all(str(data_path))
    train_df, _, test_df = run_filter_split(ratings)
    run_sparsity_ablation(train_df, test_df, movies, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
