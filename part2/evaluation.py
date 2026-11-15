import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Final test set evaluation, with 5 multi seed intervals and paired test

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.baselines import ItemKNNRecommender, MostPopularRecommender
from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import compute_user_metrics, evaluate_model
from part2.models.bpr import BPRRecommender
from part2.models.content_based import ContentBasedRecommender
from part2.models.mf_sgd import MFSGDRecommender
from part2.models.neural_vae import NeuralVAERecommender

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def get_per_user_ndcg(recommendations: dict[int, list[int]], eval_df: pd.DataFrame, k: int = 10, relevance_threshold: float = 4.0) -> dict[int, float]:
    """Computes per-user NDCG@K for paired statistical significance testing"""
    relevant_df = eval_df[eval_df["rating"] >= relevance_threshold]
    ground_truth = relevant_df.groupby("user_id")["item_id"].apply(set).to_dict()

    user_ndcg = {}
    for user_id, top_items in recommendations.items():
        user_gt = ground_truth.get(user_id, set())
        if len(user_gt) > 0:
            _, _, ndcg = compute_user_metrics(top_items, user_gt, k=k)
            user_ndcg[user_id] = ndcg
    return user_ndcg


def compute_mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Returns (mean, std, half_width_95ci) across multiple random seeds."""
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=len(arr) - 1))
    ci95 = t_crit * (std / np.sqrt(len(arr)))
    return mean, std, float(ci95)


def run_paired_tests(per_user_scores: dict[str, dict[int, float]], reference_models: list[str]) -> pd.DataFrame:
    """Runs paired t-tests on per-user NDCG@10 against each reference model"""
    rows = []
    model_names = list(per_user_scores.keys())

    for ref in reference_models:
        if ref not in per_user_scores:
            continue
        ref_scores = per_user_scores[ref]
        common_users = sorted(ref_scores.keys())

        for model_name in model_names:
            if model_name == ref:
                continue
            cand_scores = per_user_scores[model_name]
            users = [u for u in common_users if u in cand_scores]
            a = np.array([cand_scores[u] for u in users], dtype=np.float64)
            b = np.array([ref_scores[u] for u in users], dtype=np.float64)

            t_stat, p_val = stats.ttest_rel(a, b)
            rows.append({
                "Model": model_name,
                "Reference": ref,
                "Users": len(users),
                "Mean Diff (NDCG@10)": float(np.mean(a - b)),
                "t-statistic": float(t_stat),
                "p-value": float(p_val),
                "Significant (p < 0.05)": bool(p_val < 0.05),
            })

    return pd.DataFrame(rows)


def evaluate_part2_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
    seeds: list[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates Part 1 baselines and tuned Part 2 models on test_df with multi-seed CIs and paired tests."""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    test_users = sorted(test_df["user_id"].unique())

    # Deterministic models run once, stochastic models run across all seeds
    model_specs = [
        ("Most Popular (Pos)", False, lambda seed: MostPopularRecommender(variant="positive_only")),
        ("Item-kNN (k=100)", False, lambda seed: ItemKNNRecommender(k_neighbors=100, metric="cosine")),
        ("Content-Based", False, lambda seed: ContentBasedRecommender(movies_df=movies_df, random_seed=seed)),
        ("MF-SGD", True, lambda seed: MFSGDRecommender(random_seed=seed)),
        ("BPR-MF", True, lambda seed: BPRRecommender(random_seed=seed)),
        ("Mult-VAE", True, lambda seed: NeuralVAERecommender(random_seed=seed)),
    ]

    summary_rows = []
    per_user_ndcg_primary = {}

    for name, is_stochastic, builder in model_specs:
        run_seeds = seeds if is_stochastic else [seeds[0]]
        seed_metrics = []

        for idx, seed in enumerate(tqdm(run_seeds, desc=f"Evaluating {name}", unit="seed")):
            model = builder(seed)
            model.fit(train_df, catalog_items)
            recs = {u: model.recommend_for_user(u, top_k=10) for u in test_users}
            res = evaluate_model(recs, test_df, catalog_set, k=10)
            seed_metrics.append(res)

            # Save per-user NDCG@10 on the primary seed for paired statistical tests
            if idx == 0:
                per_user_ndcg_primary[name] = get_per_user_ndcg(recs, test_df, k=10)

        p_mean, _, p_ci = compute_mean_ci([m["Precision@10"] for m in seed_metrics])
        r_mean, _, r_ci = compute_mean_ci([m["Recall@10"] for m in seed_metrics])
        n_mean, _, n_ci = compute_mean_ci([m["NDCG@10"] for m in seed_metrics])
        c_mean, _, c_ci = compute_mean_ci([m["Coverage (%)"] for m in seed_metrics])

        summary_rows.append({
            "Model": name,
            "Stochastic": is_stochastic,
            "Seeds": str(run_seeds),
            "Precision@10 Mean": p_mean,
            "Precision@10 CI95": p_ci,
            "Recall@10 Mean": r_mean,
            "Recall@10 CI95": r_ci,
            "NDCG@10 Mean": n_mean,
            "NDCG@10 CI95": n_ci,
            "Coverage (%) Mean": c_mean,
            "Coverage (%) CI95": c_ci,
            "Precision@10": f"{p_mean:.4f} ± {p_ci:.4f}" if is_stochastic else f"{p_mean:.4f}",
            "Recall@10": f"{r_mean:.4f} ± {r_ci:.4f}" if is_stochastic else f"{r_mean:.4f}",
            "NDCG@10": f"{n_mean:.4f} ± {n_ci:.4f}" if is_stochastic else f"{n_mean:.4f}",
            "Coverage (%)": f"{c_mean:.2f} ± {c_ci:.2f}" if is_stochastic else f"{c_mean:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="NDCG@10 Mean", ascending=False).reset_index(drop=True)
    paired_df = run_paired_tests(per_user_ndcg_primary, reference_models=["Item-kNN (k=100)", "Mult-VAE"])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "test_results_part2.csv"
    paired_path = output_dir / "paired_tests_part2.csv"
    summary_df.to_csv(summary_path, index=False)
    paired_df.to_csv(paired_path, index=False)

    print("\n--- PART 2 TEST-SET RESULTS (Mean ± 95% CI) ---")
    display_cols = ["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage (%)", "Seeds"]
    print(summary_df[display_cols].to_string(index=False))

    print("\n--- PAIRED T-TESTS ON PER-USER NDCG@10 (Primary Seed) ---")
    print(paired_df.to_string(index=False))

    print(f"\nSaved test results to: {summary_path}")
    print(f"Saved paired tests to: {paired_path}")
    return summary_df, paired_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 1 Baselines & Tuned Part 2 Models on Test Set")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M folder")
    parser.add_argument("--output-dir", type=str, default="part2/outputs", help="Directory to save CSV outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds for stochastic models")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = root_dir / data_path

    out_path = Path(args.output_dir)
    if not out_path.is_absolute():
        out_path = root_dir / out_path

    ratings, movies, _ = load_all(str(data_path))
    train_df, _, test_df = run_filter_split(ratings)
    evaluate_part2_models(train_df, test_df, movies, out_path, seeds=args.seeds)


if __name__ == "__main__":
    main()
