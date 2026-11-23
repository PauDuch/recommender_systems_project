import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part2.models.neural_vae import NeuralVAERecommender
from part3.metrics import (
    build_user_historical_genre_distributions,
    compute_calibration_error,
    compute_gini_index,
    compute_intra_list_diversity,
)
from part3.reranker import (
    build_genre_representation,
    extract_user_candidate_pools,
    mmr_rerank_all_users,
)

# Part 3 Track A - MMR Trade-Off Sweep & Beyond-Accuracy Evaluation

DEFAULT_SEED = 42
# 3-stage adaptive sweep:
#   Stage 1 (coarse): 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0
#   Stage 2 (0.1 steps): added 0.8, 0.6, 0.4, 0.2 (revealed accuracy peak at 0.8 & knee at 0.6)
#   Stage 3 (fine zoom): added 0.85, 0.825, 0.775, 0.75, 0.725, 0.65 around the [0.9, 0.6] peak/knee
DEFAULT_LAMBDAS = [1.0, 0.9, 0.85, 0.825, 0.8, 0.775, 0.75, 0.725, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
CANDIDATE_POOL_SIZE = 100


def run_track_a_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
    lambdas: list[float] = DEFAULT_LAMBDAS,
    pool_size: int = CANDIDATE_POOL_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Fits Mult-VAE once, sweeps MMR lambda across test users, and computes accuracy + diversity metrics."""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    test_users = sorted(test_df["user_id"].unique())

    # Precompute genre vectors, pairwise cosine similarity/distance matrices, and user genre profiles
    print("\nBuilding 18-dim genre vectors and cosine distance matrix...")
    item_to_idx, idx_to_item, l1_genres, _, sim_matrix, dist_matrix = build_genre_representation(
        movies_df, catalog_items
    )
    user_genre_dist = build_user_historical_genre_distributions(train_df, item_to_idx, l1_genres)

    # Train the best Part 2 model (Mult-VAE) once on train_df
    print(f"Fitting base recommender Mult-VAE (seed={seed})...")
    base_model = NeuralVAERecommender(random_seed=seed)
    base_model.fit(train_df, catalog_items)

    # Extract top-100 candidate pools per user with min-max normalized scores
    print(f"Extracting top-{pool_size} candidate pools for {len(test_users):,} users...")
    candidate_pools = extract_user_candidate_pools(base_model, test_users, pool_size=pool_size)

    # Sweep MMR trade-off parameter lambda from 1.0 (unmodified base) down to 0.0 (pure diversity)
    rows = []
    for lam in tqdm(lambdas, desc="Sweeping MMR lambda", unit="lambda"):
        recs = mmr_rerank_all_users(
            candidate_pools=candidate_pools,
            idx_to_item=idx_to_item,
            sim_matrix=sim_matrix,
            lambda_param=lam,
            top_k=10,
        )

        # Evaluate top-10 ranking accuracy and beyond-accuracy metrics
        acc = evaluate_model(recs, test_df, catalog_set, k=10)
        ild = compute_intra_list_diversity(recs, item_to_idx, dist_matrix, k=10)
        gini = compute_gini_index(recs, catalog_items, k=10)
        cal_err = compute_calibration_error(recs, user_genre_dist, item_to_idx, l1_genres, k=10)

        is_ref = lam >= 1.0
        lam_str = f"{lam:.3f}".rstrip("0").rstrip(".") if round(lam * 1000) % 100 != 0 else f"{lam:.1f}"
        rows.append({
            "Base model": "Mult-VAE",
            "Re-ranking method": "None (Base Reference)" if is_ref else f"MMR (top-{pool_size})",
            "Trade-off parameter": f"lambda = {lam_str}",
            "Lambda": float(lam),
            "Item distance": "Cosine distance (18-dim L2 genres)",
            "Precision@10": round(acc["Precision@10"], 4),
            "Recall@10": round(acc["Recall@10"], 4),
            "NDCG@10": round(acc["NDCG@10"], 4),
            "Intra-list diversity": round(ild, 4),
            "Catalogue coverage": round(acc["Coverage (%)"], 2),
            "Gini": round(gini, 4),
            "Calibration error": round(cal_err, 4),
            "Seed": seed,
            "Notes": "Unmodified Mult-VAE reference" if is_ref else ("Pure diversity" if lam == 0.0 else f"MMR lambda={lam_str}"),
        })

    # Save full sweep results to CSV
    results_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "track_a_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n--- PART 3 TRACK A: MMR ACCURACY-DIVERSITY SWEEP ---")
    display_cols = [
        "Trade-off parameter",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Intra-list diversity",
        "Catalogue coverage",
        "Gini",
        "Calibration error",
    ]
    print(results_df[display_cols].to_string(index=False))
    print(f"\nSaved Track A results to: {out_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 Track A (MMR Diversity & Popularity Bias Sweep)")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M folder")
    parser.add_argument("--output-dir", type=str, default="part3/outputs", help="Directory to save CSV outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir) if Path(args.data_dir).is_absolute() else root_dir / args.data_dir
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir

    ratings, movies, _ = load_all(str(data_path))
    train_df, _, test_df = run_filter_split(ratings)
    run_track_a_evaluation(train_df, test_df, movies, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
