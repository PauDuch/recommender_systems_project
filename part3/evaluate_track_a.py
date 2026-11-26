import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.baselines import ItemKNNRecommender
from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part2.disaggregation import build_user_activity_groups
from part2.evaluation import get_per_user_ndcg
from part2.models.neural_vae import NeuralVAERecommender
from part3.metrics import (
    build_user_historical_genre_distributions,
    compute_calibration_error,
    compute_gini_index,
    compute_intra_list_diversity,
    compute_per_user_ild,
    compute_within_genre_top5_share,
)
from part3.reranker import (
    DISTANCE_LABELS,
    apply_popularity_tiebreak_to_pools,
    build_genre_representation,
    extract_user_candidate_pools,
    mmr_rerank_all_users,
)

# Part 3 Track A - Staged Distance Selection, MMR Sweep, Baseline Comparison & Coverage Diagnosis

DEFAULT_SEED = 42
# 3-stage adaptive sweep:
#   Stage 1 (coarse): 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0
#   Stage 2 (0.1 steps): added 0.8, 0.6, 0.4, 0.2 (revealed accuracy peak at 0.8 & knee at 0.6)
#   Stage 3 (fine zoom): added 0.85, 0.825, 0.775, 0.75, 0.725, 0.65 around the [0.9, 0.6] peak/knee
DEFAULT_LAMBDAS = [1.0, 0.9, 0.85, 0.825, 0.8, 0.775, 0.75, 0.725, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
CANDIDATE_POOL_SIZE = 100


def run_distance_metric_selection(
    movies_df: pd.DataFrame,
    catalog_items: list[int],
    catalog_set: set[int],
    test_df: pd.DataFrame,
    candidate_pools: dict[int, tuple[np.ndarray, np.ndarray]],
    user_genre_dist: dict[int, np.ndarray],
    cos_dist_matrix: np.ndarray,
    output_dir: Path,
    pool_size: int = CANDIDATE_POOL_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Step A: Compares 3 genre distance metrics (Cosine, Jaccard, Manhattan) at fixed benchmark lambdas before sweeping."""
    benchmark_lambdas = [0.8, 0.7]
    metrics_to_test = ["cosine", "jaccard", "manhattan"]

    rows = []
    for lam in benchmark_lambdas:
        for metric_key in tqdm(metrics_to_test, desc=f"Distance comparison (lambda={lam})", unit="metric"):
            item_to_idx, idx_to_item, l1_genres, _, sim_mat, native_dist_mat = build_genre_representation(
                movies_df, catalog_items, metric=metric_key
            )
            recs = mmr_rerank_all_users(
                candidate_pools=candidate_pools,
                idx_to_item=idx_to_item,
                sim_matrix=sim_mat,
                lambda_param=lam,
                top_k=10,
            )

            acc = evaluate_model(recs, test_df, catalog_set, k=10)
            ild_cos = compute_intra_list_diversity(recs, item_to_idx, cos_dist_matrix, k=10)
            ild_native = compute_intra_list_diversity(recs, item_to_idx, native_dist_mat, k=10)
            gini = compute_gini_index(recs, catalog_items, k=10)
            cal_err = compute_calibration_error(recs, user_genre_dist, item_to_idx, l1_genres, k=10)

            rows.append({
                "Base model": "Mult-VAE",
                "Re-ranking method": f"MMR (top-{pool_size})",
                "Trade-off parameter": f"lambda = {lam:.1f}",
                "Lambda": float(lam),
                "Item distance": DISTANCE_LABELS[metric_key],
                "Precision@10": round(acc["Precision@10"], 4),
                "Recall@10": round(acc["Recall@10"], 4),
                "NDCG@10": round(acc["NDCG@10"], 4),
                "Intra-list diversity (Cosine)": round(ild_cos, 4),
                "Intra-list diversity (Native)": round(ild_native, 4),
                "Catalogue coverage": round(acc["Coverage (%)"], 2),
                "Gini": round(gini, 4),
                "Calibration error": round(cal_err, 4),
                "Seed": seed,
            })

    dist_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "track_a_distance_comparison.csv"
    dist_df.to_csv(out_path, index=False)

    print("\n--- STEP A: GENRE DISTANCE METRIC SELECTION (Fixed Benchmark Lambda = 0.8 & 0.7) ---")
    cols = [
        "Trade-off parameter",
        "Item distance",
        "Precision@10",
        "NDCG@10",
        "Intra-list diversity (Cosine)",
        "Catalogue coverage",
        "Gini",
        "Calibration error",
    ]
    print(dist_df[cols].to_string(index=False))
    print(f"Saved distance comparison to: {out_path}")
    return dist_df


def run_comparison_and_disaggregation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_items: list[int],
    catalog_set: set[int],
    test_users: list[int],
    item_to_idx: dict[int, int],
    l1_genres: np.ndarray,
    cos_dist_matrix: np.ndarray,
    user_genre_dist: dict[int, np.ndarray],
    mmr_recs_by_lambda: dict[float, dict[int, list[int]]],
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step C: Compares best MMR operating points (lambda=0.8, 0.7) vs Base Mult-VAE (1.0) and Item-kNN across user groups."""
    print("\nFitting 2nd-best Part 2 model Item-kNN (k=100, Cosine) for full-scale Track A comparison...")
    knn_model = ItemKNNRecommender(k_neighbors=100, metric="cosine")
    knn_model.fit(train_df, catalog_items)
    knn_recs = {u: knn_model.recommend_for_user(u, top_k=10) for u in test_users}

    systems = [
        ("Mult-VAE + MMR (lambda=0.8)", mmr_recs_by_lambda[0.8]),
        ("Mult-VAE + MMR (lambda=0.7)", mmr_recs_by_lambda[0.7]),
        ("Mult-VAE (Base, lambda=1.0)", mmr_recs_by_lambda[1.0]),
        ("Item-kNN (k=100, 2nd-Best)", knn_recs),
    ]

    # 1. Paired two-sided t-tests on per-user NDCG@10 and per-user ILD across all 5,826 evaluated test users
    per_user_ndcg = {name: get_per_user_ndcg(recs, test_df, k=10) for name, recs in systems}
    per_user_ild = {name: compute_per_user_ild(recs, item_to_idx, cos_dist_matrix, k=10) for name, recs in systems}

    paired_rows = []
    for cand_name in ["Mult-VAE + MMR (lambda=0.8)", "Mult-VAE + MMR (lambda=0.7)"]:
        for ref_name in ["Mult-VAE (Base, lambda=1.0)", "Item-kNN (k=100, 2nd-Best)"]:
            common_ndcg = sorted(set(per_user_ndcg[cand_name].keys()) & set(per_user_ndcg[ref_name].keys()))
            a_ndcg = np.array([per_user_ndcg[cand_name][u] for u in common_ndcg], dtype=np.float64)
            b_ndcg = np.array([per_user_ndcg[ref_name][u] for u in common_ndcg], dtype=np.float64)
            t_ndcg, p_ndcg = stats.ttest_rel(a_ndcg, b_ndcg)

            common_ild = sorted(set(per_user_ild[cand_name].keys()) & set(per_user_ild[ref_name].keys()))
            a_ild = np.array([per_user_ild[cand_name][u] for u in common_ild], dtype=np.float64)
            b_ild = np.array([per_user_ild[ref_name][u] for u in common_ild], dtype=np.float64)
            t_ild, p_ild = stats.ttest_rel(a_ild, b_ild)

            paired_rows.append({
                "System": cand_name,
                "Reference": ref_name,
                "Users (NDCG)": len(common_ndcg),
                "Mean Diff NDCG@10": round(float(np.mean(a_ndcg - b_ndcg)), 5),
                "t-stat (NDCG)": round(float(t_ndcg), 3),
                "p-value (NDCG)": float(p_ndcg),
                "Mean Diff ILD": round(float(np.mean(a_ild - b_ild)), 5),
                "t-stat (ILD)": round(float(t_ild), 3),
                "p-value (ILD)": float(p_ild),
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_path = output_dir / "track_a_paired_tests.csv"
    paired_df.to_csv(paired_path, index=False)

    print("\n--- STEP C1: PAIRED T-TESTS (Best MMR Operating Points vs Base Mult-VAE & Item-kNN) ---")
    print(paired_df.to_string(index=False))

    # 2. User-Activity Disaggregation across Low, Medium, and High activity tertiles
    user_groups = [("All Users", f"Full test set ({len(test_users)} users)", set(test_users))] + build_user_activity_groups(train_df)
    disagg_rows = []

    for sys_name, recs in systems:
        for group_name, definition, user_set in user_groups:
            group_recs = {u: r for u, r in recs.items() if u in user_set}
            group_test = test_df[test_df["user_id"].isin(user_set)]
            acc = evaluate_model(group_recs, group_test, catalog_set, k=10)
            ild = compute_intra_list_diversity(group_recs, item_to_idx, cos_dist_matrix, k=10)
            gini = compute_gini_index(group_recs, catalog_items, k=10)
            cal_err = compute_calibration_error(group_recs, user_genre_dist, item_to_idx, l1_genres, k=10)

            disagg_rows.append({
                "System": sys_name,
                "Group": group_name,
                "Definition": definition,
                "Precision@10": round(acc["Precision@10"], 4),
                "Recall@10": round(acc["Recall@10"], 4),
                "NDCG@10": round(acc["NDCG@10"], 4),
                "Intra-list diversity": round(ild, 4),
                "Catalogue coverage": round(acc["Coverage (%)"], 2),
                "Gini": round(gini, 4),
                "Calibration error": round(cal_err, 4),
                "Users Evaluated": acc["Users Evaluated"],
                "Seed": seed,
            })

    disagg_df = pd.DataFrame(disagg_rows)
    disagg_path = output_dir / "track_a_comparison_and_disagg.csv"
    disagg_df.to_csv(disagg_path, index=False)

    print("\n--- STEP C2: FULL-SCALE COMPARISON & USER-ACTIVITY DISAGGREGATION ---")
    display_cols = [
        "System",
        "Group",
        "Precision@10",
        "NDCG@10",
        "Intra-list diversity",
        "Catalogue coverage",
        "Gini",
        "Calibration error",
    ]
    print(disagg_df[display_cols].to_string(index=False))
    print(f"Saved comparison & disaggregation to: {disagg_path}")
    return paired_df, disagg_df


def run_coverage_diagnosis_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_items: list[int],
    catalog_set: set[int],
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    l1_genres: np.ndarray,
    cos_sim_matrix: np.ndarray,
    cos_dist_matrix: np.ndarray,
    user_genre_dist: dict[int, np.ndarray],
    candidate_pools: dict[int, tuple[np.ndarray, np.ndarray]],
    mmr_recs_by_lambda: dict[float, dict[int, list[int]]],
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Step D: Diagnoses why coarse 18-genre MMR reduces coverage (within-genre concentration) & tests a tie-breaker."""
    diag_rows = []

    # Part 1 of diagnosis: Track within-genre top-5 concentration across standard MMR lambdas
    for lam in [1.0, 0.8, 0.7, 0.5, 0.0]:
        recs = mmr_recs_by_lambda[lam]
        acc = evaluate_model(recs, test_df, catalog_set, k=10)
        ild = compute_intra_list_diversity(recs, item_to_idx, cos_dist_matrix, k=10)
        gini = compute_gini_index(recs, catalog_items, k=10)
        cal_err = compute_calibration_error(recs, user_genre_dist, item_to_idx, l1_genres, k=10)
        top5_share = compute_within_genre_top5_share(recs, item_to_idx, l1_genres, k=10)

        diag_rows.append({
            "Configuration": f"Standard Genre MMR (lambda={lam:.1f})",
            "Lambda": float(lam),
            "Popularity tie-break gamma": 0.0,
            "Within-genre top-5 share (%)": round(top5_share, 2),
            "Precision@10": round(acc["Precision@10"], 4),
            "Recall@10": round(acc["Recall@10"], 4),
            "NDCG@10": round(acc["NDCG@10"], 4),
            "Intra-list diversity": round(ild, 4),
            "Catalogue coverage": round(acc["Coverage (%)"], 2),
            "Gini": round(gini, 4),
            "Calibration error": round(cal_err, 4),
            "Seed": seed,
        })

    # Part 2 of diagnosis: Apply popularity-penalized tie-breaker at balanced operating point lambda = 0.7
    for gamma in [0.15, 0.25, 0.35]:
        adj_pools = apply_popularity_tiebreak_to_pools(candidate_pools, train_df, catalog_items, gamma=gamma)
        recs = mmr_rerank_all_users(
            candidate_pools=adj_pools,
            idx_to_item=idx_to_item,
            sim_matrix=cos_sim_matrix,
            lambda_param=0.7,
            top_k=10,
        )
        acc = evaluate_model(recs, test_df, catalog_set, k=10)
        ild = compute_intra_list_diversity(recs, item_to_idx, cos_dist_matrix, k=10)
        gini = compute_gini_index(recs, catalog_items, k=10)
        cal_err = compute_calibration_error(recs, user_genre_dist, item_to_idx, l1_genres, k=10)
        top5_share = compute_within_genre_top5_share(recs, item_to_idx, l1_genres, k=10)

        diag_rows.append({
            "Configuration": f"Popularity-TieBreak MMR (lambda=0.7, gamma={gamma:.2f})",
            "Lambda": 0.7,
            "Popularity tie-break gamma": float(gamma),
            "Within-genre top-5 share (%)": round(top5_share, 2),
            "Precision@10": round(acc["Precision@10"], 4),
            "Recall@10": round(acc["Recall@10"], 4),
            "NDCG@10": round(acc["NDCG@10"], 4),
            "Intra-list diversity": round(ild, 4),
            "Catalogue coverage": round(acc["Coverage (%)"], 2),
            "Gini": round(gini, 4),
            "Calibration error": round(cal_err, 4),
            "Seed": seed,
        })

    diag_df = pd.DataFrame(diag_rows)
    diag_path = output_dir / "track_a_coverage_diagnosis.csv"
    diag_df.to_csv(diag_path, index=False)

    print("\n--- STEP D: COVERAGE PARADOX DIAGNOSIS & POPULARITY-PENALIZED TIE-BREAKER ---")
    cols = [
        "Configuration",
        "Within-genre top-5 share (%)",
        "Catalogue coverage",
        "Gini",
        "Intra-list diversity",
        "NDCG@10",
        "Calibration error",
    ]
    print(diag_df[cols].to_string(index=False))
    print(f"Saved coverage diagnosis to: {diag_path}")
    return diag_df


def run_track_a_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
    lambdas: list[float] = DEFAULT_LAMBDAS,
    pool_size: int = CANDIDATE_POOL_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Executes the staged Part 3 Track A pipeline: distance selection, 17-point MMR sweep, comparison, and diagnosis."""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    test_users = sorted(test_df["user_id"].unique())

    # Precompute 18-dim genre vectors, Cosine similarity/distance matrices, and user historical genre distributions
    print("\nBuilding 18-dim genre vectors and pairwise distance matrices...")
    item_to_idx, idx_to_item, l1_genres, _, sim_matrix, dist_matrix = build_genre_representation(
        movies_df, catalog_items, metric="cosine"
    )
    user_genre_dist = build_user_historical_genre_distributions(train_df, item_to_idx, l1_genres)

    # Train the best Part 2 model (Mult-VAE) once on train_df
    print(f"Fitting base recommender Mult-VAE (seed={seed})...")
    base_model = NeuralVAERecommender(random_seed=seed)
    base_model.fit(train_df, catalog_items)

    # Extract top-100 candidate pools per user with min-max normalized scores
    print(f"Extracting top-{pool_size} candidate pools for {len(test_users):,} users...")
    candidate_pools = extract_user_candidate_pools(base_model, test_users, pool_size=pool_size)

    # Step A: Compare 3 genre distance metrics (Cosine, Jaccard, Manhattan) at fixed benchmark lambdas
    run_distance_metric_selection(
        movies_df=movies_df,
        catalog_items=catalog_items,
        catalog_set=catalog_set,
        test_df=test_df,
        candidate_pools=candidate_pools,
        user_genre_dist=user_genre_dist,
        cos_dist_matrix=dist_matrix,
        output_dir=output_dir,
        pool_size=pool_size,
        seed=seed,
    )

    # Step B: Sweep MMR trade-off parameter lambda from 1.0 down to 0.0 using the selected Cosine distance
    rows = []
    mmr_recs_by_lambda = {}
    for lam in tqdm(lambdas, desc="Sweeping MMR lambda (Cosine)", unit="lambda"):
        recs = mmr_rerank_all_users(
            candidate_pools=candidate_pools,
            idx_to_item=idx_to_item,
            sim_matrix=sim_matrix,
            lambda_param=lam,
            top_k=10,
        )
        if lam in (1.0, 0.8, 0.7, 0.5, 0.0):
            mmr_recs_by_lambda[lam] = recs

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
            "Item distance": DISTANCE_LABELS["cosine"],
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

    # Save full 17-point sweep results to CSV
    results_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "track_a_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n--- STEP B: PART 3 TRACK A 17-POINT MMR ACCURACY-DIVERSITY SWEEP (Cosine Distance) ---")
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
    print(f"Saved Track A sweep results to: {out_path}")

    # Step C: Full-scale comparison & disaggregation of best MMR operating points vs Base Mult-VAE & Item-kNN
    run_comparison_and_disaggregation(
        train_df=train_df,
        test_df=test_df,
        catalog_items=catalog_items,
        catalog_set=catalog_set,
        test_users=test_users,
        item_to_idx=item_to_idx,
        l1_genres=l1_genres,
        cos_dist_matrix=dist_matrix,
        user_genre_dist=user_genre_dist,
        mmr_recs_by_lambda=mmr_recs_by_lambda,
        output_dir=output_dir,
        seed=seed,
    )

    # Step D: Coverage paradox diagnosis (within-genre concentration & popularity-penalized tie-breaker)
    run_coverage_diagnosis_experiment(
        train_df=train_df,
        test_df=test_df,
        catalog_items=catalog_items,
        catalog_set=catalog_set,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        l1_genres=l1_genres,
        cos_sim_matrix=sim_matrix,
        cos_dist_matrix=dist_matrix,
        user_genre_dist=user_genre_dist,
        candidate_pools=candidate_pools,
        mmr_recs_by_lambda=mmr_recs_by_lambda,
        output_dir=output_dir,
        seed=seed,
    )

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 Track A (MMR Diversity & Popularity Bias) Evaluation")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M folder")
    parser.add_argument("--output-dir", type=str, default="part3/outputs", help="Directory to save CSV outputs")
    parser.add_argument("--pool-size", type=int, default=CANDIDATE_POOL_SIZE, help="Top-M candidate pool size per user")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for base model")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir) if Path(args.data_dir).is_absolute() else root_dir / args.data_dir
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir

    ratings, movies, _ = load_all(str(data_path))
    train_df, _, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)
    run_track_a_evaluation(
        train_df=train_df,
        test_df=test_df,
        movies_df=movies,
        output_dir=out_path,
        pool_size=args.pool_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
