import argparse
import random
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.data_loader import load_all
from part1.filter_split import run_filter_split
from part1.metrics import evaluate_model
from part2.models.content_based import ContentBasedRecommender
from part2.models.mf_sgd import MFSGDRecommender
from part2.models.bpr import BPRRecommender
from part2.models.neural_vae import NeuralVAERecommender

# Fixed random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def evaluate_config(model, train_df, val_df, catalog_items, catalog_set, val_users) -> dict:
    """Fits a single model configuration on train_df and evaluates it on val_df."""
    model.fit(train_df, catalog_items)
    recs = {u: model.recommend_for_user(u, top_k=10) for u in val_users}
    return evaluate_model(recs, val_df, catalog_set, k=10)


def tune_part2_models(train_df: pd.DataFrame, val_df: pd.DataFrame, movies_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Tunes all Part 2 models on val_df with an equal 8 configurations per model."""
    catalog_items = sorted(movies_df["item_id"].unique())
    catalog_set = set(catalog_items)
    val_users = sorted(val_df["user_id"].unique())

    all_results = []

    # Content-Based Profile Matching (8 configs)
    print("\n Tuning Content-Based Recommender")
    cb_configs = [
        {"use_tfidf": tfidf, "weight_mode": mode, "year_weight": yw}
        for tfidf in [True, False]
        for mode in ["centered", "positive_only"]
        for yw in [0.0, 0.5]
    ]
    for cfg in tqdm(cb_configs, desc="Tuning Content-Based", unit="cfg"):
        model = ContentBasedRecommender(movies_df=movies_df, random_seed=RANDOM_SEED, **cfg)
        res = evaluate_config(model, train_df, val_df, catalog_items, catalog_set, val_users)
        res["Model"] = "Content-Based"
        res["Config"] = str(cfg)
        all_results.append(res)

    # 2. Latent-Factor CF with SGD (8 configs)
    print("\n Tuning MF-SGD Recommender")
    mf_configs = [
        {"n_factors": f, "reg_all": reg, "lr_all": lr, "n_epochs": 25}
        for f in [32, 64]
        for reg in [0.02, 0.1]
        for lr in [0.005, 0.01]
    ]
    for cfg in tqdm(mf_configs, desc="Tuning MF-SGD", unit="cfg"):
        model = MFSGDRecommender(random_seed=RANDOM_SEED, **cfg)
        res = evaluate_config(model, train_df, val_df, catalog_items, catalog_set, val_users)
        res["Model"] = "MF-SGD"
        res["Config"] = str(cfg)
        all_results.append(res)

    # 3. Ranking-Loss CF with BPR-MF (8 configs)
    print("\n Tuning BPR-MF Recommender")
    bpr_configs = [
        {"factors": f, "learning_rate": lr, "regularization": reg, "iterations": 100}
        for f in [64, 128]
        for lr in [0.01, 0.05]
        for reg in [0.001, 0.01]
    ]
    for cfg in tqdm(bpr_configs, desc="Tuning BPR-MF", unit="cfg"):
        model = BPRRecommender(random_seed=RANDOM_SEED, **cfg)
        res = evaluate_config(model, train_df, val_df, catalog_items, catalog_set, val_users)
        res["Model"] = "BPR-MF"
        res["Config"] = str(cfg)
        all_results.append(res)

    # 4. Neural Recommender with Mult-VAE (8 configs)
    print("\n Tuning Mult-VAE Recommendee")
    vae_configs = [
        {"hidden_dim": h, "latent_dim": l, "dropout": d, "beta_max": b, "epochs": 30, "lr": 1e-3}
        for (h, l) in [(400, 128), (600, 200)]
        for d in [0.3, 0.5]
        for b in [0.2, 0.5]
    ]
    for cfg in tqdm(vae_configs, desc="Tuning Mult-VAE", unit="cfg"):
        model = NeuralVAERecommender(random_seed=RANDOM_SEED, **cfg)
        res = evaluate_config(model, train_df, val_df, catalog_items, catalog_set, val_users)
        res["Model"] = "Mult-VAE"
        res["Config"] = str(cfg)
        all_results.append(res)

    results_df = pd.DataFrame(all_results)[
        ["Model", "Config", "Precision@10", "Recall@10", "NDCG@10", "Coverage (%)"]
    ]
    results_df = results_df.sort_values(by=["Model", "NDCG@10"], ascending=[True, False]).reset_index(drop=True)

    out_path = output_dir / "tuning_results_part2.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== PART 2 BEST CONFIG PER MODEL (Validation Set) ===")
    best_df = results_df.groupby("Model", as_index=False).first().sort_values(by="NDCG@10", ascending=False)
    display_df = best_df[["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage (%)"]].copy()
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest Hyperparameters:")
    for _, row in best_df.iterrows():
        print(f"  - {row['Model']}: {row['Config']}")
    print(f"\nFull tuning results saved to: {out_path.resolve()}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Part 2 Hyperparameter Tuning (Validation Set Only)")
    parser.add_argument("--data-dir", type=str, default="ml-1m", help="Path to MovieLens 1M directory")
    args = parser.parse_args()

    outputs_dir = Path("part2") / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, users = load_all(args.data_dir)
    train_df, val_df, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)
    tune_part2_models(train_df, val_df, movies, outputs_dir)


if __name__ == "__main__":
    main()
