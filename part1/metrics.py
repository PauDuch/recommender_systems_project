import numpy as np
import pandas as pd

def compute_user_metrics(recommended_items: list[int] | np.ndarray, ground_truth_relevant: set[int], 
    k: int = 10) -> tuple[float, float, float]:
    """Computes Precision@K, Recall@K, and NDCG@K for one user given recommended items, groundtruth items, k"""
    if not ground_truth_relevant:
        return 0.0, 0.0, 0.0

    # Check which recommended items are in ground truth (1 if hit, 0 otherwise)
    top_k = recommended_items[:k]
    hits = [1 if item in ground_truth_relevant else 0 for item in top_k]
    num_hits = sum(hits)

    # Precision@K: Proportion of top-k recommended items that are relevant (hits / k)
    precision = num_hits / k

    # Recall@K: Proportion of total relevant items that were successfully retrieved in top-k (hits /ground_truth)
    recall = num_hits / len(ground_truth_relevant)

    # NDCG@K: Ranking quality; discounts hits at lower positions using 1 / log2(rank + 1)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.array(hits) / discounts)
    # Ideal DCG: best possible score if all hits were placed at the very top ranks
    num_ideal_hits = min(k, len(ground_truth_relevant))
    idcg = np.sum(1.0 / discounts[:num_ideal_hits])
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_model(recommendations: dict[int, list[int]], eval_df: pd.DataFrame, catalog_item_ids: set[int], k: int = 10, relevance_threshold: float = 4.0) -> dict[str, float]:
    """Evaluates all users: averages Precision@K, Recall@K, NDCG@K, and computes total Catalogue Coverage"""

    # Filter ground truth by relevance threshold (rating >= 4.0 counts as positive feedback)
    relevant_df = eval_df[eval_df["rating"] >= relevance_threshold]
    ground_truth = relevant_df.groupby("user_id")["item_id"].apply(set).to_dict()

    precisions = []
    recalls = []
    ndcgs = []
    all_recommended_items = set()

    for user_id, top_items in recommendations.items():
        user_gt = ground_truth.get(user_id, set())

        # Evaluate users who have at least one relevant item in the evaluation set
        if len(user_gt) > 0:
            p, r, n = compute_user_metrics(top_items, user_gt, k=k)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)

        # Track unique items recommended across the entire population
        all_recommended_items.update(top_items[:k])

    # Catalogue Coverage: percentage of the full catalogue recommended to at least one user
    coverage = (len(all_recommended_items) / len(catalog_item_ids)) * 100.0

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Coverage (%)": float(coverage),
        "Users Evaluated": len(precisions),
    }
