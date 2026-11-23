import numpy as np
import pandas as pd

# Beyond-Accuracy Metrics for Track A (Intra-List Diversity, Gini Index, Calibration Error)

def compute_intra_list_diversity(
    recommendations: dict[int, list[int]],
    item_to_idx: dict[int, int],
    dist_matrix: np.ndarray,
    k: int = 10,
) -> float:
    """Computes mean Intra-List Diversity (average pairwise cosine genre distance in top-K)."""
    user_ilds = []
    # Precompute upper-triangle pair indices for a list of length K (45 pairs when K=10)
    triu_i, triu_j = np.triu_indices(k, k=1)

    for top_items in recommendations.values():
        indices = [item_to_idx[i] for i in top_items[:k] if i in item_to_idx]
        if len(indices) < 2:
            continue

        # Extract all pairwise cosine distances among the recommended items
        if len(indices) == k:
            pair_dists = dist_matrix[np.ix_(indices, indices)][triu_i, triu_j]
        else:
            ti, tj = np.triu_indices(len(indices), k=1)
            pair_dists = dist_matrix[np.ix_(indices, indices)][ti, tj]
        user_ilds.append(float(np.mean(pair_dists)))

    return float(np.mean(user_ilds)) if user_ilds else 0.0


def compute_gini_index(
    recommendations: dict[int, list[int]],
    catalog_items: list[int],
    k: int = 10,
) -> float:
    """Computes the Gini index of item recommendation exposure across the full catalog [0, 1]."""
    n_items = len(catalog_items)
    if n_items <= 1:
        return 0.0

    # Count how many times each catalog item appears in any user's top-K list
    counts = {item_id: 0 for item_id in catalog_items}
    for top_items in recommendations.values():
        for item_id in top_items[:k]:
            if item_id in counts:
                counts[item_id] += 1

    # Sort item exposure counts in ascending order for the standard Lorenz-curve Gini formula
    sorted_counts = np.sort(np.array(list(counts.values()), dtype=np.float64))
    total_recs = float(np.sum(sorted_counts))
    if total_recs == 0.0:
        return 0.0

    ranks = np.arange(1, n_items + 1, dtype=np.float64)
    gini = float(np.sum((2.0 * ranks - n_items - 1.0) * sorted_counts) / (n_items * total_recs))
    return gini


def build_user_historical_genre_distributions(
    train_df: pd.DataFrame,
    item_to_idx: dict[int, int],
    l1_genres: np.ndarray,
    positive_threshold: float = 4.0,
) -> dict[int, np.ndarray]:
    """Computes each user's historical genre distribution p(g|u) from positive training items."""
    pos_df = train_df[train_df["rating"] >= positive_threshold]
    user_pos_items = pos_df.groupby("user_id")["item_id"].apply(list).to_dict()
    all_user_items = train_df.groupby("user_id")["item_id"].apply(list).to_dict()

    user_genre_dist = {}
    for user_id, all_items in all_user_items.items():
        # Prefer positive interactions (rating >= 4.0); fall back to all training items if none exist
        items = user_pos_items.get(user_id, all_items)
        indices = [item_to_idx[i] for i in items if i in item_to_idx]
        if not indices:
            continue

        # Average the L1-normalized genre vectors across the user's liked history
        p_u = np.mean(l1_genres[indices], axis=0)
        total = float(np.sum(p_u))
        if total > 0:
            user_genre_dist[user_id] = (p_u / total).astype(np.float64)

    return user_genre_dist


def compute_calibration_error(
    recommendations: dict[int, list[int]],
    user_genre_dist: dict[int, np.ndarray],
    item_to_idx: dict[int, int],
    l1_genres: np.ndarray,
    k: int = 10,
    alpha: float = 0.01,
) -> float:
    """Computes mean Calibration Error via KL divergence KL(p_u || q_tilde_u) (Steck 2018)."""
    kl_values = []

    for user_id, top_items in recommendations.items():
        p_u = user_genre_dist.get(user_id)
        if p_u is None:
            continue

        indices = [item_to_idx[i] for i in top_items[:k] if i in item_to_idx]
        if not indices:
            continue

        # Compute recommended genre distribution q(g|u) over top-K items
        q_u = np.mean(l1_genres[indices], axis=0).astype(np.float64)
        q_sum = float(np.sum(q_u))
        if q_sum > 0:
            q_u /= q_sum

        # Smooth q_u with p_u (alpha = 0.01) to avoid log(0) when a genre is missing from top-K
        q_tilde = (1.0 - alpha) * q_u + alpha * p_u
        mask = p_u > 0
        kl = float(np.sum(p_u[mask] * np.log(p_u[mask] / q_tilde[mask])))
        kl_values.append(kl)

    return float(np.mean(kl_values)) if kl_values else 0.0
