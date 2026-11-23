import numpy as np
import pandas as pd

# Genre Vector Representation and Maximal Marginal Relevance (MMR) Re-ranking


def build_genre_representation(
    movies_df: pd.DataFrame, catalog_items: list[int]
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds 18-dim genre vectors, cosine similarity matrix, and cosine distance matrix for catalog items."""
    unique_items = sorted(set(catalog_items))
    item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}
    n_items = len(unique_items)

    movies_indexed = movies_df.set_index("item_id")

    # Collect all 18 unique genres across the catalog in sorted order
    genre_set = set()
    for genres_str in movies_df["genres"].dropna():
        for g in str(genres_str).split("|"):
            if g:
                genre_set.add(g)
    all_genres = sorted(genre_set)
    genre_to_col = {g: idx for idx, g in enumerate(all_genres)}
    n_genres = len(all_genres)

    # Build multi-hot genre matrix of shape (n_items, 18)
    raw_genres = np.zeros((n_items, n_genres), dtype=np.float32)
    for item_id, idx in item_to_idx.items():
        if item_id not in movies_indexed.index:
            continue
        genres_str = str(movies_indexed.loc[item_id, "genres"])
        for g in genres_str.split("|"):
            if g in genre_to_col:
                raw_genres[idx, genre_to_col[g]] = 1.0

    # L1-normalize each item row so its genre weights sum to 1.0 (used for Calibration Error)
    row_sums = raw_genres.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    l1_genres = raw_genres / row_sums

    # L2-normalize each item row to unit length (used for Cosine similarity and distance)
    l2_norms = np.linalg.norm(raw_genres, axis=1, keepdims=True)
    l2_norms[l2_norms == 0] = 1.0
    l2_genres = raw_genres / l2_norms

    # Precompute full pairwise cosine similarity and cosine distance (1 - cos_sim) matrices in [0, 1]
    sim_matrix = np.clip(l2_genres @ l2_genres.T, 0.0, 1.0).astype(np.float32)
    dist_matrix = (1.0 - sim_matrix).astype(np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    return item_to_idx, idx_to_item, l1_genres, l2_genres, sim_matrix, dist_matrix


def extract_user_candidate_pools(
    base_model,
    user_ids: list[int],
    pool_size: int = 100,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Extracts top-M unseen candidate item indices and min-max normalized scores [0, 1] per user."""
    candidate_pools = {}
    for user_id in user_ids:
        if user_id not in base_model.user_to_idx:
            continue
        u_idx = base_model.user_to_idx[user_id]
        scores = base_model._predict_scores(u_idx).astype(np.float64, copy=True)

        # Mask out items already rated by the user in the training set
        seen_indices = base_model.user_seen_indices.get(u_idx, [])
        if seen_indices:
            scores[seen_indices] = -np.inf

        # Retrieve top-M highest-scoring candidate indices sorted descending
        if len(scores) > pool_size:
            cand_idx = np.argpartition(scores, -pool_size)[-pool_size:]
            cand_idx = cand_idx[np.argsort(scores[cand_idx])[::-1]]
        else:
            cand_idx = np.argsort(scores)[::-1]

        # Min-max normalize the top-M candidate scores into [0, 1] so lambda is on the same scale as similarity
        cand_scores = scores[cand_idx]
        s_min = float(np.min(cand_scores))
        s_max = float(np.max(cand_scores))
        if s_max > s_min:
            norm_scores = (cand_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(cand_scores)

        candidate_pools[user_id] = (cand_idx, norm_scores.astype(np.float32))

    return candidate_pools


def mmr_rerank_user(
    cand_indices: np.ndarray,
    norm_scores: np.ndarray,
    sim_matrix: np.ndarray,
    lambda_param: float,
    top_k: int = 10,
) -> list[int]:
    """Greedy Maximal Marginal Relevance (MMR) selection of top-K item indices for one user."""
    # When lambda = 1.0, return unmodified base model top-K ranking
    if lambda_param >= 1.0:
        return cand_indices[:top_k].tolist()

    m = len(cand_indices)
    if m <= top_k:
        return cand_indices.tolist()

    # Slice the M x M pairwise similarity matrix for this user's candidate pool
    cand_sim = sim_matrix[np.ix_(cand_indices, cand_indices)]

    # Step 1: Always pick the top-scoring base item first (index 0 in sorted pool)
    selected_pos = [0]
    selected_mask = np.zeros(m, dtype=bool)
    selected_mask[0] = True

    # Track each candidate's maximum genre similarity to any item already in selected_pos
    max_sim_to_selected = cand_sim[:, 0].copy()

    # Steps 2..K: Greedily select item maximizing lambda * rel - (1 - lambda) * max_sim(i, S)
    for _ in range(1, top_k):
        mmr_scores = lambda_param * norm_scores - (1.0 - lambda_param) * max_sim_to_selected
        mmr_scores[selected_mask] = -np.inf

        best_pos = int(np.argmax(mmr_scores))
        selected_pos.append(best_pos)
        selected_mask[best_pos] = True

        # Update running maximum similarity against the newly selected item
        np.maximum(max_sim_to_selected, cand_sim[:, best_pos], out=max_sim_to_selected)

    return [int(cand_indices[pos]) for pos in selected_pos]


def mmr_rerank_all_users(
    candidate_pools: dict[int, tuple[np.ndarray, np.ndarray]],
    idx_to_item: dict[int, int],
    sim_matrix: np.ndarray,
    lambda_param: float,
    top_k: int = 10,
) -> dict[int, list[int]]:
    """Applies MMR re-ranking across all users for a given trade-off parameter lambda."""
    recommendations = {}
    for user_id, (cand_indices, norm_scores) in candidate_pools.items():
        top_indices = mmr_rerank_user(
            cand_indices=cand_indices,
            norm_scores=norm_scores,
            sim_matrix=sim_matrix,
            lambda_param=lambda_param,
            top_k=top_k,
        )
        # Convert internal indices back to original movie item_ids
        recommendations[user_id] = [idx_to_item[idx] for idx in top_indices]
    return recommendations
