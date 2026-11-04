from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class MostPopularRecommender:
    """Ranks items by popularity in the training set according to chosen variant."""

    def __init__(self, variant: str = "all", positive_threshold: float = 4.0):
        self.variant = variant
        self.positive_threshold = positive_threshold
        self.ranked_items = []
        self.full_ranking = []
        self.user_seen_items = {}

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int]):
        """Learns popularity ranking from training interactions and precomputes full ranking."""
        self.user_seen_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        if self.variant == "all":
            # Variant 1: All-time interaction count
            counts = train_df.groupby("item_id").size()
            self.ranked_items = counts.sort_values(ascending=False).index.tolist()

        elif self.variant == "positive_only":
            # Variant 2: Positive-only interaction count (rating >= 4.0)
            pos_df = train_df[train_df["rating"] >= self.positive_threshold]
            counts = pos_df.groupby("item_id").size()
            self.ranked_items = counts.sort_values(ascending=False).index.tolist()

        elif self.variant == "average_rating":
            # Variant 3: Damped average rating (Bayesian mean with prior m=50)
            global_mean = train_df["rating"].mean()
            stats = train_df.groupby("item_id")["rating"].agg(["count", "mean"])
            m = 50.0
            damped_score = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
            self.ranked_items = damped_score.sort_values(ascending=False).index.tolist()

        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Precompute the complete global ranking once
        ranked_set = set(self.ranked_items)
        unranked_catalog = [i for i in catalog_items if i not in ranked_set]
        self.full_ranking = self.ranked_items + unranked_catalog

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[int]:
        """Returns top-K unseen items for a single user in microseconds."""
        seen = self.user_seen_items.get(user_id, set())
        recs = []
        for item in self.full_ranking:
            if item not in seen:
                recs.append(item)
                if len(recs) == top_k:
                    break
        return recs


class ItemKNNRecommender:
    """Item-to-Item Collaborative Filtering (Lecture 1.2, slides 5-10)."""

    def __init__(self, k_neighbors: int = 20, metric: str = "cosine"):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.user_seen_items = {}
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.raw_similarity_matrix = None
        self.topk_similarity_matrix = None
        self.R = None

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int]):
        """Builds sparse interaction matrix and computes full pairwise item similarities."""
        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(list(set(catalog_items)))

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_seen_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        u_indices = train_df["user_id"].map(self.user_to_idx).values
        i_indices = train_df["item_id"].map(self.item_to_idx).values
        ratings = train_df["rating"].values.astype(np.float32)

        # Standard Cosine uses raw ratings; Adjusted Cosine subtracts user mean
        if self.metric == "adjusted_cosine":
            user_means = train_df.groupby("user_id")["rating"].transform("mean").values
            ratings = ratings - user_means

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        self.R = csr_matrix((ratings, (u_indices, i_indices)), shape=(n_users, n_items))

        # Compute full item-item similarity matrix (dense float32 for fast top-k pruning)
        sim = cosine_similarity(self.R.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        self.raw_similarity_matrix = sim
        self.set_k(self.k_neighbors)

    def set_k(self, k: int):
        """Prunes similarity matrix to retain only top-k neighbors per item (Lecture 1.2, slide 7)."""
        self.k_neighbors = k
        n_items = self.raw_similarity_matrix.shape[0]
        
        # For each item column, keep only top-k positive similarities
        pruned_sim = np.zeros_like(self.raw_similarity_matrix)
        for j in range(n_items):
            col = self.raw_similarity_matrix[:, j]
            # Find indices of top-k largest similarities
            topk_idx = np.argpartition(col, -k)[-k:]
            # Only keep positive similarities
            positive_mask = col[topk_idx] > 0
            valid_idx = topk_idx[positive_mask]
            pruned_sim[valid_idx, j] = col[valid_idx]

        self.topk_similarity_matrix = csr_matrix(pruned_sim)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[int]:
        """Returns top-K unseen items for a single user using strictly top-k item neighbors."""
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        seen_items = self.user_seen_items.get(user_id, set())

        # Fast sparse dot product: score_j = sum_{i in N_k(j)} r_{u,i} * sim(i, j)
        u_ratings = self.R[u_idx]
        scores = u_ratings.dot(self.topk_similarity_matrix).toarray().ravel()

        # Mask out already rated items
        for seen in seen_items:
            idx = self.item_to_idx.get(seen)
            if idx is not None:
                scores[idx] = -np.inf

        # argpartition for fast top-10 retrieval
        if len(scores) > top_k:
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            sorted_top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        else:
            sorted_top_indices = np.argsort(scores)[::-1]

        return [self.idx_to_item[idx] for idx in sorted_top_indices]
