import numpy as np
import pandas as pd

# Base class for all models of Part 2

class BaseRecommender:
    """Handles ID-to-index mappings, seen-item tracking, and top-K recommendation extraction"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_seen_items = {}
        self.user_seen_indices = {}

    def _build_mappings(self, train_df: pd.DataFrame, catalog_items: list[int]):
        """Builds user/item index mappings and records training interactions per user"""
        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(set(catalog_items))

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for u, idx in self.user_to_idx.items()}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        self.user_seen_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.user_seen_indices = {}
        for u, items in self.user_seen_items.items():
            u_idx = self.user_to_idx[u]
            self.user_seen_indices[u_idx] = [self.item_to_idx[i] for i in items if i in self.item_to_idx]

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int], **kwargs):
        raise NotImplementedError("Each model must implement fit().")

    def _predict_scores(self, u_idx: int) -> np.ndarray:
        raise NotImplementedError("Each model must implement _predict_scores().")

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[int]:
        """Returns top-K unseen catalog item IDs for a given user_id."""
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        scores = self._predict_scores(u_idx).astype(np.float64, copy=True)

        # Mask out items the user already rated in the training set
        seen_indices = self.user_seen_indices.get(u_idx, [])
        if seen_indices:
            scores[seen_indices] = -np.inf

        # Sort and pick the top-K highest scoring items
        if len(scores) > top_k:
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            sorted_top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        else:
            sorted_top_indices = np.argsort(scores)[::-1]

        return [self.idx_to_item[idx] for idx in sorted_top_indices]
