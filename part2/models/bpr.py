import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from part2.models.base import BaseRecommender

# Ranking-Loss Collaborative Filtering with Bayesian Personalized Ranking

class BPRRecommender(BaseRecommender):
    """Pairwise ranking-loss Matrix Factorization (BPR-MF) using the library."""

    def __init__(
        self,
        factors: int = 128,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        positive_threshold: float = 4.0,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.positive_threshold = positive_threshold
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int], **kwargs):
        """Builds positive interaction matrix and trains BPR-MF."""
        self._build_mappings(train_df, catalog_items)

        # BPR trains on positive interactions (rating >= 4.0) vs. unobserved items
        pos_df = train_df[train_df["rating"] >= self.positive_threshold]

        u_indices = pos_df["user_id"].map(self.user_to_idx).values
        i_indices = pos_df["item_id"].map(self.item_to_idx).values
        values = np.ones(len(pos_df), dtype=np.float32)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        user_item_matrix = csr_matrix((values, (u_indices, i_indices)), shape=(n_users, n_items))

        model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,
            random_state=self.random_seed,
        )
        model.fit(user_item_matrix, show_progress=False)

        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def _predict_scores(self, u_idx: int) -> np.ndarray:
        """Computes ranking scores via dot product of learned user and item factors."""
        return self.user_factors[u_idx] @ self.item_factors.T
