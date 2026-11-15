import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from part2.models.base import BaseRecommender

# Latent-Factor Collaborative Filtering with Biases and SGD

class MFSGDRecommender(BaseRecommender):
    """Biased Matrix Factorization trained via SGD using the Surprise library."""

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 25,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.global_mean = 0.0
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int], **kwargs):
        """Fits Surprise SVD and aligns learned biases/factors with our catalog indices."""
        self._build_mappings(train_df, catalog_items)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
        trainset = data.build_full_trainset()

        model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=True,
            random_state=self.random_seed,
        )
        model.fit(trainset)

        # Extract learned parameters into NumPy arrays aligned with self.user_to_idx and self.item_to_idx
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        self.global_mean = float(trainset.global_mean)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        self.P = np.zeros((n_users, self.n_factors), dtype=np.float32)
        self.Q = np.zeros((n_items, self.n_factors), dtype=np.float32)

        for inner_u in trainset.all_users():
            raw_u = trainset.to_raw_uid(inner_u)
            if raw_u in self.user_to_idx:
                u_idx = self.user_to_idx[raw_u]
                self.b_u[u_idx] = model.bu[inner_u]
                self.P[u_idx] = model.pu[inner_u]

        for inner_i in trainset.all_items():
            raw_i = trainset.to_raw_iid(inner_i)
            if raw_i in self.item_to_idx:
                i_idx = self.item_to_idx[raw_i]
                self.b_i[i_idx] = model.bi[inner_i]
                self.Q[i_idx] = model.qi[inner_i]

    def _predict_scores(self, u_idx: int) -> np.ndarray:
        """Computes predicted ratings r_hat = mu + b_u + b_i + p_u . q_i across all catalog items."""
        return self.global_mean + self.b_u[u_idx] + self.b_i + (self.P[u_idx] @ self.Q.T)
