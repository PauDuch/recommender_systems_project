import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from part2.models.base import BaseRecommender

# Neural Collaborative Filtering with Multinomial Variational Autoencoder (Mult-VAE)

class _MultVAENetwork(nn.Module):
    """PyTorch Mult-VAE architecture: Encoder -> Gaussian Latent z -> Multinomial Decoder."""

    def __init__(self, n_items: int, hidden_dim: int = 600, latent_dim: int = 200, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_enc1 = nn.Linear(n_items, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, n_items)

    def encode(self, x: torch.Tensor):
        h = F.normalize(x, p=2, dim=1)
        h = self.dropout(h)
        h = torch.tanh(self.fc_enc1(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc_dec1(z))
        return self.fc_dec2(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


class NeuralVAERecommender(BaseRecommender):
    """Mult-VAE trained on binary positive user interaction vectors using PyTorch."""

    def __init__(
        self,
        hidden_dim: int = 600,
        latent_dim: int = 200,
        dropout: float = 0.3,
        lr: float = 1e-3,
        beta_max: float = 0.2,
        epochs: int = 30,
        batch_size: int = 256,
        positive_threshold: float = 4.0,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.beta_max = beta_max
        self.epochs = epochs
        self.batch_size = batch_size
        self.positive_threshold = positive_threshold
        self.predicted_scores = None

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int], **kwargs):
        """Trains Mult-VAE on positive user interaction vectors and caches predicted scores."""
        self._build_mappings(train_df, catalog_items)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        # Build dense binary user-item interaction matrix (rating >= positive_threshold)
        X = np.zeros((n_users, n_items), dtype=np.float32)
        pos_df = train_df[train_df["rating"] >= self.positive_threshold]
        u_indices = pos_df["user_id"].map(self.user_to_idx).values
        i_indices = pos_df["item_id"].map(self.item_to_idx).values
        X[u_indices, i_indices] = 1.0

        dataset = TensorDataset(torch.from_numpy(X))
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        model = _MultVAENetwork(
            n_items=n_items,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Train loop with linear KL annealing up to beta_max
        model.train()
        total_steps = max(1, self.epochs * len(loader))
        step = 0
        for _ in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                beta = self.beta_max * min(1.0, step / max(1, total_steps // 2))
                step += 1

                logits, mu, logvar = model(batch_x)
                # Multinomial negative log-likelihood + KL divergence
                log_softmax = F.log_softmax(logits, dim=1)
                recon_loss = -torch.sum(log_softmax * batch_x, dim=1).mean()
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                loss = recon_loss + beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Precompute scores in eval mode (no dropout, deterministic mu)
        model.eval()
        scores_list = []
        eval_loader = DataLoader(dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in eval_loader:
                logits, _, _ = model(batch_x.to(device))
                scores_list.append(logits.cpu().numpy())

        self.predicted_scores = np.vstack(scores_list)

    def _predict_scores(self, u_idx: int) -> np.ndarray:
        """Returns cached Mult-VAE output logits for user index u_idx."""
        return self.predicted_scores[u_idx]
