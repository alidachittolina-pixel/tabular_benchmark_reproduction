"""
Deep models for the tabular benchmark (Phase 5): MLP and ResNet.

Binary classifiers (single logit + BCEWithLogitsLoss) wrapped in a scikit-learn-style
TorchClassifier so they slot into the same random-search loop as the tree models.

Paper-faithfulness fix (Sec. 3.5): features for neural nets are Gaussianized with
scikit-learn's QuantileTransformer(output_distribution="normal"), not merely
standardized. QuantileTransformer maps each feature to a normal distribution, taming
heavy tails and outliers that are common in tabular data and that hurt MLPs -- this is
what the paper specifies.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import QuantileTransformer

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class MLP(nn.Module):
    def __init__(self, input_dim, n_layers=2, width=128, dropout=0.1):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = width

        layers.append(nn.Linear(current_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class ResNetBlock(nn.Module):
    def __init__(self, d, d_hidden, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_dim, n_blocks=2, d=128, d_hidden=256, dropout=0.1):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, d)
        self.blocks = nn.Sequential(
            *[ResNetBlock(d, d_hidden, dropout) for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(d, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x).squeeze(1)


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        module="mlp",
        seed=0,
        n_layers=2,
        width=128,
        n_blocks=2,
        d=128,
        d_hidden=256,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=30,
    ):
        self.module = module
        self.seed = seed
        self.n_layers = n_layers
        self.width = width
        self.n_blocks = n_blocks
        self.d = d
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Gaussianize features (Sec. 3.5). n_quantiles must not exceed n_samples.
        self.scaler_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, X.shape[0]),
            random_state=self.seed,
        )
        X = self.scaler_.fit_transform(X).astype(np.float32)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]

        if self.module == "mlp":
            self.model_ = MLP(
                input_dim=input_dim,
                n_layers=self.n_layers,
                width=self.width,
                dropout=self.dropout,
            )
        elif self.module == "resnet":
            self.model_ = ResNet(
                input_dim=input_dim,
                n_blocks=self.n_blocks,
                d=self.d,
                d_hidden=self.d_hidden,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown module: {self.module}")

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_fn = nn.BCEWithLogitsLoss()

        self.model_.train()

        for _ in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self.scaler_.transform(X).astype(np.float32)

        X_tensor = torch.tensor(X)

        self.model_.eval()

        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba_1 = torch.sigmoid(logits).numpy()

        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
