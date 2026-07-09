"""
Hyperparameter search spaces for random search (Sec. 3.3).

The paper draws its spaces from Hyperopt-Sklearn and Gorishniy et al. (2021), listed
in Appendix A.3 and the original repo (github.com/LeoGrin/tabular-benchmark). Those
exact tables are not in the PDF body, so the spaces below are reasonable, widely-used
ranges for each model. They are intentionally kept in one place so you can later swap
in the paper's exact grids without touching the search loop.

`sample_params(name, rng)` returns one random configuration. The random-search loop is
responsible for using the model's *defaults* on the first iteration.
"""

from __future__ import annotations

import numpy as np


def _choice(rng: np.random.Generator, seq):
    """Pick one element, preserving its exact Python type (avoids numpy dtype coercion)."""
    return seq[int(rng.integers(len(seq)))]


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample on a log scale (for learning rates, regularization, etc.)."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_params(name: str, rng: np.random.Generator) -> dict:
    """Draw one random hyperparameter configuration for the named model."""
    if name == "RandomForest":
        return {
            "n_estimators": _choice(rng, [100, 200, 300, 500]),
            "max_depth": _choice(rng, [None, 5, 10, 20, 30]),
            "max_features": _choice(rng, ["sqrt", "log2", 0.5, 0.75, 1.0]),
            "min_samples_split": _choice(rng, [2, 5, 10]),
            "min_samples_leaf": _choice(rng, [1, 2, 5]),
            "bootstrap": _choice(rng, [True, False]),
        }
    if name == "GradientBoosting":
        return {
            "n_estimators": _choice(rng, [100, 200, 300]),
            "learning_rate": _loguniform(rng, 0.01, 0.3),
            "max_depth": _choice(rng, [2, 3, 4, 5]),
            "subsample": _choice(rng, [0.6, 0.8, 1.0]),
            "min_samples_leaf": _choice(rng, [1, 5, 10]),
        }
    if name == "HistGradientBoosting":
        return {
            "learning_rate": _loguniform(rng, 0.01, 0.3),
            "max_iter": _choice(rng, [100, 200, 300]),
            "max_leaf_nodes": _choice(rng, [15, 31, 63]),
            "max_depth": _choice(rng, [None, 5, 10]),
            "min_samples_leaf": _choice(rng, [10, 20, 50]),
            "l2_regularization": _choice(rng, [0.0, 0.1, 1.0]),
        }
    if name == "XGBoost":
        return {
            "n_estimators": _choice(rng, [100, 200, 300]),
            "learning_rate": _loguniform(rng, 0.01, 0.3),
            "max_depth": _choice(rng, [3, 5, 7, 9]),
            "subsample": _choice(rng, [0.6, 0.8, 1.0]),
            "colsample_bytree": _choice(rng, [0.6, 0.8, 1.0]),
            "min_child_weight": _choice(rng, [1, 3, 5]),
            "reg_lambda": _loguniform(rng, 1.0, 4.0),
        }
    if name == "MLP":
        return {
            "n_layers": _choice(rng, [1, 2, 3, 4, 5, 8]),
            "width": _choice(rng, [64, 128, 256, 512]),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "learning_rate": _loguniform(rng, 1e-4, 1e-2),
            "weight_decay": _loguniform(rng, 1e-6, 1e-3),
            "batch_size": _choice(rng, [128, 256, 512]),
        }
    if name == "ResNet":
        return {
            "n_blocks": _choice(rng, [1, 2, 3, 4, 6, 8]),
            "d": _choice(rng, [64, 128, 256]),
            "d_hidden": _choice(rng, [128, 256, 512]),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "learning_rate": _loguniform(rng, 1e-4, 1e-2),
            "weight_decay": _loguniform(rng, 1e-6, 1e-3),
            "batch_size": _choice(rng, [128, 256, 512]),
        }
    raise ValueError(f"No search space defined for model: {name}")
