"""
Model registry.

Covers the paper's tree-based models (Sec. 4.1). `get_tree_models` returns them with
default hyperparameters (Phase 2); `build_model` constructs any of them from a dict of
sampled hyperparameters (Phase 3 random search).

The paper uses:
  - RandomForest
  - GradientBoostingTrees for the numerical-only setting; HistGradientBoosting when
    categorical features are present (the fast, modern equivalent)
  - XGBoost

XGBoost is optional: it registers only if it imports cleanly.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # ImportError, or the libomp/OpenMP load error on some macOS setups
    _HAS_XGB = False

# XGBoost is registered only if it imports cleanly, so a broken install (e.g. the
# macOS libomp issue) degrades gracefully instead of crashing the whole run.
TREE_MODEL_NAMES = ["RandomForest", "GradientBoosting", "HistGradientBoosting"]
if _HAS_XGB:
    TREE_MODEL_NAMES.append("XGBoost")

# Deep models (Phase 5) are optional and require PyTorch.
try:
    from nn_models import TorchClassifier, TORCH_AVAILABLE
except Exception:
    TORCH_AVAILABLE = False

DEEP_MODEL_NAMES = ["MLP", "ResNet"] if TORCH_AVAILABLE else []
ALL_MODEL_NAMES = TREE_MODEL_NAMES + DEEP_MODEL_NAMES


def build_model(name: str, params: dict | None = None, seed: int = 0):
    """Construct a classifier by name, applying sampled hyperparameters over defaults.

    An empty params dict yields the model's defaults (used as the first random-search
    iteration, per Sec. 3.3).
    """
    params = dict(params or {})
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if name == "XGBoost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not available in this environment")
        return XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric="logloss", **params
        )
    if name == "MLP":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available in this environment")
        return TorchClassifier(module="mlp", seed=seed, **params)
    if name == "ResNet":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available in this environment")
        return TorchClassifier(module="resnet", seed=seed, **params)
    raise ValueError(f"Unknown model: {name}")


def get_tree_models(seed: int = 0) -> dict:
    """Return the available tree-based classifiers with default hyperparameters."""
    return {name: build_model(name, seed=seed) for name in TREE_MODEL_NAMES}
