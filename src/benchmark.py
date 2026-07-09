"""
Phase 4: aggregate across datasets and reproduce the Figure 1 curves.

Pipeline:
  1. For each dataset and model, run random search once (reusing Phase 3), recording
     every configuration's validation and test score.
  2. Normalize test scores *per dataset* (Sec. 3.4): affine rescale between the best
     score and a robust low anchor (the 10% quantile for classification, 50% for
     regression), then clip to [0, 1].
  3. Reproduce the "best-on-validation up to iteration k" curve, and do it for many
     random-search orderings by shuffling the saved results post-hoc (no refitting).
  4. Average across datasets; take mean over shuffles as the line and min/max as the
     ribbon (as in Figure 1).

The heavy step is (1); (2)-(4) are cheap arithmetic on the saved numbers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data import preprocess_numerical_classification, split_train_val_test
from random_search import run_random_search


# --------------------------------------------------------------------------- #
# 1. Run the benchmark (the expensive part)
# --------------------------------------------------------------------------- #
def benchmark_dataset(label, X_raw, y_raw, models, n_iter, seed):
    """Preprocess/split one dataset and run random search for each model."""
    X, y = preprocess_numerical_classification(X_raw, y_raw, seed=seed)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_train_val_test(X, y, seed=seed)
    frames = []
    for name in models:
        df = run_random_search(
            name, X_tr, y_tr, X_val, y_val, X_te, y_te, n_iter=n_iter, seed=seed
        )
        df["dataset"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_benchmark(dataset_items, models, n_iter, seed, out_dir="results",
                  raw_filename="benchmark_raw.csv"):
    """Run the benchmark over several datasets and save the combined raw records.

    `dataset_items` is a list of (label, X_raw, y_raw) tuples. `raw_filename` lets each
    experiment (phase 4, phase 5, ablation) write to its own file.
    """
    all_frames = []
    for label, X_raw, y_raw in dataset_items:
        print(f"  dataset '{label}'...")
        all_frames.append(benchmark_dataset(label, X_raw, y_raw, models, n_iter, seed))
    raw = pd.concat(all_frames, ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    # Filename is configurable so Phase 4 / Phase 5 / ablation runs don't overwrite
    # each other's raw records.
    raw_path = os.path.join(out_dir, raw_filename)
    cols = ["dataset", "model", "iteration", "is_default", "val_score", "test_score"]
    for extra in ("test_f1", "fit_time", "params"):   # keep Step 6 reporting columns
        if extra in raw.columns:
            cols.append(extra)
    raw[cols].to_csv(raw_path, index=False)
    return raw, raw_path


# --------------------------------------------------------------------------- #
# 2. Per-dataset normalization (Sec. 3.4)
# --------------------------------------------------------------------------- #
def add_normalized_scores(raw: pd.DataFrame, task: str = "classification") -> pd.DataFrame:
    """Add a `test_norm` column, normalized within each dataset.

    Top anchor = best test score on the dataset (across all models/configs).
    Low anchor = a robust bad score: the 10% (classification) / 50% (regression)
    quantile of test scores. Values are clipped to [0, 1].
    """
    q = 0.10 if task == "classification" else 0.50
    raw = raw.copy()
    raw["test_norm"] = np.nan
    for dataset, grp in raw.groupby("dataset"):
        top = grp["test_score"].max()
        bottom = grp["test_score"].quantile(q)
        denom = top - bottom
        if denom <= 0:  # degenerate: all scores equal
            norm = np.ones(len(grp))
        else:
            norm = (grp["test_score"] - bottom) / denom
        raw.loc[grp.index, "test_norm"] = np.clip(norm, 0.0, 1.0)
    return raw


# --------------------------------------------------------------------------- #
# 3. Best-on-validation curves, shuffled post-hoc
# --------------------------------------------------------------------------- #
def _best_so_far_curve(val: np.ndarray, test_norm: np.ndarray) -> np.ndarray:
    """Test score of the config that is best on validation among the first k seen."""
    curve = np.empty(len(val))
    best_val, best_test = -np.inf, np.nan
    for k in range(len(val)):
        if val[k] > best_val:
            best_val, best_test = val[k], test_norm[k]
        curve[k] = best_test
    return curve


def compute_shuffled_curves(raw_norm: pd.DataFrame, n_shuffles: int = 15,
                            seed: int = 0) -> pd.DataFrame:
    """For each (model, dataset, shuffle), the normalized best-on-val test curve.

    Returns a long DataFrame: model, dataset, shuffle, iteration, norm_score.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (model, dataset), grp in raw_norm.groupby(["model", "dataset"]):
        grp = grp.sort_values("iteration")
        val = grp["val_score"].to_numpy()
        test_norm = grp["test_norm"].to_numpy()
        n = len(val)
        for s in range(n_shuffles):
            order = rng.permutation(n)
            curve = _best_so_far_curve(val[order], test_norm[order])
            for k in range(n):
                rows.append((model, dataset, s, k + 1, curve[k]))
    return pd.DataFrame(rows, columns=["model", "dataset", "shuffle",
                                       "iteration", "norm_score"])


# --------------------------------------------------------------------------- #
# 4. Aggregate across datasets and shuffles
# --------------------------------------------------------------------------- #
def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Average across datasets, then summarize over shuffles (mean, min, max)."""
    # Average across datasets for each (model, shuffle, iteration).
    per_shuffle = (curves.groupby(["model", "shuffle", "iteration"])["norm_score"]
                   .mean().reset_index())
    # Summarize across shuffles.
    agg = (per_shuffle.groupby(["model", "iteration"])["norm_score"]
           .agg(["mean", "min", "max"]).reset_index())
    return agg


def plot_curves(agg: pd.DataFrame, out_path: str, task: str = "classification"):
    """Reproduce the Figure 1 style: normalized score vs. search iterations (log x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylabel = ("Normalized test accuracy" if task == "classification"
              else "Normalized test R2")
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in agg.groupby("model"):
        grp = grp.sort_values("iteration")
        ax.plot(grp["iteration"], grp["mean"], label=model, linewidth=2)
        ax.fill_between(grp["iteration"], grp["min"], grp["max"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Number of random search iterations")
    ax.set_ylabel(f"{ylabel} of best model (on valid set)")
    ax.set_title("Tree-based models on tabular data (reproduction)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
