"""
Extension: robustness to uninformative features (reproduces Finding 2, Fig. 4b).

The paper argues MLP-like networks degrade much faster than tree-based models as
uninformative features are added, while trees are largely unaffected. Here we test that
directly: for k = 0, 5, 10, 20, 40, we append k standard-Gaussian noise columns
(uncorrelated with the target and each other) to each dataset, re-run the exact same
benchmark protocol, and record the raw test accuracy / F1 of the validation-selected
model. Plotting accuracy vs. k shows each model's degradation slope.

This changes nothing about the methodology (same preprocessing, random search,
select-on-validation). It only adds noise features as a new experimental axis. Because
row selection and splits are seeded independently of the feature columns, the underlying
rows are identical across all k -- only the noise columns differ.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import run_benchmark


def add_noise_features(X: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Append k standard-normal columns to a copy of X (uninformative features)."""
    if k <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(len(X), k)).astype(float)
    noise_df = pd.DataFrame(
        noise, columns=[f"noise_{i}" for i in range(k)], index=X.index
    )
    return pd.concat([X.copy(), noise_df], axis=1)


def _best_per_dataset_model(raw: pd.DataFrame) -> pd.DataFrame:
    """Row of the best-on-validation config for each (dataset, model)."""
    idx = raw.groupby(["dataset", "model"])["val_score"].idxmax()
    return raw.loc[idx]


def run_ablation(dataset_items, models, k_values, n_iter, seeds, out_dir="results"):
    """Run the noise-feature ablation and return a tidy results table.

    Returns rows of [model, k, seed, mean_accuracy, mean_f1] where the means are over
    the datasets in `dataset_items`.
    """
    rows = []
    for seed in seeds:
        for k in k_values:
            print(f"[seed {seed}] k={k} noise features")
            noised = []
            for di, (label, X, y) in enumerate(dataset_items):
                # Independent-but-deterministic noise per (seed, dataset, k).
                Xk = add_noise_features(X, k, seed=seed * 10_000 + di * 100 + k)
                noised.append((label, Xk, y))

            raw, _ = run_benchmark(
                noised, models, n_iter=n_iter, seed=seed, out_dir=out_dir,
                raw_filename=f"ablation_raw_k{k}_seed{seed}.csv",
            )
            best = _best_per_dataset_model(raw)

            agg_dict = {"mean_accuracy": ("test_score", "mean")}
            if "test_f1" in best.columns:
                agg_dict["mean_f1"] = ("test_f1", "mean")
            agg = best.groupby("model").agg(**agg_dict).reset_index()
            agg["k"] = k
            agg["seed"] = seed
            rows.append(agg)

    results = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    return results


def plot_ablation(results: pd.DataFrame, out_path: str, metric: str = "mean_accuracy"):
    """Plot metric vs. number of noise features, one line per model.

    With multiple seeds, a shaded min/max band across seeds is drawn.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, g in results.groupby("model"):
        stats = g.groupby("k")[metric].agg(["mean", "min", "max"]).reset_index()
        ax.plot(stats["k"], stats["mean"], marker="o", linewidth=2, label=model)
        if g["seed"].nunique() > 1:
            ax.fill_between(stats["k"], stats["min"], stats["max"], alpha=0.15)
    ax.set_xlabel("Number of added uninformative (noise) features")
    ax.set_ylabel("Test accuracy (mean over datasets)"
                  if metric == "mean_accuracy" else metric)
    ax.set_title("Robustness to uninformative features (Finding 2 reproduction)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def degradation_table(results: pd.DataFrame, metric: str = "mean_accuracy"):
    """Per model: metric at k=0 vs k=max, and the drop (averaged over seeds)."""
    m = results.groupby(["model", "k"])[metric].mean().reset_index()
    kmin, kmax = int(m["k"].min()), int(m["k"].max())
    out = []
    for model, g in m.groupby("model"):
        a0 = g.loc[g["k"] == kmin, metric].iloc[0]
        a1 = g.loc[g["k"] == kmax, metric].iloc[0]
        out.append((model, a0, a1, a0 - a1))
    return (pd.DataFrame(out, columns=["model", f"k={kmin}", f"k={kmax}", "drop"])
            .sort_values("drop").reset_index(drop=True))
