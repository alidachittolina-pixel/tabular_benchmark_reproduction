"""
Phase 3: hyperparameter search with the paper's protocol.

For one dataset, split into train/validation/test, then run random search for each
tree-based model. Report each model's default score (iteration 1) next to its tuned
score (test score of the best-on-validation config after the full budget). The full
per-iteration records are saved to CSV, which is what Phase 4 will aggregate and plot.

Usage:
    python src/run_phase3.py --synthetic --n-iter 15
    python src/run_phase3.py --n-iter 30                  # MagicTelescope from OpenML
    python src/run_phase3.py --models RandomForest XGBoost --n-iter 30

Note on budget: the paper uses ~400 iterations. Start small (15-30) to keep runs quick;
GradientBoosting in particular is slow, so raise the budget once you trust the pipeline.
"""

import argparse
import json
import os
import time

import pandas as pd

from data import (
    load_openml_classification,
    make_synthetic_classification,
    preprocess_numerical_classification,
    split_train_val_test,
)
from models import TREE_MODEL_NAMES
from random_search import run_random_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="MagicTelescope")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=20,
                        help="random-search iterations per model (incl. defaults)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"subset of {TREE_MODEL_NAMES}; default runs all available")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if args.synthetic:
        print("Loading synthetic dataset (offline)...")
        X, y = make_synthetic_classification(seed=args.seed)
        dataset_label = "synthetic"
    else:
        print(f"Loading '{args.name}' (v{args.version}) from OpenML...")
        X, y = load_openml_classification(args.name, args.version)
        dataset_label = args.name

    X, y = preprocess_numerical_classification(X, y, seed=args.seed)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_train_val_test(X, y, seed=args.seed)
    print(f"train: {X_tr.shape}, val: {X_val.shape}, test: {X_te.shape}\n")

    models_to_run = args.models or TREE_MODEL_NAMES
    all_records = []
    summary = []

    for name in models_to_run:
        print(f"Random search: {name} ({args.n_iter} iterations)...")
        t0 = time.perf_counter()
        df = run_random_search(
            name, X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_iter=args.n_iter, seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        all_records.append(df)

        default_test = df.loc[df["iteration"] == 1, "test_score"].iloc[0]
        tuned_test = df["test_at_best_val"].iloc[-1]
        summary.append((name, default_test, tuned_test, elapsed))
        print(f"  default={default_test:.4f}  tuned={tuned_test:.4f}  ({elapsed:.1f}s)\n")

    # Save raw records for Phase 4.
    os.makedirs(args.out_dir, exist_ok=True)
    out = pd.concat(all_records, ignore_index=True)
    out["params"] = out["params"].apply(lambda p: json.dumps(p, default=str))
    out_path = os.path.join(args.out_dir, f"phase3_{dataset_label}.csv")
    out.to_csv(out_path, index=False)

    # Summary table.
    print("=" * 60)
    print(f"{'Model':<22}{'Default':>11}{'Tuned':>11}{'Search (s)':>14}")
    print("-" * 60)
    for name, d, t, e in sorted(summary, key=lambda r: r[2], reverse=True):
        print(f"{name:<22}{d:>11.4f}{t:>11.4f}{e:>14.1f}")
    print("=" * 60)
    print(f"\nRaw per-iteration records saved to: {out_path}")


if __name__ == "__main__":
    main()
