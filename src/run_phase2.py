"""
Phase 2: run all tree-based models through the same pipeline.

Loads one dataset, preprocesses and splits it once, then trains each default
tree-based model on the identical train/test split and reports test accuracy and
training time. Still not a faithful benchmark result (single dataset, default
hyperparameters, raw accuracy, no cross-dataset normalization) -- those pieces
arrive in Phases 3-4. Training time is shown because the paper stresses that
tree-based models reach good scores far more cheaply than deep models.

Usage:
    python src/run_phase2.py --synthetic
    python src/run_phase2.py                       # MagicTelescope from OpenML
    python src/run_phase2.py --name electricity --version 1
"""

import argparse
import time

from sklearn.metrics import accuracy_score

from data import (
    load_openml_classification,
    make_synthetic_classification,
    preprocess_numerical_classification,
    split_and_truncate,
)
from models import get_tree_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="MagicTelescope")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synthetic:
        print("Loading synthetic dataset (offline)...")
        X, y = make_synthetic_classification(seed=args.seed)
    else:
        print(f"Loading '{args.name}' (v{args.version}) from OpenML...")
        X, y = load_openml_classification(args.name, args.version)

    X, y = preprocess_numerical_classification(X, y, seed=args.seed)
    X_train, X_test, y_train, y_test = split_and_truncate(X, y, seed=args.seed)
    print(f"train: {X_train.shape}, test: {X_test.shape}\n")

    results = []
    for name, model in get_tree_models(seed=args.seed).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append((name, acc, fit_time))
        print(f"  {name:<22} acc={acc:.4f}  ({fit_time:.2f}s)")

    # Comparison table, best accuracy first.
    results.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 46)
    print(f"{'Model':<22}{'Accuracy':>12}{'Train (s)':>12}")
    print("-" * 46)
    for name, acc, fit_time in results:
        print(f"{name:<22}{acc:>12.4f}{fit_time:>12.2f}")
    print("=" * 46)


if __name__ == "__main__":
    main()
