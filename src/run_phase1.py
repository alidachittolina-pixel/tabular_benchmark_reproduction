"""
Phase 1: end-to-end smoke test of the pipeline.

Loads one numerical-only classification dataset, applies the paper's preprocessing,
trains a *default* RandomForest, and reports test accuracy. This is the "it runs"
milestone, not a faithful benchmark result yet (no hyperparameter search, no
normalization, single dataset).

Usage:
    python src/run_phase1.py                      # real dataset from OpenML
    python src/run_phase1.py --synthetic          # offline, no network needed
    python src/run_phase1.py --name electricity --version 1
"""

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data import (
    load_openml_classification,
    make_synthetic_classification,
    preprocess_numerical_classification,
    split_and_truncate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="MagicTelescope",
                        help="OpenML dataset name")
    parser.add_argument("--version", type=int, default=1,
                        help="OpenML dataset version")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the offline synthetic dataset instead of OpenML")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synthetic:
        print("Loading synthetic dataset (offline)...")
        X, y = make_synthetic_classification(seed=args.seed)
    else:
        print(f"Loading '{args.name}' (v{args.version}) from OpenML...")
        X, y = load_openml_classification(args.name, args.version)

    print(f"  raw shape: {X.shape}, target classes: {sorted(y.unique().tolist())}")

    X, y = preprocess_numerical_classification(X, y, seed=args.seed)
    print(f"  after preprocessing: {X.shape}, "
          f"class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = split_and_truncate(X, y, seed=args.seed)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    # Default hyperparameters on purpose — tuning is Phase 3.
    clf = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\nDefault RandomForest test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
