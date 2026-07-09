"""
Extension runner: noise-feature ablation (Finding 2).

Defaults match the Phase 5 table (MagicTelescope + phoneme; HistGradientBoosting,
XGBoost, MLP) so results are directly comparable.

Usage:
    # Offline machinery check (trees only; no torch needed):
    python src/run_ablation.py --synthetic 2 --models RandomForest XGBoost \
        --k-values 0 10 20 --n-iter 5

    # Real run matching Phase 5 (needs torch for the MLP; slow on CPU):
    python src/run_ablation.py --datasets MagicTelescope phoneme \
        --models HistGradientBoosting XGBoost MLP --k-values 0 5 10 20 40 --n-iter 15

    # Add error bands with multiple seeds (multiplies runtime):
    python src/run_ablation.py --seeds 0 1 2 ...

Runtime note: the MLP retrains for every (k, iteration, dataset, seed), so it dominates
cost. Start with a single seed and a short --n-iter, then scale up.
"""

import argparse

from ablation import degradation_table, plot_ablation, run_ablation
from data import load_openml_classification, make_synthetic_classification
from models import ALL_MODEL_NAMES, DEEP_MODEL_NAMES, TREE_MODEL_NAMES
from run_phase4 import DEFAULT_DATASET_VERSIONS


def build_synthetic_suite(n_datasets, seed):
    items = []
    for i in range(n_datasets):
        X, y = make_synthetic_classification(n_samples=8000, seed=seed + i)
        items.append((f"synthetic_{i}", X, y))
    return items


def build_openml_suite(names):
    items = []
    for name in names:
        version = DEFAULT_DATASET_VERSIONS.get(name, 1)
        print(f"Loading '{name}' (v{version}) from OpenML...")
        X, y = load_openml_classification(name, version)
        items.append((name, X, y))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, metavar="N", default=0)
    parser.add_argument("--datasets", nargs="+", default=["MagicTelescope", "phoneme"])
    parser.add_argument("--models", nargs="+",
                        default=["HistGradientBoosting", "XGBoost", "MLP"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 5, 10, 20, 40])
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    # Drop deep models silently if torch is unavailable.
    models = [m for m in args.models
              if m in TREE_MODEL_NAMES or m in DEEP_MODEL_NAMES]
    dropped = [m for m in args.models if m not in models]
    if dropped:
        print(f"NOTE: {dropped} unavailable (missing dependency) -> skipped.\n")

    if args.synthetic:
        print(f"Building {args.synthetic} synthetic datasets (offline)...")
        dataset_items = build_synthetic_suite(args.synthetic, seed=0)
    else:
        dataset_items = build_openml_suite(args.datasets)

    print(f"\nAblation: {len(dataset_items)} datasets x {len(models)} models "
          f"x k={args.k_values} x seeds={args.seeds}\n")

    results = run_ablation(dataset_items, models, args.k_values,
                           n_iter=args.n_iter, seeds=args.seeds, out_dir=args.out_dir)

    fig_path = f"{args.out_dir}/ablation_noise.png"
    plot_ablation(results, fig_path)

    print("\n" + "=" * 56)
    print("Accuracy drop from fewest to most noise features "
          "(averaged over seeds & datasets)")
    print("-" * 56)
    print(degradation_table(results).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 56)
    print(f"\nResults CSV: {args.out_dir}/ablation_results.csv")
    print(f"Figure:      {fig_path}")


if __name__ == "__main__":
    main()
