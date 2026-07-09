"""
Phase 4: run the multi-dataset benchmark and produce the Figure 1 reproduction.

Usage:
    # Offline machinery check on a suite of synthetic datasets:
    python src/run_phase4.py --synthetic 3 --n-iter 15 --n-shuffles 15

    # Real run on a handful of the paper's numerical classification datasets:
    python src/run_phase4.py --datasets MagicTelescope electricity phoneme --n-iter 40

Budget guidance: the paper uses ~400 iterations and 45 datasets. For a portfolio-scale
reproduction, 3-5 datasets at 40-100 iterations already shows the trend. GradientBoosting
is slow; consider --models RandomForest HistGradientBoosting XGBoost to keep runtime sane.
"""

import argparse

from benchmark import (
    add_normalized_scores,
    aggregate_curves,
    compute_shuffled_curves,
    plot_curves,
    run_benchmark,
)
from data import load_openml_classification, make_synthetic_classification
from models import TREE_MODEL_NAMES

# Sensible OpenML versions for a few numerical-only binary datasets from the paper's
# benchmark. Adjust versions if OpenML resolves a different default.
DEFAULT_DATASET_VERSIONS = {
    "MagicTelescope": 1,
    "electricity": 1,
    "phoneme": 1,
    "bank-marketing": 1,
    "MiniBooNE": 1,
}


def build_synthetic_suite(n_datasets, seed):
    """A few synthetic datasets of varying difficulty, for an offline machinery check."""
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
    parser.add_argument("--synthetic", type=int, metavar="N", default=0,
                        help="use N synthetic datasets instead of OpenML")
    parser.add_argument("--datasets", nargs="+", default=["MagicTelescope"],
                        help="OpenML dataset names (ignored if --synthetic is set)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"subset of {TREE_MODEL_NAMES}; default runs all available")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-shuffles", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    models = args.models or TREE_MODEL_NAMES

    if args.synthetic:
        print(f"Building {args.synthetic} synthetic datasets (offline)...")
        dataset_items = build_synthetic_suite(args.synthetic, args.seed)
    else:
        dataset_items = build_openml_suite(args.datasets)

    print(f"\nRunning benchmark: {len(dataset_items)} datasets x {len(models)} models "
          f"x {args.n_iter} iterations")
    raw, raw_path = run_benchmark(
        dataset_items, models, args.n_iter, args.seed, out_dir=args.out_dir,
        raw_filename="phase4_raw_records.csv",
    )
    print(f"Raw records saved to: {raw_path}\n")

    raw_norm = add_normalized_scores(raw, task="classification")
    curves = compute_shuffled_curves(raw_norm, n_shuffles=args.n_shuffles, seed=args.seed)
    agg = aggregate_curves(curves)

    fig_path = f"{args.out_dir}/figure1_repro.png"
    plot_curves(agg, fig_path, task="classification")

    # Final normalized score per model (last iteration), best first.
    final = (agg[agg["iteration"] == agg["iteration"].max()]
             .sort_values("mean", ascending=False))
    print("=" * 52)
    print(f"Final normalized score (@ {int(agg['iteration'].max())} iters), "
          f"averaged over {len(dataset_items)} datasets")
    print("-" * 52)
    for _, row in final.iterrows():
        print(f"{row['model']:<24}{row['mean']:>10.3f}"
              f"   [{row['min']:.3f}, {row['max']:.3f}]")
    print("=" * 52)
    print(f"\nFigure saved to: {fig_path}")


if __name__ == "__main__":
    main()
