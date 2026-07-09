"""
Phase 5 (Road B): tree-vs-deep comparison.

Reuses the entire Phase 4 pipeline (benchmark -> normalize -> shuffle -> aggregate ->
plot) but defaults to running every available model, tree and deep, so the resulting
Figure-1-style curve directly compares tree-based models against MLP / ResNet. If the
paper's finding reproduces, the tree curves sit above the deep ones.

Requires PyTorch for the deep models. If torch is not installed, MLP/ResNet are simply
absent and this reduces to the Phase 4 tree comparison.

Usage:
    # Machinery check (needs torch); keep it tiny, deep models are slow on CPU:
    python src/run_phase5.py --synthetic 2 --models MLP --n-iter 3 --n-shuffles 5

    # Real tree-vs-deep run (GPU strongly recommended for the deep models):
    python src/run_phase5.py --datasets MagicTelescope electricity phoneme \
        --models RandomForest XGBoost MLP ResNet --n-iter 30
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
from models import ALL_MODEL_NAMES, DEEP_MODEL_NAMES, TREE_MODEL_NAMES
from run_phase4 import DEFAULT_DATASET_VERSIONS, build_synthetic_suite


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
    parser.add_argument("--datasets", nargs="+", default=["MagicTelescope"])
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"default runs all available: {ALL_MODEL_NAMES}")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-shuffles", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    models = args.models or ALL_MODEL_NAMES
    if not DEEP_MODEL_NAMES:
        print("NOTE: PyTorch not found -> deep models unavailable; "
              "running tree models only.\n")

    if args.synthetic:
        print(f"Building {args.synthetic} synthetic datasets (offline)...")
        dataset_items = build_synthetic_suite(args.synthetic, args.seed)
    else:
        dataset_items = build_openml_suite(args.datasets)

    print(f"\nBenchmark: {len(dataset_items)} datasets x {len(models)} models "
          f"x {args.n_iter} iterations")
    print(f"  tree: {[m for m in models if m in TREE_MODEL_NAMES]}")
    print(f"  deep: {[m for m in models if m in DEEP_MODEL_NAMES]}\n")

    raw, raw_path = run_benchmark(
        dataset_items, models, args.n_iter, args.seed, out_dir=args.out_dir,
        raw_filename="phase5_raw_records.csv",
    )
    print(f"Raw records saved to: {raw_path}\n")

    raw_norm = add_normalized_scores(raw, task="classification")
    curves = compute_shuffled_curves(raw_norm, n_shuffles=args.n_shuffles, seed=args.seed)
    agg = aggregate_curves(curves)

    fig_path = f"{args.out_dir}/figure1_tree_vs_deep.png"
    plot_curves(agg, fig_path, task="classification")

    final = (agg[agg["iteration"] == agg["iteration"].max()]
             .sort_values("mean", ascending=False))
    print("=" * 56)
    print(f"Final normalized score (@ {int(agg['iteration'].max())} iters), "
          f"averaged over {len(dataset_items)} datasets")
    print("-" * 56)
    for _, row in final.iterrows():
        kind = "deep" if row["model"] in DEEP_MODEL_NAMES else "tree"
        print(f"{row['model']:<22}{kind:>6}{row['mean']:>10.3f}"
              f"   [{row['min']:.3f}, {row['max']:.3f}]")
    print("=" * 56)
    print(f"\nFigure saved to: {fig_path}")


if __name__ == "__main__":
    main()
