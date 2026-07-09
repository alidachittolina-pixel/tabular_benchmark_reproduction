"""
Step 6 reporting helpers.

Turns the raw per-iteration records (from run_benchmark / run_random_search) into the
tables the write-up needs: raw test accuracy, macro-F1, training time, and the selected
hyperparameters of the validation-chosen model for each dataset and model.

Usage in a notebook cell:
    from reporting import summarize
    summarize("results/phase5_raw_records.csv")     # path to the saved raw records
    # or pass an in-memory DataFrame returned by run_benchmark:
    # summarize(raw_df)
"""

from __future__ import annotations

import ast

import pandas as pd


def _load(raw) -> pd.DataFrame:
    """Accept a path or a DataFrame; parse the params column back into dicts."""
    df = pd.read_csv(raw) if isinstance(raw, str) else raw.copy()
    if "params" in df.columns:
        def parse(p):
            if isinstance(p, dict):
                return p
            try:
                return ast.literal_eval(p)
            except Exception:
                return {}
        df["params"] = df["params"].apply(parse)
    return df


def best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, model), the row of the best-on-validation configuration."""
    idx = df.groupby(["dataset", "model"])["val_score"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def summarize(raw, decimals: int = 4):
    """Print the Step 6 tables and return the best-config summary DataFrame."""
    df = _load(raw)
    has_f1 = "test_f1" in df.columns
    has_time = "fit_time" in df.columns
    if not (has_f1 and has_time):
        print("NOTE: this raw file predates the accuracy/F1/timing update. "
              "Re-run the benchmark with the updated random_search.py to populate "
              "test_f1 and fit_time.\n")

    best = best_configs(df)

    # --- Table 1: raw accuracy (and F1) per dataset x model -------------------
    print("=" * 70)
    print("Raw test performance of the validation-selected model")
    print("=" * 70)
    cols = ["dataset", "model", "test_score"]
    if has_f1:
        cols.append("test_f1")
    table1 = best[cols].rename(columns={"test_score": "accuracy", "test_f1": "macro_f1"})
    print(table1.to_string(index=False, float_format=lambda x: f"{x:.{decimals}f}"))

    # --- Table 2: averaged across datasets, per model -------------------------
    print("\n" + "=" * 70)
    print("Averaged across datasets, per model")
    print("=" * 70)
    agg_cols = {"test_score": "mean"}
    if has_f1:
        agg_cols["test_f1"] = "mean"
    avg = (best.groupby("model").agg(agg_cols)
           .rename(columns={"test_score": "mean_accuracy", "test_f1": "mean_macro_f1"})
           .sort_values("mean_accuracy", ascending=False))
    print(avg.to_string(float_format=lambda x: f"{x:.{decimals}f}"))

    # --- Table 3: training time -----------------------------------------------
    if has_time:
        print("\n" + "=" * 70)
        print("Training time (seconds)")
        print("=" * 70)
        time_tbl = df.groupby("model")["fit_time"].agg(
            mean_fit="mean", total_search="sum").sort_values("mean_fit")
        print(time_tbl.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- Table 4: selected hyperparameters ------------------------------------
    print("\n" + "=" * 70)
    print("Selected hyperparameters (best on validation)")
    print("=" * 70)
    for _, row in best.iterrows():
        params = row["params"] if isinstance(row["params"], dict) else {}
        default_note = " (defaults)" if not params else ""
        print(f"\n{row['dataset']} / {row['model']}{default_note}")
        for k, v in params.items():
            print(f"    {k}: {v}")

    return best


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/phase5_raw_records.csv"
    summarize(path)
