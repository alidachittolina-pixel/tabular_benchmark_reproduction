"""
Data loading and preprocessing for the tabular benchmark reproduction.

Implements the preprocessing described in Grinsztajn et al. (2022), Sec. 3.2 & 3.5,
for the *numerical-only classification, medium-sized* setting. Each step below is
annotated with the paper rule it corresponds to.

The loader (OpenML) is deliberately separated from the preprocessing so the
preprocessing can be tested offline. A synthetic generator is provided for exactly
that purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split

# Medium-sized regime: training set is truncated to this many samples (Sec. 3.2).
MEDIUM_TRAIN_SIZE = 10_000
# "Remove numerical features with less than 10 unique values" (Sec. 3.2).
MIN_UNIQUE_NUMERICAL = 10
# Drop a column if more than this fraction of its values are missing (Sec. 3.2:
# "first remove columns containing many missing data"). The paper does not fix a
# threshold; 0.5 is a reasonable, documented choice.
MAX_MISSING_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_openml_classification(name: str = "MagicTelescope", version: int = 1):
    """Load a classification dataset from OpenML as (X, y).

    MagicTelescope is one of the numerical-only binary classification datasets in
    the paper's benchmark: ~19k rows, 10 continuous features, target in {g, h}.
    Requires network access to openml.org.
    """
    bunch = fetch_openml(name=name, version=version, as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    return X, y


def make_synthetic_classification(n_samples: int = 12_000, seed: int = 0):
    """Offline stand-in that mimics a raw OpenML frame.

    Intentionally messy so preprocessing branches get exercised: 3 classes (forces
    binarization), some injected missing values, and one low-cardinality numerical
    column (should be dropped).
    """
    rng = np.random.default_rng(seed)
    Xarr, yarr = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=2,
        weights=[0.5, 0.3, 0.2],  # imbalanced, like real data
        random_state=seed,
    )
    X = pd.DataFrame(Xarr, columns=[f"num_{i}" for i in range(Xarr.shape[1])])
    # A low-cardinality numerical column (5 unique values) -> should be removed.
    X["low_card_num"] = rng.integers(0, 5, size=n_samples).astype(float)
    # Inject ~2% missing values into two columns.
    for col in ["num_0", "num_1"]:
        mask = rng.random(n_samples) < 0.02
        X.loc[mask, col] = np.nan
    y = pd.Series(yarr, name="target").astype("category")
    return X, y


# --------------------------------------------------------------------------- #
# Preprocessing steps (each maps to a paper rule)
# --------------------------------------------------------------------------- #
def keep_numerical_only(X: pd.DataFrame) -> pd.DataFrame:
    """Numerical-only setting: drop non-numeric columns (Sec. 3.2)."""
    return X.select_dtypes(include="number").copy()


def drop_missing(X: pd.DataFrame, y: pd.Series):
    """Drop high-missing columns, then any row still containing a NaN (Sec. 3.2)."""
    keep_cols = X.columns[X.isna().mean() <= MAX_MISSING_FRACTION]
    X = X[keep_cols]
    row_mask = X.notna().all(axis=1)
    return X[row_mask].reset_index(drop=True), y[row_mask].reset_index(drop=True)


def drop_low_cardinality_numerical(X: pd.DataFrame) -> pd.DataFrame:
    """Remove numerical features with < 10 unique values (Sec. 3.2)."""
    keep_cols = [c for c in X.columns if X[c].nunique() >= MIN_UNIQUE_NUMERICAL]
    return X[keep_cols].copy()


def binarize_and_balance(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Keep the two most frequent classes and balance them 50/50 (Sec. 3.2).

    'the target is binarised ... by taking the two most numerous classes, and we
    keep half of samples in each class' -> we downsample each kept class to the
    size of the smaller one.
    """
    # Work with plain string labels so a leftover categorical dtype can't keep
    # zero-count classes around (which would break the balancing below).
    y = y.astype(str)
    counts = y.value_counts()
    top2 = list(counts.index[:2])
    mask = y.isin(top2)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    n_per_class = int(min(counts[top2[0]], counts[top2[1]]))
    rng = np.random.default_rng(seed)
    keep_idx = []
    for cls in top2:
        idx = np.where(y.values == cls)[0]
        keep_idx.extend(rng.choice(idx, size=n_per_class, replace=False))
    keep_idx = np.sort(keep_idx)

    X = X.iloc[keep_idx].reset_index(drop=True)
    y = y.iloc[keep_idx].reset_index(drop=True)
    # Map the two class labels to {0, 1}.
    y = (y == top2[1]).astype(int)
    return X, y


def preprocess_numerical_classification(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Full dataset-level preprocessing for the numerical classification setting.

    Order matters: filter features, clean missing data, then binarize+balance the
    target. Train truncation to 10k happens *after* the split (see split_and_truncate).
    """
    X = keep_numerical_only(X)
    X, y = drop_missing(X, y)
    X = drop_low_cardinality_numerical(X)
    X, y = binarize_and_balance(X, y, seed=seed)
    return X, y


def split_and_truncate(X, y, test_size: float = 0.3, seed: int = 0):
    """Stratified train/test split, then truncate the training set to 10k (Sec. 3.2).

    Used by Phases 1-2 (default models, no tuning). Phase 3 uses the train/val/test
    version below.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    if len(X_train) > MEDIUM_TRAIN_SIZE:
        X_train = X_train.iloc[:MEDIUM_TRAIN_SIZE]
        y_train = y_train.iloc[:MEDIUM_TRAIN_SIZE]
    return X_train, X_test, y_train, y_test


def split_train_val_test(X, y, val_size: float = 0.2, test_size: float = 0.2, seed: int = 0):
    """Stratified train/validation/test split, then truncate train to 10k (Sec. 3.2/3.3).

    Validation is used for hyperparameter selection; test is only ever used to report
    the score of the validation-selected model. Truncation is applied to train only, so
    the medium-sized regime does not shrink the evaluation splits.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )
    if len(X_train) > MEDIUM_TRAIN_SIZE:
        X_train = X_train.iloc[:MEDIUM_TRAIN_SIZE]
        y_train = y_train.iloc[:MEDIUM_TRAIN_SIZE]
    return X_train, X_val, X_test, y_train, y_val, y_test
