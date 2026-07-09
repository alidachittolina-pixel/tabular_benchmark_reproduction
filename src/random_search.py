"""
Random search with the paper's selection protocol (Sec. 3.3).

For a fixed train/validation/test split, we draw `n_iter` configurations (iteration 1 =
defaults), fit each on train, and score on validation and test. Selection is on
validation accuracy (the paper's classification metric); the test set is only ever used
to report the score of the validation-selected model.

This version additionally records, per iteration:
  - test_score  : test accuracy   (kept under this name so benchmark.py works unchanged)
  - test_f1     : test macro-F1
  - fit_time    : seconds to fit that configuration (training time)
and tracks the accuracy AND F1 of the best-on-validation config so far.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import build_model
from search_spaces import sample_params


def run_random_search(
    name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    n_iter: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Run random search for one model and return a per-iteration record table."""
    rng = np.random.default_rng(seed)
    records = []
    best_val = -np.inf
    test_at_best_val = np.nan   # test accuracy of the val-selected config
    f1_at_best_val = np.nan     # test F1 of the val-selected config

    for i in range(n_iter):
        params = {} if i == 0 else sample_params(name, rng)
        model = build_model(name, params, seed=seed)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_score = float(accuracy_score(y_val, val_pred))          # selection metric
        test_score = float(accuracy_score(y_test, test_pred))       # test accuracy
        test_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))

        if val_score > best_val:
            best_val = val_score
            test_at_best_val = test_score
            f1_at_best_val = test_f1

        records.append({
            "model": name,
            "iteration": i + 1,
            "is_default": i == 0,
            "val_score": val_score,          # validation accuracy (for selection)
            "test_score": test_score,        # test accuracy
            "test_f1": test_f1,              # test macro-F1
            "fit_time": fit_time,            # training time (s)
            "best_val_so_far": best_val,
            "test_at_best_val": test_at_best_val,   # accuracy plotted by the paper
            "f1_at_best_val": f1_at_best_val,
            "params": params,
        })

    return pd.DataFrame(records)
