import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.benchmark import evaluate_models, split_data


def synthetic_regression():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = 3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.1, size=120)
    return X, y


def test_split_is_deterministic_and_80_20():
    X, y = synthetic_regression()
    first = split_data(X, y)
    second = split_data(X, y)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[1].index.tolist() == second[1].index.tolist()
    assert len(first[0]) == 96
    assert len(first[1]) == 24


def test_metrics_schema_contains_three_models():
    X, y = synthetic_regression()
    result = evaluate_models(*split_data(X, y))
    assert set(result["models"]) == {
        "dummy_median",
        "linear_regression",
        "hist_gradient_boosting",
    }
    for values in result["models"].values():
        assert {
            "cv_mae_mean",
            "cv_mae_std",
            "test_mae",
            "test_rmse",
            "test_r2",
        } <= values.keys()
        assert all(isinstance(value, float) for value in values.values())


def test_evaluation_is_deterministic():
    X, y = synthetic_regression()
    split = split_data(X, y)
    assert evaluate_models(*split) == evaluate_models(*split)
