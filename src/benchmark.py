from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def split_data(X, y, random_state: int = 42):
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def build_models(random_state: int = 42) -> dict:
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state: int = 42,
) -> dict:
    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    model_results: dict[str, dict[str, float]] = {}
    for name, model in build_models(random_state).items():
        cv_scores = -cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=folds,
        )
        fitted = clone(model).fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        model_results[name] = {
            "cv_mae_mean": float(np.mean(cv_scores)),
            "cv_mae_std": float(np.std(cv_scores)),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "test_r2": float(r2_score(y_test, predictions)),
        }

    return {
        "dataset": {
            "rows": int(len(X_train) + len(X_test)),
            "features": int(X_train.shape[1]),
            "feature_names": [str(column) for column in X_train.columns],
        },
        "evaluation": {
            "random_seed": int(random_state),
            "train_ratio": 0.8,
            "test_ratio": 0.2,
            "cv_folds": 5,
            "cv_scope": "training split only",
        },
        "models": model_results,
    }
