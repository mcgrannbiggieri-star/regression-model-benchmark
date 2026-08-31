# Methodology

## Dataset

The command loads `fetch_california_housing(as_frame=True)` from scikit-learn. The returned table contains 20,640 rows, eight numeric predictors, and the `MedHouseVal` target.

## Split and leakage control

`train_test_split` creates one 80/20 split with random seed 42. The holdout partition is not used for model selection or cross-validation. Five-fold `KFold(shuffle=True, random_state=42)` operates only on the training partition.

## Models

1. Median `DummyRegressor` establishes a non-learning baseline.
2. `LinearRegression` tests a simple linear relationship.
3. `HistGradientBoostingRegressor(random_state=42)` tests nonlinear interactions without manual hyperparameter search.

## Metrics

Cross-validation reports mean and standard deviation of mean absolute error. After cross-validation, each model is fit on the full training partition and evaluated once on the untouched holdout partition using MAE, RMSE, and R².

## Reproducibility

The script writes native numeric values to sorted JSON. SVG metadata omits the generation timestamp and uses a fixed hash salt, allowing deterministic artifact comparison under the same library versions.
