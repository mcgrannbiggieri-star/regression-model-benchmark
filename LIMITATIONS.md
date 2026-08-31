# Limitations

- The California Housing dataset reflects 1990 California census-derived data and is not a current housing-price model.
- The comparison includes three deliberately simple model families and no hyperparameter search.
- The fixed holdout estimate depends on one random split; cross-validation variation is reported only for training folds.
- Results may vary across major scikit-learn, NumPy, or data-source revisions despite a fixed seed.
- R², MAE, and RMSE describe predictive fit on this dataset; they do not establish causal relationships or fairness.
- The residual plot supports diagnostic review but does not replace subgroup, robustness, or drift analysis.
- This is an independent portfolio benchmark, not a production model, deployed service, or client engagement.
