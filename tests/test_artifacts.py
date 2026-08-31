import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_artifacts_are_complete_and_text_based():
    metrics = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["rows"] == 20640
    assert metrics["dataset"]["features"] == 8
    assert metrics["evaluation"]["cv_folds"] == 5
    assert metrics["evaluation"]["random_seed"] == 42
    assert (ROOT / "artifacts" / "model_comparison.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert (ROOT / "artifacts" / "residuals.svg").read_text(encoding="utf-8").startswith("<?xml")


def test_readme_matches_generated_metrics_and_method():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))
    for model in metrics["models"].values():
        for key in ["test_mae", "test_rmse", "test_r2"]:
            assert f"{model[key]:.4f}" in readme
    for phrase in [
        "Independent portfolio benchmark",
        "80/20",
        "five-fold",
        "random seed 42",
        "Limitations",
    ]:
        assert phrase.lower() in readme.lower()
    assert "0.8436" not in readme
