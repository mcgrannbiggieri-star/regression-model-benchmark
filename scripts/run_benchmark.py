from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmark import build_models, evaluate_models, split_data


def save_svg(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Jinkun Huang"},
    )
    plt.close(figure)


def plot_model_comparison(metrics: dict, output_path: Path) -> None:
    labels = list(metrics["models"])
    display = [name.replace("_", " ").title() for name in labels]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#94a3b8", "#38bdf8", "#2563eb"]
    definitions = [
        ("test_mae", "Holdout MAE", "Lower is better"),
        ("test_rmse", "Holdout RMSE", "Lower is better"),
        ("test_r2", "Holdout R²", "Higher is better"),
    ]
    for axis, (metric, title, subtitle) in zip(axes, definitions):
        values = [metrics["models"][name][metric] for name in labels]
        bars = axis.bar(display, values, color=colors)
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], fontsize=8)
    figure.suptitle("California Housing — Deterministic Model Comparison", fontsize=14)
    figure.tight_layout()
    save_svg(figure, output_path)


def plot_residuals(best_name: str, X_train, X_test, y_train, y_test, output_path: Path) -> None:
    model = clone(build_models(42)[best_name]).fit(X_train, y_train)
    predictions = model.predict(X_test)
    residuals = np.asarray(y_test) - np.asarray(predictions)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.hist(residuals, bins=40, color="#2563eb", alpha=0.85, edgecolor="white")
    axis.axvline(0, color="#0f172a", linestyle="--", linewidth=1.2)
    axis.set_title(f"Holdout Residual Distribution — {best_name.replace('_', ' ').title()}")
    axis.set_xlabel("Observed minus predicted target")
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_svg(figure, output_path)


def run() -> dict:
    matplotlib.rcParams["svg.hashsalt"] = "regression-model-benchmark-v1"
    dataset = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = split_data(dataset.data, dataset.target, 42)
    metrics = evaluate_models(X_train, X_test, y_train, y_test, 42)
    metrics["dataset"].update(
        {
            "name": "California Housing",
            "source": "scikit-learn fetch_california_housing",
            "target": str(dataset.target.name),
        }
    )
    metrics["best_model_by_test_mae"] = min(
        metrics["models"], key=lambda name: metrics["models"][name]["test_mae"]
    )

    data_dir = ROOT / "data"
    artifact_dir = ROOT / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_model_comparison(metrics, artifact_dir / "model_comparison.svg")
    plot_residuals(
        metrics["best_model_by_test_mae"],
        X_train,
        X_test,
        y_train,
        y_test,
        artifact_dir / "residuals.svg",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
