from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CONFIG, ProjectConfig


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    config: ProjectConfig = CONFIG,
) -> dict[str, float]:
    thresholds = np.linspace(0.05, 0.95, config.probability_grid_size)
    best_result: dict[str, float] | None = None

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        cost = (fn * config.false_negative_cost) + (fp * config.false_positive_cost)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        current = {
            "threshold": float(threshold),
            "cost": float(cost),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        if best_result is None or current["cost"] < best_result["cost"]:
            best_result = current

    if best_result is None:
        raise ValueError("Threshold optimization failed.")
    return best_result


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold_result: dict[str, float],
) -> dict[str, Any]:
    predictions = (probabilities >= threshold_result["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold_analysis": threshold_result,
    }


def plot_precision_recall_curve(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    display = PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    display.ax_.set_title(f"Precision-Recall Curve: {model_name}")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=200)
    plt.close(display.figure_)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
