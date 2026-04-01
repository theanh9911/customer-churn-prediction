from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CONFIG, ProjectConfig
from src.data import DatasetBundle, load_dataset_bundle
from src.evaluate import evaluate_predictions, optimize_threshold, plot_precision_recall_curve, save_metrics
from src.features import build_preprocessor

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - surfaced during execution
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


@dataclass
class TrainingArtifacts:
    bundle: DatasetBundle
    summary: dict[str, Any]
    best_model_name: str
    best_estimator: Pipeline
    best_probabilities: np.ndarray


def build_model_candidates(config: ProjectConfig = CONFIG) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=config.random_state,
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 5.0],
                "model__penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=config.random_state,
            ),
            "param_grid": {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [1, 5],
            },
        },
    }

    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required to run this project. Install dependencies from requirements.txt."
        ) from XGBOOST_IMPORT_ERROR

    candidates["xgboost"] = {
        "estimator": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.08,
            n_estimators=180,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "param_grid": {
            "model__max_depth": [4, 6],
            "model__min_child_weight": [1, 3],
        },
    }
    return candidates


def maybe_sample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    if config.train_sample_size is None or len(X) <= config.train_sample_size:
        return X, y

    target_ratio = y.mean()
    positive_count = int(round(config.train_sample_size * target_ratio))
    negative_count = config.train_sample_size - positive_count
    sample = pd.concat(
        [
            X.loc[y == 1].sample(n=positive_count, random_state=config.random_state),
            X.loc[y == 0].sample(n=negative_count, random_state=config.random_state),
        ],
        axis=0,
    ).sample(frac=1.0, random_state=config.random_state)
    sampled_y = y.loc[sample.index].astype(int)
    sample = sample.reset_index(drop=True)
    sampled_y = sampled_y.reset_index(drop=True)
    return sample, sampled_y


def train_and_select_model(config: ProjectConfig = CONFIG) -> TrainingArtifacts:
    config.ensure_directories()
    bundle = load_dataset_bundle(config)
    X_train, y_train = maybe_sample_training_data(bundle.X_train, bundle.y_train, config)

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    summary: dict[str, Any] = {
        "business_context": bundle.business_context,
        "data_profile": {
            "train_rows": int(len(bundle.X_train)),
            "sampled_train_rows": int(len(X_train)),
            "test_rows": int(len(bundle.X_test)),
        },
        "model_results": {},
    }

    best_model_name = ""
    best_estimator: Pipeline | None = None
    best_probabilities: np.ndarray | None = None
    best_score = float("-inf")

    for model_name, candidate in build_model_candidates(config).items():
        print(f"Training {model_name}...", flush=True)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(config)),
                ("model", clone(candidate["estimator"])),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=candidate["param_grid"],
            scoring=config.scoring_metric,
            cv=cv,
            n_jobs=1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        print(f"Finished {model_name}. Best CV score: {search.best_score_:.4f}", flush=True)

        probabilities = search.best_estimator_.predict_proba(bundle.X_test)[:, 1]
        threshold_result = optimize_threshold(bundle.y_test, probabilities, config)
        metrics = evaluate_predictions(bundle.y_test, probabilities, threshold_result)
        metrics["best_params"] = search.best_params_
        metrics["cv_best_score"] = float(search.best_score_)
        summary["model_results"][model_name] = metrics

        plot_precision_recall_curve(
            bundle.y_test,
            probabilities,
            model_name,
            config.plots_dir / f"{model_name}_pr_curve.png",
        )

        if metrics["average_precision"] > best_score:
            best_score = metrics["average_precision"]
            best_model_name = model_name
            best_estimator = search.best_estimator_
            best_probabilities = probabilities

    if best_estimator is None or best_probabilities is None:
        raise ValueError("No valid model was trained.")

    summary["selected_model"] = {
        "model_name": best_model_name,
        "selection_reason": "Highest average precision on the holdout test set.",
        "average_precision": summary["model_results"][best_model_name]["average_precision"],
        "threshold": summary["model_results"][best_model_name]["threshold_analysis"]["threshold"],
    }

    joblib.dump(best_estimator, config.models_dir / "best_model.joblib")
    save_metrics(summary, config.reports_dir / "model_comparison.json")
    pd.DataFrame(summary["model_results"]).T.to_csv(
        config.reports_dir / "model_comparison.csv",
        index=True,
    )

    return TrainingArtifacts(
        bundle=bundle,
        summary=summary,
        best_model_name=best_model_name,
        best_estimator=best_estimator,
        best_probabilities=best_probabilities,
    )


def main() -> None:
    artifacts = train_and_select_model(CONFIG)
    print(json.dumps(artifacts.summary["selected_model"], indent=2))


if __name__ == "__main__":
    main()
