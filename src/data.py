from __future__ import annotations

from os import PathLike
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import CONFIG, ProjectConfig


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    business_context: dict[str, Any]


def _read_csv(path: str | PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path).dropna().reset_index(drop=True)


def validate_schema(df: pd.DataFrame, config: ProjectConfig = CONFIG) -> None:
    missing_columns = set(config.expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def build_business_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> dict[str, Any]:
    train_rate = float(train_df[config.target_column].mean())
    test_rate = float(test_df[config.target_column].mean())

    return {
        "target_definition": "Churn = 1 means a customer is likely to leave, so the retention team should intervene.",
        "positive_class": "customer churns",
        "negative_class": "customer stays",
        "leakage_risk": [config.id_column],
        "model_selection_metric": config.scoring_metric,
        "cost_assumptions": {
            "false_negative_cost": config.false_negative_cost,
            "false_positive_cost": config.false_positive_cost,
            "business_reasoning": (
                "Missing a true churner is assumed to cost around 8x more than contacting "
                "a customer who would have stayed."
            ),
        },
        "distribution_shift": {
            "train_churn_rate": round(train_rate, 4),
            "test_churn_rate": round(test_rate, 4),
            "absolute_gap": round(abs(train_rate - test_rate), 4),
        },
        "limitations": [
            "The dataset appears synthetic or curated, so real-world behavior may be noisier.",
            "Train and test sets show different churn rates, indicating potential distribution shift.",
            "No campaign outcome data is available, so threshold tuning is based on assumed business costs.",
        ],
    }


def split_features_target(
    df: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        column
        for column in config.expected_columns
        if column not in {config.target_column, config.id_column}
    ]
    X = df.loc[:, feature_columns].copy()
    y = df[config.target_column].astype(int).copy()
    return X, y


def load_dataset_bundle(config: ProjectConfig = CONFIG) -> DatasetBundle:
    train_df = _read_csv(config.training_file)
    test_df = _read_csv(config.testing_file)
    validate_schema(train_df, config)
    validate_schema(test_df, config)

    X_train, y_train = split_features_target(train_df, config)
    X_test, y_test = split_features_target(test_df, config)

    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        business_context=build_business_context(train_df, test_df, config),
    )
