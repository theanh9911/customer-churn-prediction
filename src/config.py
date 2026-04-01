from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "artifacts"
    plots_dir: Path = artifacts_dir / "plots"
    reports_dir: Path = artifacts_dir / "reports"
    models_dir: Path = artifacts_dir / "models"
    training_file: Path = data_dir / "customer_churn_dataset-training-master.csv"
    testing_file: Path = data_dir / "customer_churn_dataset-testing-master.csv"
    target_column: str = "Churn"
    id_column: str = "CustomerID"
    random_state: int = 42
    cv_folds: int = 3
    scoring_metric: str = "average_precision"
    false_negative_cost: float = 8.0
    false_positive_cost: float = 1.0
    probability_grid_size: int = 200
    top_shap_sample_size: int = 3000
    train_sample_size: int | None = 60000
    model_names: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    categorical_features: tuple[str, ...] = (
        "Gender",
        "Subscription Type",
        "Contract Length",
    )
    numeric_features: tuple[str, ...] = (
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
    )
    expected_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "CustomerID",
            "Age",
            "Gender",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Subscription Type",
            "Contract Length",
            "Total Spend",
            "Last Interaction",
            "Churn",
        )
    )

    def ensure_directories(self) -> None:
        for path in (self.artifacts_dir, self.plots_dir, self.reports_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


CONFIG = ProjectConfig()
