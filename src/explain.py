from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG, ProjectConfig
from src.train import TrainingArtifacts, train_and_select_model

try:
    import shap
except ImportError as exc:  # pragma: no cover - surfaced during execution
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None


def _get_transformed_feature_names(artifacts: TrainingArtifacts) -> list[str]:
    preprocessor = artifacts.best_estimator.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def build_business_recommendations(top_features: list[str]) -> list[str]:
    recommendations: list[str] = []
    if any("Payment Delay" in feature for feature in top_features):
        recommendations.append(
            "Prioritize proactive payment reminders or billing support for customers with frequent payment delays."
        )
    if any("Support Calls" in feature for feature in top_features):
        recommendations.append(
            "Escalate service recovery flows when support-call volume spikes, because repeated issues are linked to churn."
        )
    if any("Usage Frequency" in feature for feature in top_features):
        recommendations.append(
            "Trigger retention journeys for customers whose product usage drops below their historical norm."
        )
    if any("Tenure" in feature for feature in top_features):
        recommendations.append(
            "Strengthen onboarding and early-life engagement for lower-tenure customers who are still forming habits."
        )
    if not recommendations:
        recommendations.append(
            "Review the highest-impact features and translate them into targeted retention outreach rules."
        )
    return recommendations


def run_shap_analysis(
    artifacts: TrainingArtifacts,
    config: ProjectConfig = CONFIG,
) -> dict[str, Any]:
    if shap is None:
        raise ImportError(
            "shap is required to generate model explanations. Install dependencies from requirements.txt."
        ) from SHAP_IMPORT_ERROR

    sample_size = min(config.top_shap_sample_size, len(artifacts.bundle.X_test))
    sample = artifacts.bundle.X_test.sample(sample_size, random_state=config.random_state)
    transformed = artifacts.best_estimator.named_steps["preprocessor"].transform(sample)
    model = artifacts.best_estimator.named_steps["model"]

    explainer = shap.Explainer(model, transformed)
    shap_values = explainer(transformed)
    feature_names = _get_transformed_feature_names(artifacts)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top_features = feature_importance.head(8)["feature"].tolist()
    explanation = {
        "selected_model": artifacts.best_model_name,
        "top_features": feature_importance.head(12).to_dict(orient="records"),
        "business_recommendations": build_business_recommendations(top_features),
    }
    (config.reports_dir / "shap_summary.json").write_text(
        json.dumps(explanation, indent=2),
        encoding="utf-8",
    )
    feature_importance.to_csv(config.reports_dir / "shap_feature_importance.csv", index=False)
    return explanation


def main() -> None:
    artifacts = train_and_select_model(CONFIG)
    explanation = run_shap_analysis(artifacts, CONFIG)
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
