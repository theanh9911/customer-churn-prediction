from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import CONFIG, ProjectConfig


def score_dataset(
    input_path: Path,
    output_path: Path,
    config: ProjectConfig = CONFIG,
) -> Path:
    model = joblib.load(config.models_dir / "best_model.joblib")
    df = pd.read_csv(input_path).dropna().reset_index(drop=True)

    feature_columns = [
        column
        for column in config.expected_columns
        if column not in {config.target_column, config.id_column}
    ]
    threshold = json.loads((config.reports_dir / "model_comparison.json").read_text(encoding="utf-8"))[
        "selected_model"
    ]["threshold"]
    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(df[feature_columns])[:, 1]
    scored["predicted_churn"] = (scored["churn_probability"] >= threshold).astype(int)
    scored.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score customer churn risk.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = score_dataset(args.input, args.output)
    print(f"Scored dataset saved to {output_path}")


if __name__ == "__main__":
    main()
