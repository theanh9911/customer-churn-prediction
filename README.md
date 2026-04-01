# Customer Churn Prediction

An end-to-end churn prediction project built to frame customer churn as a retention business problem, not just a binary classification exercise. The project compares `Logistic Regression`, `Random Forest`, and `XGBoost`, selects the final model using the Precision-Recall tradeoff, and tunes the operating threshold based on the business cost of false negatives versus false positives.

## Business Problem

The retention team needs a model that flags customers with high churn risk early enough to trigger outreach. In this project:

- `Churn = 1` means the customer is likely to leave.
- False negatives are treated as more expensive than false positives because missing a true churner can directly impact revenue retention.
- The decision threshold is optimized with an assumed `8:1` cost ratio between false negatives and false positives.

## Dataset

- Training data: `data/customer_churn_dataset-training-master.csv`
- Test data: `data/customer_churn_dataset-testing-master.csv`
- Training rows after null removal: `440,832`
- Holdout test rows after null removal: `64,374`

Features include customer demographics, contract details, billing behavior, support usage, engagement, and spend:

- Numeric: `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`
- Categorical: `Gender`, `Subscription Type`, `Contract Length`
- Excluded from modeling: `CustomerID` to avoid leakage

## Project Structure

```text
customer-churn/
|-- artifacts/
|   |-- models/
|   |   `-- best_model.joblib
|   |-- plots/
|   |   |-- logistic_regression_pr_curve.png
|   |   |-- random_forest_pr_curve.png
|   |   `-- xgboost_pr_curve.png
|   `-- reports/
|       |-- model_comparison.csv
|       |-- model_comparison.json
|       |-- shap_feature_importance.csv
|       `-- shap_summary.json
|-- data/
|-- notebooks/
|   `-- churn_analysis.ipynb
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- evaluate.py
|   |-- explain.py
|   |-- features.py
|   |-- predict.py
|   `-- train.py
|-- main.py
`-- requirements.txt
```

## Methodology

1. Validate schema and define the target variable.
2. Drop rows with missing values and remove `CustomerID` from the feature set.
3. Build a reusable preprocessing pipeline with `ColumnTransformer`.
4. Train three candidate models:
   - `Logistic Regression`
   - `Random Forest`
   - `XGBoost`
5. Tune hyperparameters with `GridSearchCV` and `StratifiedKFold`.
6. Compare models using `average_precision` instead of accuracy alone.
7. Optimize the classification threshold using business costs.
8. Run SHAP-based explainability and translate key drivers into business actions.

## Model Results

The project sampled `60,000` rows from training data for grid search runtime control while keeping evaluation on the full holdout test set.

| Model | Average Precision | ROC AUC | Precision | Recall | Best Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6379 | 0.6875 | 0.5246 | 0.9910 | 0.4299 |
| Random Forest | 0.5708 | 0.6557 | 0.4936 | 0.9979 | 0.8188 |
| XGBoost | 0.6072 | 0.7089 | 0.4913 | 0.9982 | 0.9500 |

Final model selected: `Logistic Regression`

Why it was selected:

- It achieved the strongest Precision-Recall performance on the holdout set.
- It generalized better than the tree-based models despite those models producing near-perfect CV scores on sampled training data.
- Its coefficients and SHAP explanations are easier to communicate to business stakeholders.

## Threshold Tuning

The final threshold is not fixed at `0.50`. Instead, it is optimized for campaign economics:

- False negative cost: `8.0`
- False positive cost: `1.0`
- Best operating threshold on holdout test: `0.4299`

At that threshold, the final model achieved:

- Precision: `0.5246`
- Recall: `0.9910`
- F1 score: `0.6861`
- Confusion matrix: `TN=6502`, `FP=27379`, `FN=275`, `TP=30218`

## Explainability

Top churn drivers identified with SHAP:

- `Contract Length = Monthly`
- `Support Calls`
- `Total Spend`
- `Payment Delay`
- `Last Interaction`
- `Age`

Business recommendations:

- Prioritize payment reminder and billing support workflows for customers with repeated payment delays.
- Escalate service recovery for customers with unusually high support-call volume.
- Trigger retention journeys when usage frequency declines.
- Strengthen onboarding and early lifecycle engagement for lower-tenure customers.

## Limitations

- The train and test sets show different churn rates (`56.71%` vs `47.37%`), which suggests distribution shift.
- The dataset appears synthetic or pre-curated, so real-world noise may be higher.
- Threshold tuning is based on assumed cost ratios because campaign outcome data is not available.
- Tree-based models may be overfitting the sampled training data despite excellent CV scores.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full training, evaluation, and SHAP pipeline:

```bash
python main.py
```

Score a new dataset with the saved model:

```bash
python -m src.predict --input data/customer_churn_dataset-testing-master.csv --output artifacts/reports/scored_test.csv
```
