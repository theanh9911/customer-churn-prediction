# Customer Churn Case Study

## Objective

Predict which customers are most likely to churn so a retention campaign can prioritize intervention where missed churners are more expensive than unnecessary outreach.

## Dataset Snapshot

- Training rows: `440,832`
- Test rows: `64,374`
- Positive target: `Churn = 1`
- Leakage removed: `CustomerID`
- Train churn rate: `56.71%`
- Test churn rate: `47.37%`

## Modeling Approach

- Preprocessing with `ColumnTransformer`
- Models compared: `Logistic Regression`, `Random Forest`, `XGBoost`
- Hyperparameter tuning with `GridSearchCV`
- Validation with `StratifiedKFold (3 folds)`
- Selection metric: `average_precision`
- Threshold tuning based on `FN cost = 8`, `FP cost = 1`

## Final Model

Selected model: `Logistic Regression`

Reason:

- Best holdout `average_precision` among all candidates
- More stable generalization than tree-based models
- Easier to explain to non-technical stakeholders

## Holdout Results

| Metric | Value |
|---|---:|
| Average Precision | 0.6379 |
| ROC AUC | 0.6875 |
| Precision | 0.5246 |
| Recall | 0.9910 |
| F1 Score | 0.6861 |
| Threshold | 0.4299 |

Confusion matrix at the optimized threshold:

- True negatives: `6,502`
- False positives: `27,379`
- False negatives: `275`
- True positives: `30,218`

## SHAP Findings

Top drivers:

- Monthly contract length
- Support calls
- Total spend
- Payment delay
- Last interaction recency
- Age

## Business Recommendations

- Contact customers on monthly contracts earlier in the lifecycle.
- Trigger payment support or reminder workflows for delayed payments.
- Escalate service recovery after repeated support issues.
- Launch re-engagement campaigns when product usage weakens.

## Limitations

- Train and test churn rates differ, suggesting distribution shift.
- Threshold tuning uses assumed cost ratios rather than observed campaign ROI.
- The dataset likely does not capture all real-world noise and intervention effects.
