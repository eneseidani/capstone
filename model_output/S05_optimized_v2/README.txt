S05 Classification Model v2 outputs

correlation_excluded_features.csv: fixed pre-2022 high-correlation exclusion candidates; all kept out of S05 model input.

model_metrics.csv: validation and test classification metrics. Precision is TP/(TP+FP) at probability threshold 0.50.
predictions_validation.csv / predictions_test.csv: ticker-date probabilities and validation-selected screening flags.
hyperparameter_trials.csv: Optuna trial parameters, tuning-subset objective values and recorded metrics. Optuna tunes on six evenly spaced validation dates by default; winning parameters are then evaluated on all validation dates.
selected_hyperparameters.json: selected parameters per ML model.
screening_validation_candidates.csv: validation-only screening candidates and precision lift.
feature_ablation.csv: original features, original plus historical frequency, and historical frequency only.
feature_importance_by_model.csv: permutation importance for transformed input columns on a representative validation fold.
run_manifest.json: experiment dates, settings and limitations.

ROC-AUC: ranking/discrimination. MCC: balanced correlation between labels and hard predictions. Balanced Accuracy: mean sensitivity and specificity. PR-AUC: average precision. Log Loss: probability calibration/quality (lower is better). F1: harmonic mean of precision and recall. Precision: fraction of predicted positives that are truly positive. Recall: fraction of actual positives identified.
