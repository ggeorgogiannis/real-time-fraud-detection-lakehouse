# Baseline Fraud Models

## Purpose

The model training workflow establishes reproducible fraud-classification benchmarks using the leakage-safe, chronologically partitioned dataset.

It trains three models against the same feature contract:

* A dummy classifier that represents a non-informative prior-probability baseline.
* A class-balanced logistic-regression classifier that provides an interpretable statistical baseline.
* An imbalance-aware XGBoost classifier that captures nonlinear relationships and feature interactions.

The dummy and logistic-regression models provide reference points for assessing whether the additional complexity of XGBoost produces a meaningful improvement.

## Inputs

The training command reads the model dataset artifacts created by `fraud-lakehouse build-ml-dataset`:

* `train.parquet`
* `validation.parquet`
* `test.parquet`

Each partition must contain all model features and the binary `tx_fraud` target defined in [`ml_dataset.md`](ml_dataset.md).

## Training-Only Preprocessing

The preprocessing pipeline is fitted using the training partition only.

It applies:

1. Median imputation for missing feature values.
2. Standard scaling using statistics calculated from the training partition.

The fitted preprocessing pipeline then transforms the validation and test partitions without recalculating imputation values or scaling statistics.

This prevents information from later periods from influencing model training.

## Fraud Models

### Dummy Prior Classifier

The dummy classifier uses the fraud prevalence observed in the training partition as its predicted probability.

It provides a reference against which the logistic-regression model can be compared. Its purpose is not to detect fraud but to show whether a trained model improves on a non-informative baseline.

### Logistic Regression

The logistic-regression baseline uses:

* Class-balanced training weights.
* The `lbfgs` solver.
* A maximum of 1,000 iterations.
* A fixed random state of `42`.

Class balancing gives additional weight to the minority fraud class without resampling or altering the chronological partitions.

### XGBoost

The XGBoost classifier uses:

* 200 boosting rounds.
* A maximum tree depth of `4`.
* A learning rate of `0.05`.
* Row and feature subsampling rates of `0.8`.
* Histogram-based tree construction.
* Average precision as its evaluation metric.
* A fixed random state of `42`.
* Single-threaded training for reproducible execution.

Class imbalance is handled through `scale_pos_weight`, calculated as the number of legitimate training transactions divided by the number of fraudulent training transactions. This value is derived exclusively from the training partition.

The initial configuration is fixed rather than tuned against the test partition. Validation-based tuning and early stopping can be introduced after performance has been measured on the complete dataset.

## Evaluation

Each model is evaluated separately on the validation and test partitions.

The primary metric is average precision because fraud detection is an imbalanced binary-classification problem. Average precision summarizes the precision-recall relationship across probability thresholds and is more informative than accuracy when legitimate transactions substantially outnumber fraudulent transactions.

The workflow also reports:

| Metric            | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| Average precision | Primary ranking metric for the minority fraud class.          |
| ROC AUC           | Measures ranking quality across classification thresholds.    |
| Precision         | Fraction of predicted fraud cases that are fraudulent.        |
| Recall            | Fraction of fraudulent transactions successfully identified.  |
| F1 score          | Harmonic mean of precision and recall.                        |
| True negatives    | Legitimate transactions classified as legitimate.             |
| False positives   | Legitimate transactions incorrectly classified as fraud.      |
| False negatives   | Fraudulent transactions incorrectly classified as legitimate. |
| True positives    | Fraudulent transactions correctly classified as fraud.        |

Precision, recall, F1 and confusion-matrix counts use a configurable probability threshold. The default threshold is `0.5`.

Validation metrics should guide model development and threshold selection. Test metrics should be treated as the final held-out evaluation rather than repeatedly used for model selection.

## Running the Training Workflow

Build the chronological model dataset first, then run:

```bash
fraud-lakehouse train-baselines \
  --dataset-dir data/ml \
  --output-dir data/models \
  --threshold 0.5
```

The command fails if any required temporal partition is missing, if the training target does not contain both classes or if validation metrics cannot be calculated safely.

## Output Artifacts

The command creates or replaces:

| Artifact                     | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `dummy_prior.joblib`         | Fitted dummy classifier and training preprocessor. |
| `logistic_regression.joblib` | Fitted logistic model and training preprocessor.   |
| `xgboost.joblib`             | Fitted XGBoost model and training preprocessor.    |
| `metrics.json`               | Validation and test metrics for all three models.  |

Each serialized model artifact contains:

* The preprocessing pipeline fitted on training data.
* The fitted estimator.
* The ordered feature list.
* The target-column name.
* The classification threshold.

The metrics file records:

* Metrics schema version `2`.
* The primary evaluation metric.
* The classification threshold.
* The installed scikit-learn and XGBoost versions.
* Separate validation and test metrics for each model.

Model files are written to temporary paths before publication. The metrics file is published last so that it describes a complete set of model artifacts.

Generated models and metrics remain local and are excluded from Git.

## Reproducibility

The workflow uses fixed model configurations, ordered feature columns, chronological input partitions and training-only preprocessing. XGBoost runs with a fixed random state and a single worker.

Given the same input partitions, supported dependency versions and execution environment, repeated executions produce equivalent fitted models and identical metrics.

Joblib artifacts use Python's pickle-based serialization and must only be loaded from trusted sources.

## Interpretation of Test Fixtures

The repository's automated tests use small, deliberately constructed datasets. High scores from these fixtures verify that the implementation is wired correctly; they do not represent expected performance on the full fraud dataset.

Meaningful model assessment requires running the workflow on the complete chronological dataset and reviewing class prevalence, average precision, recall, false positives and false negatives.

## Current Limitations

The current modeling workflow does not yet include:

* Probability calibration.
* Validation-based threshold optimization.
* Hyperparameter tuning.
* Validation-based early stopping.
* Time-series cross-validation.
* Cost-sensitive business metrics.
* Model explainability or feature-attribution reports.
* Experiment tracking.

These capabilities can be introduced after the three models have been evaluated on the complete chronological dataset.
