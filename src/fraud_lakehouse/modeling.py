from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)


@dataclass(frozen=True)
class ModelInputs:
    """Transformed model features and their corresponding target."""

    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class PreparedModelPartitions:
    """Training-fitted preprocessing applied to all temporal partitions."""

    preprocessor: Pipeline
    train: ModelInputs
    validation: ModelInputs
    test: ModelInputs


BaselineEstimator = DummyClassifier | LogisticRegression | XGBClassifier


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Evaluation metrics for an imbalanced binary classifier."""

    average_precision: float
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class BaselineModelEvaluation:
    """A fitted baseline estimator and its temporal evaluations."""

    name: str
    estimator: BaselineEstimator
    validation_metrics: BinaryClassificationMetrics
    test_metrics: BinaryClassificationMetrics


def build_model_preprocessor() -> Pipeline:
    """Create the baseline numeric preprocessing pipeline."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    ).set_output(transform="pandas")


def prepare_model_partitions(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> PreparedModelPartitions:
    """Fit preprocessing on training data and transform every partition."""
    train_inputs = extract_model_inputs(train)
    validation_inputs = extract_model_inputs(validation)
    test_inputs = extract_model_inputs(test)

    preprocessor = build_model_preprocessor()

    transformed_train = preprocessor.fit_transform(
        train_inputs.features,
        train_inputs.target,
    )
    transformed_validation = preprocessor.transform(validation_inputs.features)
    transformed_test = preprocessor.transform(test_inputs.features)

    return PreparedModelPartitions(
        preprocessor=preprocessor,
        train=ModelInputs(
            features=transformed_train.reset_index(drop=True),
            target=train_inputs.target,
        ),
        validation=ModelInputs(
            features=transformed_validation.reset_index(drop=True),
            target=validation_inputs.target,
        ),
        test=ModelInputs(
            features=transformed_test.reset_index(drop=True),
            target=test_inputs.target,
        ),
    )


def train_baseline_models(
    prepared: PreparedModelPartitions,
    *,
    threshold: float = 0.5,
) -> tuple[BaselineModelEvaluation, ...]:
    """Train and evaluate deterministic baseline fraud classifiers."""
    if prepared.train.target.nunique() != 2:
        raise ValueError("The training target must contain both fraud classes.")

    negative_count = int((prepared.train.target == 0).sum())
    positive_count = int((prepared.train.target == 1).sum())
    scale_pos_weight = negative_count / positive_count

    estimators: tuple[tuple[str, BaselineEstimator], ...] = (
        (
            "dummy_prior",
            DummyClassifier(strategy="prior"),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
        ),
    )

    evaluations = []

    for name, estimator in estimators:
        estimator.fit(
            prepared.train.features,
            prepared.train.target,
        )

        validation_probability = fraud_probability(
            estimator,
            prepared.validation.features,
        )
        test_probability = fraud_probability(
            estimator,
            prepared.test.features,
        )

        evaluations.append(
            BaselineModelEvaluation(
                name=name,
                estimator=estimator,
                validation_metrics=evaluate_binary_classifier(
                    target=prepared.validation.target,
                    fraud_probability=validation_probability,
                    threshold=threshold,
                ),
                test_metrics=evaluate_binary_classifier(
                    target=prepared.test.target,
                    fraud_probability=test_probability,
                    threshold=threshold,
                ),
            )
        )

    return tuple(evaluations)


def evaluate_binary_classifier(
    *,
    target: pd.Series,
    fraud_probability: np.ndarray,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Evaluate fraud probabilities using ranking and threshold metrics."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    target_values = pd.to_numeric(target, errors="coerce")

    if target_values.isna().any() or not target_values.isin((0, 1)).all():
        raise ValueError("target must contain only 0 or 1.")

    if target_values.nunique() != 2:
        raise ValueError("target must contain both classes for metric calculation.")

    probability_values = np.asarray(
        fraud_probability,
        dtype=float,
    )

    if probability_values.ndim != 1:
        raise ValueError("fraud_probability must be one-dimensional.")

    if len(probability_values) != len(target_values):
        raise ValueError("target and fraud_probability must have equal lengths.")

    if not np.isfinite(probability_values).all():
        raise ValueError("fraud_probability must contain finite values.")

    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError("fraud_probability must contain values between 0 and 1.")

    predictions = (probability_values >= threshold).astype("int8")
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target_values,
        predictions,
        labels=(0, 1),
    ).ravel()

    return BinaryClassificationMetrics(
        average_precision=float(
            average_precision_score(
                target_values,
                probability_values,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                target_values,
                probability_values,
            )
        ),
        precision=float(
            precision_score(
                target_values,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(recall_score(target_values, predictions)),
        f1_score=float(f1_score(target_values, predictions)),
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


def extract_model_inputs(partition: pd.DataFrame) -> ModelInputs:
    required_columns = (*MODEL_FEATURE_COLUMNS, MODEL_TARGET_COLUMN)
    missing_columns = sorted(set(required_columns).difference(partition.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    features = partition.loc[:, MODEL_FEATURE_COLUMNS].copy()

    for column in MODEL_FEATURE_COLUMNS:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid_values = features[column].notna() & converted.isna()

        if invalid_values.any():
            raise ValueError(f"{column} must contain numeric values.")

        features[column] = converted

    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Model features must not contain infinite values.")

    target = pd.to_numeric(
        partition[MODEL_TARGET_COLUMN],
        errors="coerce",
    )

    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError("tx_fraud must contain only 0 or 1.")

    return ModelInputs(
        features=features.reset_index(drop=True),
        target=target.astype("int8").reset_index(drop=True),
    )


def fraud_probability(
    estimator: BaselineEstimator,
    features: pd.DataFrame,
) -> np.ndarray:
    positive_class_indexes = np.flatnonzero(estimator.classes_ == 1)

    if len(positive_class_indexes) != 1:
        raise ValueError("The fitted estimator must contain fraud class 1.")

    positive_class_index = int(positive_class_indexes[0])

    return np.asarray(
        estimator.predict_proba(features)[:, positive_class_index],
        dtype=float,
    )
