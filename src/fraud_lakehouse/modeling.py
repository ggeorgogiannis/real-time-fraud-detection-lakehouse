from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    train_inputs = _extract_model_inputs(train)
    validation_inputs = _extract_model_inputs(validation)
    test_inputs = _extract_model_inputs(test)

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


def _extract_model_inputs(partition: pd.DataFrame) -> ModelInputs:
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
