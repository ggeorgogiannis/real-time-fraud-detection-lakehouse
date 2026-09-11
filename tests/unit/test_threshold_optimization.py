import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.optimization import TunedModelArtifact
from fraud_lakehouse.threshold_optimization import (
    optimize_and_publish_thresholds,
)


class _IdentityPreprocessor:
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return features


class _FixedProbabilityEstimator:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert len(features) == len(self.probabilities)

        return np.column_stack(
            (
                1.0 - self.probabilities,
                self.probabilities,
            )
        )


def _validation_partition() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-02T01:00:00Z",
            "2026-01-02T02:00:00Z",
            "2026-01-02T03:00:00Z",
            "2026-01-03T01:00:00Z",
            "2026-01-03T02:00:00Z",
            "2026-01-03T03:00:00Z",
        ]
    )
    target = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    return pd.DataFrame(
        {
            "tx_datetime": timestamps,
            "transaction_date": timestamps.date,
            "customer_id": range(1, 10),
            "tx_amount": np.linspace(10.0, 90.0, 9),
            "transaction_hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "is_weekend": 0,
            "is_night": 1,
            "customer_previous_transaction_count": range(9),
            "customer_previous_mean_amount": 50.0,
            "amount_to_customer_previous_mean": np.linspace(
                0.2,
                1.8,
                9,
            ),
            "terminal_previous_transaction_count": range(9),
            MODEL_TARGET_COLUMN: target,
        }
    )


def _write_artifact(
    path: Path,
    probabilities: list[float],
) -> None:
    joblib.dump(
        TunedModelArtifact(
            preprocessor=_IdentityPreprocessor(),
            estimator=_FixedProbabilityEstimator(probabilities),
            feature_columns=MODEL_FEATURE_COLUMNS,
            target_column=MODEL_TARGET_COLUMN,
            hyperparameters={"test_parameter": 1},
        ),
        path,
    )


def test_optimize_and_publish_thresholds_uses_validation_only(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "policy"
    dataset_dir.mkdir()
    model_dir.mkdir()

    _validation_partition().to_parquet(
        dataset_dir / "validation.parquet",
        engine="pyarrow",
        index=False,
    )

    _write_artifact(
        model_dir / "tuned_logistic_regression.joblib",
        [
            0.90,
            0.80,
            0.70,
            0.95,
            0.85,
            0.75,
            0.90,
            0.80,
            0.74,
        ],
    )
    _write_artifact(
        model_dir / "tuned_xgboost.joblib",
        [
            0.90,
            0.80,
            0.70,
            0.80,
            0.90,
            0.70,
            0.80,
            0.70,
            0.90,
        ],
    )

    outputs = optimize_and_publish_thresholds(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        daily_card_capacity=2,
    )

    assert outputs.policy_path == output_dir / "threshold_policy.json"
    assert outputs.selected_model_name == "xgboost"
    assert outputs.selected_threshold == pytest.approx(np.nextafter(0.70, np.inf))

    policy = json.loads(outputs.policy_path.read_text(encoding="utf-8"))

    assert policy["schema_version"] == 1
    assert policy["selection_data"] == "validation"
    assert policy["test_evaluation"] == {"performed": False}
    assert policy["operating_constraint"] == {
        "daily_card_capacity": 2,
        "scope": "unique_customer_cards_per_day",
    }
    assert policy["selection_order"] == [
        "card_day_recall",
        "card_day_precision",
        "average_precision",
    ]
    assert policy["selected_policy"]["model_name"] == "xgboost"
    assert policy["selected_policy"]["threshold"] == pytest.approx(np.nextafter(0.70, np.inf))
    assert set(policy["models"]) == {
        "logistic_regression",
        "xgboost",
    }
    first_policy = outputs.policy_path.read_bytes()

    repeated_outputs = optimize_and_publish_thresholds(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        daily_card_capacity=2,
    )

    assert repeated_outputs.policy_path.read_bytes() == first_policy
    assert not list(output_dir.glob(".*.tmp"))


def test_optimize_and_publish_thresholds_requires_validation_partition(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="validation.parquet",
    ):
        optimize_and_publish_thresholds(
            dataset_dir=tmp_path / "dataset",
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "policy",
            daily_card_capacity=100,
        )


def test_optimize_and_publish_thresholds_requires_tuned_models(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"
    dataset_dir.mkdir()
    model_dir.mkdir()

    _validation_partition().to_parquet(
        dataset_dir / "validation.parquet",
        engine="pyarrow",
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="tuned_logistic_regression.joblib",
    ):
        optimize_and_publish_thresholds(
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            output_dir=tmp_path / "policy",
            daily_card_capacity=100,
        )
