import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_lakehouse.final_evaluation import (
    evaluate_and_publish_final_policy,
)
from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.optimization import TunedModelArtifact


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


def _test_partition() -> pd.DataFrame:
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
            MODEL_TARGET_COLUMN: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        }
    )


def test_evaluate_and_publish_final_policy_uses_locked_policy(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "evaluation"
    dataset_dir.mkdir()
    model_dir.mkdir()

    _test_partition().to_parquet(
        dataset_dir / "test.parquet",
        engine="pyarrow",
        index=False,
    )

    model_path = model_dir / "tuned_xgboost.joblib"
    joblib.dump(
        TunedModelArtifact(
            preprocessor=_IdentityPreprocessor(),
            estimator=_FixedProbabilityEstimator(
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
                ]
            ),
            feature_columns=MODEL_FEATURE_COLUMNS,
            target_column=MODEL_TARGET_COLUMN,
            hyperparameters={"test_parameter": 1},
        ),
        model_path,
    )

    policy_path = model_dir / "threshold_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_data": "validation",
                "test_evaluation": {"performed": False},
                "selected_policy": {
                    "model_name": "xgboost",
                    "model_artifact": model_path.name,
                    "threshold": 0.75,
                },
                "operating_constraint": {
                    "daily_card_capacity": 2,
                    "scope": "unique_customer_cards_per_day",
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = evaluate_and_publish_final_policy(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        policy_path=policy_path,
        output_dir=output_dir,
    )

    assert outputs.results_path == (output_dir / "final_evaluation.json")
    assert outputs.model_name == "xgboost"
    assert outputs.threshold == pytest.approx(0.75)

    results = json.loads(outputs.results_path.read_text(encoding="utf-8"))

    assert results["evaluation_data"] == "test"
    assert results["selection_locked"] is True
    assert results["model_name"] == "xgboost"
    assert results["threshold"] == pytest.approx(0.75)

    transaction = results["transaction_metrics"]
    assert transaction["true_positives"] == 2
    assert transaction["false_positives"] == 5
    assert transaction["false_negatives"] == 1
    assert transaction["true_negatives"] == 1

    card_day = results["card_day_metrics"]
    assert card_day["card_day_precision"] == pytest.approx(2 / 7)
    assert card_day["card_day_recall"] == pytest.approx(2 / 3)
    assert card_day["budget_exceeded_days"] == 1

    assert not list(output_dir.glob(".*.tmp"))
