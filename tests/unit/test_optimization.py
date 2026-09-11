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
from fraud_lakehouse.optimization import (
    optimize_and_publish_hyperparameters,
)


def _training_partition() -> pd.DataFrame:
    rows = []

    for day_offset in range(14):
        transaction_day = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=day_offset)

        for target in (0, 1):
            timestamp = transaction_day + pd.Timedelta(hours=target)

            rows.append(
                {
                    "tx_datetime": timestamp,
                    "transaction_date": timestamp.date(),
                    "customer_id": day_offset * 2 + target,
                    "tx_amount": 10.0 + (90.0 * target),
                    "transaction_hour": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "is_night": int(timestamp.hour < 6),
                    "customer_previous_transaction_count": day_offset,
                    "customer_previous_mean_amount": 50.0,
                    "amount_to_customer_previous_mean": 0.2 + (1.8 * target),
                    "terminal_previous_transaction_count": day_offset,
                    MODEL_TARGET_COLUMN: target,
                }
            )

    return pd.DataFrame(rows)


def test_optimize_and_publish_hyperparameters_writes_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "models"
    dataset_dir.mkdir()

    training_partition = _training_partition()
    training_partition.to_parquet(
        dataset_dir / "train.parquet",
        engine="pyarrow",
        index=False,
    )

    outputs = optimize_and_publish_hyperparameters(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        logistic_iterations=1,
        xgboost_iterations=1,
        n_folds=2,
        assessment_days=2,
        gap_days=1,
        card_precision_k=1,
        random_state=42,
    )

    assert outputs.logistic_model_path == (output_dir / "tuned_logistic_regression.joblib")
    assert outputs.xgboost_model_path == (output_dir / "tuned_xgboost.joblib")
    assert outputs.results_path == (output_dir / "hyperparameter_search.json")

    results = json.loads(outputs.results_path.read_text(encoding="utf-8"))

    assert results["schema_version"] == 1
    assert results["selection_data"] == "train"
    assert results["primary_metric"] == "average_precision"
    assert results["secondary_metric"] == "card_precision_at_k"
    assert results["threshold_optimization"] == {
        "performed": False,
    }
    assert results["validation_strategy"] == {
        "type": "prequential_expanding_window",
        "n_folds": 2,
        "assessment_days": 2,
        "gap_days": 1,
    }
    assert set(results["models"]) == {
        "logistic_regression",
        "xgboost",
    }

    for model_name in ("logistic_regression", "xgboost"):
        model_results = results["models"][model_name]

        assert model_results["candidate_count"] == 1
        assert 0.0 <= model_results["best_candidate"]["mean_average_precision"] <= 1.0
        assert 0.0 <= model_results["best_candidate"]["mean_card_precision_at_k"] <= 1.0

    for model_path in (
        outputs.logistic_model_path,
        outputs.xgboost_model_path,
    ):
        artifact = joblib.load(model_path)
        transformed = artifact.preprocessor.transform(
            training_partition.loc[:, MODEL_FEATURE_COLUMNS]
        )
        probabilities = artifact.estimator.predict_proba(transformed)[:, 1]

        assert artifact.feature_columns == MODEL_FEATURE_COLUMNS
        assert artifact.target_column == MODEL_TARGET_COLUMN
        assert artifact.hyperparameters
        assert not hasattr(artifact, "threshold")
        assert probabilities.shape == (len(training_partition),)
        assert np.isfinite(probabilities).all()

    first_results = outputs.results_path.read_bytes()

    repeated_outputs = optimize_and_publish_hyperparameters(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        logistic_iterations=1,
        xgboost_iterations=1,
        n_folds=2,
        assessment_days=2,
        gap_days=1,
        card_precision_k=1,
        random_state=42,
    )

    assert repeated_outputs.results_path.read_bytes() == first_results
    assert not list(output_dir.glob(".*.tmp"))


def test_optimize_and_publish_hyperparameters_requires_training_partition(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="train.parquet",
    ):
        optimize_and_publish_hyperparameters(
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "models",
            logistic_iterations=1,
            xgboost_iterations=1,
            n_folds=2,
            assessment_days=2,
            gap_days=1,
            card_precision_k=1,
            random_state=42,
        )
