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
from fraud_lakehouse.training import train_and_publish_baselines


def _partition(
    amounts: list[float],
    fraud_labels: list[int],
) -> pd.DataFrame:
    row_count = len(amounts)
    previous_means = [None if index == 0 else amounts[index - 1] for index in range(row_count)]
    amount_ratios = [
        None if previous_mean is None else amount / previous_mean
        for amount, previous_mean in zip(
            amounts,
            previous_means,
            strict=True,
        )
    ]

    return pd.DataFrame(
        {
            "tx_amount": amounts,
            "transaction_hour": range(row_count),
            "day_of_week": range(row_count),
            "is_weekend": [0] * row_count,
            "is_night": [1, *([0] * (row_count - 1))],
            "customer_previous_transaction_count": range(row_count),
            "customer_previous_mean_amount": previous_means,
            "amount_to_customer_previous_mean": amount_ratios,
            "terminal_previous_transaction_count": range(row_count),
            MODEL_TARGET_COLUMN: fraud_labels,
        }
    )


def _write_model_partitions(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)

    partitions = {
        "train": _partition(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [0, 0, 0, 1, 0, 1],
        ),
        "validation": _partition(
            [15.0, 25.0, 45.0, 55.0],
            [0, 0, 1, 1],
        ),
        "test": _partition(
            [12.0, 22.0, 42.0, 62.0],
            [0, 0, 1, 1],
        ),
    }

    for name, partition in partitions.items():
        partition.to_parquet(
            dataset_dir / f"{name}.parquet",
            engine="pyarrow",
            index=False,
        )


def test_train_and_publish_baselines_writes_models_and_metrics(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "models"
    _write_model_partitions(dataset_dir)

    outputs = train_and_publish_baselines(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        threshold=0.5,
    )

    assert outputs.dummy_model_path == output_dir / "dummy_prior.joblib"
    assert outputs.logistic_model_path == (output_dir / "logistic_regression.joblib")
    assert outputs.metrics_path == output_dir / "metrics.json"

    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))

    assert metrics["schema_version"] == 1
    assert metrics["threshold"] == 0.5
    assert metrics["primary_metric"] == "average_precision"
    assert set(metrics["models"]) == {
        "dummy_prior",
        "logistic_regression",
    }

    for model_metrics in metrics["models"].values():
        assert set(model_metrics) == {"validation", "test"}
        assert 0.0 <= model_metrics["validation"]["average_precision"] <= 1.0
        assert 0.0 <= model_metrics["test"]["average_precision"] <= 1.0

    artifact = joblib.load(outputs.logistic_model_path)
    test_partition = pd.read_parquet(dataset_dir / "test.parquet")
    transformed = artifact.preprocessor.transform(test_partition.loc[:, MODEL_FEATURE_COLUMNS])
    probabilities = artifact.estimator.predict_proba(transformed)[:, 1]

    assert artifact.feature_columns == MODEL_FEATURE_COLUMNS
    assert artifact.target_column == MODEL_TARGET_COLUMN
    assert artifact.threshold == 0.5
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()

    first_metrics = outputs.metrics_path.read_bytes()

    repeated_outputs = train_and_publish_baselines(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        threshold=0.5,
    )

    assert repeated_outputs.metrics_path.read_bytes() == first_metrics
    assert not list(output_dir.glob(".*.tmp"))


def test_train_and_publish_baselines_requires_all_partitions(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    _partition(
        [10.0, 20.0, 30.0],
        [0, 0, 1],
    ).to_parquet(
        dataset_dir / "train.parquet",
        engine="pyarrow",
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="validation.parquet",
    ):
        train_and_publish_baselines(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "models",
        )
