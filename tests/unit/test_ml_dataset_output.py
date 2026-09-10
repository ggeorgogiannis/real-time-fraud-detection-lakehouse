import json
from datetime import UTC, datetime

import pandas as pd
from pandas.testing import assert_frame_equal

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    ModelDatasetSplit,
    write_model_dataset_splits,
)


def _model_dataset_split() -> ModelDatasetSplit:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T12:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T12:00:00Z",
            "2026-01-03T00:00:00Z",
            "2026-01-03T12:00:00Z",
        ],
        utc=True,
    )

    dataset = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "tx_datetime": timestamps,
            "transaction_date": timestamps.date,
            "customer_id": [10, 10, 20, 20, 30, 30],
            "terminal_id": [100, 101, 100, 102, 103, 103],
            "source_file": ["2026-01-01.pkl"] * 6,
            "source_file_date": [datetime(2026, 1, 1).date()] * 6,
            "source_row_number": [1, 2, 3, 4, 5, 6],
            "ingested_at_utc": pd.to_datetime(
                ["2026-01-04T00:00:00Z"] * 6,
                utc=True,
            ),
            "tx_amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "transaction_hour": [0, 12, 0, 12, 0, 12],
            "day_of_week": [3, 3, 4, 4, 5, 5],
            "is_weekend": [0, 0, 0, 0, 1, 1],
            "is_night": [1, 0, 1, 0, 1, 0],
            "customer_previous_transaction_count": [0, 1, 0, 1, 0, 1],
            "customer_previous_mean_amount": [
                None,
                10.0,
                None,
                30.0,
                None,
                50.0,
            ],
            "amount_to_customer_previous_mean": [
                None,
                2.0,
                None,
                4.0 / 3.0,
                None,
                1.2,
            ],
            "terminal_previous_transaction_count": [0, 0, 1, 0, 0, 1],
            "tx_fraud": [0, 1, 0, 0, 1, 1],
        }
    )

    return ModelDatasetSplit(
        train=dataset.iloc[:2].reset_index(drop=True),
        validation=dataset.iloc[2:4].reset_index(drop=True),
        test=dataset.iloc[4:].reset_index(drop=True),
    )


def test_write_model_dataset_splits_creates_parquet_files_and_manifest(
    tmp_path,
) -> None:
    split = _model_dataset_split()
    train_end = datetime(2026, 1, 2, tzinfo=UTC)
    validation_end = datetime(2026, 1, 3, tzinfo=UTC)

    outputs = write_model_dataset_splits(
        split,
        tmp_path,
        train_end=train_end,
        validation_end=validation_end,
    )

    assert outputs.train_path == tmp_path / "train.parquet"
    assert outputs.validation_path == tmp_path / "validation.parquet"
    assert outputs.test_path == tmp_path / "test.parquet"
    assert outputs.metadata_path == tmp_path / "dataset_metadata.json"

    assert_frame_equal(pd.read_parquet(outputs.train_path), split.train)
    assert_frame_equal(
        pd.read_parquet(outputs.validation_path),
        split.validation,
    )
    assert_frame_equal(pd.read_parquet(outputs.test_path), split.test)

    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))

    assert metadata == {
        "schema_version": 1,
        "target_column": "tx_fraud",
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "split_boundaries_utc": {
            "train_end": "2026-01-02T00:00:00Z",
            "validation_end": "2026-01-03T00:00:00Z",
        },
        "partitions": {
            "train": {
                "filename": "train.parquet",
                "row_count": 2,
                "fraud_count": 1,
            },
            "validation": {
                "filename": "validation.parquet",
                "row_count": 2,
                "fraud_count": 0,
            },
            "test": {
                "filename": "test.parquet",
                "row_count": 2,
                "fraud_count": 2,
            },
        },
    }


def test_write_model_dataset_splits_is_repeatable(tmp_path) -> None:
    split = _model_dataset_split()
    train_end = datetime(2026, 1, 2, tzinfo=UTC)
    validation_end = datetime(2026, 1, 3, tzinfo=UTC)

    first_outputs = write_model_dataset_splits(
        split,
        tmp_path,
        train_end=train_end,
        validation_end=validation_end,
    )
    first_metadata = first_outputs.metadata_path.read_bytes()

    second_outputs = write_model_dataset_splits(
        split,
        tmp_path,
        train_end=train_end,
        validation_end=validation_end,
    )

    assert second_outputs.metadata_path.read_bytes() == first_metadata
    assert not list(tmp_path.glob(".*.tmp"))
