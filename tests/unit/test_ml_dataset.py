from datetime import UTC, date, datetime

import pandas as pd
import pytest

from fraud_lakehouse.ml_dataset import (
    MODEL_DATASET_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_model_dataset,
    split_model_dataset,
)


def _transaction_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [3, 1, 4, 2],
            "tx_datetime": pd.to_datetime(
                [
                    "2026-01-03T10:00:00Z",
                    "2026-01-01T08:00:00Z",
                    "2026-01-04T11:00:00Z",
                    "2026-01-02T09:00:00Z",
                ],
                utc=True,
            ),
            "transaction_date": [
                date(2026, 1, 3),
                date(2026, 1, 1),
                date(2026, 1, 4),
                date(2026, 1, 2),
            ],
            "customer_id": [102, 101, 101, 101],
            "terminal_id": [201, 201, 202, 201],
            "source_file": [
                "2026-01-03.pkl",
                "2026-01-01.pkl",
                "2026-01-04.pkl",
                "2026-01-02.pkl",
            ],
            "source_file_date": [
                date(2026, 1, 3),
                date(2026, 1, 1),
                date(2026, 1, 4),
                date(2026, 1, 2),
            ],
            "source_row_number": [0, 0, 0, 0],
            "ingested_at_utc": pd.to_datetime(
                ["2026-01-05T00:00:00Z"] * 4,
                utc=True,
            ),
            "tx_amount": [30.0, 10.0, 40.0, 20.0],
            "transaction_hour": [10, 8, 11, 9],
            "day_of_week": [5, 3, 6, 4],
            "is_weekend": [1, 0, 1, 0],
            "is_night": [0, 0, 0, 0],
            "customer_previous_transaction_count": [0, 0, 2, 1],
            "customer_previous_mean_amount": [None, None, 15.0, 10.0],
            "amount_to_customer_previous_mean": [None, None, 40 / 15, 2.0],
            "terminal_previous_transaction_count": [1, 0, 0, 1],
            "tx_fraud": [0, 0, 1, 1],
            "tx_fraud_scenario": [0, 0, 2, 1],
        }
    )


def test_model_features_exclude_target_and_leakage_columns() -> None:
    forbidden_columns = {
        "tx_fraud",
        "tx_fraud_scenario",
        "customer_id",
        "terminal_id",
    }

    assert forbidden_columns.isdisjoint(MODEL_FEATURE_COLUMNS)


def test_build_model_dataset_selects_and_orders_columns() -> None:
    result = build_model_dataset(_transaction_features())

    assert tuple(result.columns) == MODEL_DATASET_COLUMNS
    assert result["transaction_id"].tolist() == [1, 2, 3, 4]
    assert "tx_fraud_scenario" not in result.columns


def test_build_model_dataset_rejects_missing_columns() -> None:
    data = _transaction_features().drop(columns="tx_amount")

    with pytest.raises(ValueError, match="tx_amount"):
        build_model_dataset(data)


def test_build_model_dataset_rejects_duplicate_transaction_ids() -> None:
    data = _transaction_features()
    data.loc[1, "transaction_id"] = 3

    with pytest.raises(ValueError, match="transaction_id"):
        build_model_dataset(data)


def test_build_model_dataset_rejects_invalid_target() -> None:
    data = _transaction_features()
    data.loc[0, "tx_fraud"] = 2

    with pytest.raises(ValueError, match="tx_fraud"):
        build_model_dataset(data)


def test_split_model_dataset_uses_chronological_boundaries() -> None:
    dataset = build_model_dataset(_transaction_features())

    split = split_model_dataset(
        dataset,
        train_end=datetime(2026, 1, 3, tzinfo=UTC),
        validation_end=datetime(2026, 1, 4, tzinfo=UTC),
    )

    assert split.train["transaction_id"].tolist() == [1, 2]
    assert split.validation["transaction_id"].tolist() == [3]
    assert split.test["transaction_id"].tolist() == [4]


def test_split_model_dataset_requires_timezone_aware_boundaries() -> None:
    dataset = build_model_dataset(_transaction_features())

    with pytest.raises(ValueError, match="timezone-aware"):
        split_model_dataset(
            dataset,
            train_end=datetime(2026, 1, 3),
            validation_end=datetime(2026, 1, 4, tzinfo=UTC),
        )


def test_split_model_dataset_rejects_empty_partitions() -> None:
    dataset = build_model_dataset(_transaction_features())

    with pytest.raises(ValueError, match="non-empty"):
        split_model_dataset(
            dataset,
            train_end=datetime(2026, 1, 1, tzinfo=UTC),
            validation_end=datetime(2026, 1, 4, tzinfo=UTC),
        )
