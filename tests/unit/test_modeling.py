import numpy as np
import pandas as pd
import pytest

from fraud_lakehouse.ml_dataset import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
from fraud_lakehouse.modeling import prepare_model_partitions


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
            "transaction_id": range(1, row_count + 1),
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


def test_prepare_model_partitions_selects_features_and_target() -> None:
    train = _partition([10.0, 20.0, 30.0], [0, 0, 1])
    validation = _partition([40.0, 50.0], [0, 1])
    test = _partition([60.0, 70.0], [1, 0])

    prepared = prepare_model_partitions(
        train=train,
        validation=validation,
        test=test,
    )

    assert prepared.train.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert prepared.validation.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert prepared.test.features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)

    assert prepared.train.target.tolist() == [0, 0, 1]
    assert prepared.validation.target.tolist() == [0, 1]
    assert prepared.test.target.tolist() == [1, 0]


def test_prepare_model_partitions_imputes_missing_historical_values() -> None:
    prepared = prepare_model_partitions(
        train=_partition([10.0, 20.0, 30.0], [0, 0, 1]),
        validation=_partition([40.0, 50.0], [0, 1]),
        test=_partition([60.0, 70.0], [1, 0]),
    )

    for partition in (
        prepared.train,
        prepared.validation,
        prepared.test,
    ):
        values = partition.features.to_numpy()

        assert not partition.features.isna().any().any()
        assert np.isfinite(values).all()


def test_prepare_model_partitions_fits_preprocessor_on_training_only() -> None:
    prepared = prepare_model_partitions(
        train=_partition([10.0, 20.0, 30.0], [0, 0, 1]),
        validation=_partition([10_000.0, 20_000.0], [0, 1]),
        test=_partition([30_000.0, 40_000.0], [1, 0]),
    )

    amount_index = list(MODEL_FEATURE_COLUMNS).index("tx_amount")
    scaler = prepared.preprocessor.named_steps["scaler"]

    assert scaler.mean_[amount_index] == pytest.approx(20.0)


def test_prepare_model_partitions_rejects_missing_columns() -> None:
    train = _partition([10.0, 20.0, 30.0], [0, 0, 1]).drop(columns="tx_amount")

    with pytest.raises(
        ValueError,
        match="Missing required columns: tx_amount",
    ):
        prepare_model_partitions(
            train=train,
            validation=_partition([40.0, 50.0], [0, 1]),
            test=_partition([60.0, 70.0], [1, 0]),
        )
