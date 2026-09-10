from dataclasses import dataclass
from datetime import datetime

import pandas as pd

MODEL_METADATA_COLUMNS = (
    "transaction_id",
    "tx_datetime",
    "transaction_date",
    "customer_id",
    "terminal_id",
    "source_file",
    "source_file_date",
    "source_row_number",
    "ingested_at_utc",
)

MODEL_FEATURE_COLUMNS = (
    "tx_amount",
    "transaction_hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "customer_previous_transaction_count",
    "customer_previous_mean_amount",
    "amount_to_customer_previous_mean",
    "terminal_previous_transaction_count",
)

MODEL_TARGET_COLUMN = "tx_fraud"

MODEL_DATASET_COLUMNS = (
    *MODEL_METADATA_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)

_NULLABLE_FEATURE_COLUMNS = {
    "customer_previous_mean_amount",
    "amount_to_customer_previous_mean",
}

_BINARY_COLUMNS = (
    "is_weekend",
    "is_night",
)

_COUNT_COLUMNS = (
    "customer_previous_transaction_count",
    "terminal_previous_transaction_count",
)


@dataclass(frozen=True)
class ModelDatasetSplit:
    """Chronological training, validation and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_model_dataset(transaction_features: pd.DataFrame) -> pd.DataFrame:
    """Create a validated, leakage-safe model dataset."""
    missing_columns = sorted(set(MODEL_DATASET_COLUMNS).difference(transaction_features.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    dataset = transaction_features.loc[:, MODEL_DATASET_COLUMNS].copy()

    dataset["tx_datetime"] = _parse_utc_timestamp_column(
        dataset["tx_datetime"],
        "tx_datetime",
    )
    dataset["ingested_at_utc"] = _parse_utc_timestamp_column(
        dataset["ingested_at_utc"],
        "ingested_at_utc",
    )

    for column in ("transaction_date", "source_file_date"):
        parsed_dates = pd.to_datetime(dataset[column], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"{column} must contain valid dates.")
        dataset[column] = parsed_dates.dt.date

    numeric_columns = (*MODEL_FEATURE_COLUMNS, MODEL_TARGET_COLUMN)
    for column in numeric_columns:
        converted = pd.to_numeric(dataset[column], errors="coerce")
        invalid_values = dataset[column].notna() & converted.isna()

        if invalid_values.any():
            raise ValueError(f"{column} must contain numeric values.")

        dataset[column] = converted

    non_nullable_columns = [
        column for column in MODEL_DATASET_COLUMNS if column not in _NULLABLE_FEATURE_COLUMNS
    ]
    columns_with_nulls = [column for column in non_nullable_columns if dataset[column].isna().any()]
    if columns_with_nulls:
        invalid = ", ".join(columns_with_nulls)
        raise ValueError(f"Columns must not contain missing values: {invalid}")

    if dataset["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique.")

    if not dataset[MODEL_TARGET_COLUMN].isin((0, 1)).all():
        raise ValueError("tx_fraud must contain only 0 or 1.")

    for column in _BINARY_COLUMNS:
        if not dataset[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain only 0 or 1.")

    if not dataset["transaction_hour"].between(0, 23).all():
        raise ValueError("transaction_hour must be between 0 and 23.")

    if not dataset["day_of_week"].between(0, 6).all():
        raise ValueError("day_of_week must be between 0 and 6.")

    if (dataset.loc[:, _COUNT_COLUMNS] < 0).to_numpy().any():
        raise ValueError("Historical transaction counts must be non-negative.")

    if dataset.loc[:, numeric_columns].isin((float("inf"), float("-inf"))).to_numpy().any():
        raise ValueError("Model features must not contain infinite values.")

    has_customer_history = dataset["customer_previous_transaction_count"] > 0
    for column in _NULLABLE_FEATURE_COLUMNS:
        if dataset.loc[has_customer_history, column].isna().any():
            raise ValueError(f"{column} may be missing only when the customer has no history.")

    expected_dates = dataset["tx_datetime"].dt.date
    if not dataset["transaction_date"].eq(expected_dates).all():
        raise ValueError("transaction_date must match the UTC date of tx_datetime.")

    dataset[MODEL_TARGET_COLUMN] = dataset[MODEL_TARGET_COLUMN].astype("int8")

    for column in _BINARY_COLUMNS:
        dataset[column] = dataset[column].astype("int8")

    return dataset.sort_values(
        ["tx_datetime", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def split_model_dataset(
    dataset: pd.DataFrame,
    *,
    train_end: datetime,
    validation_end: datetime,
) -> ModelDatasetSplit:
    """Split a model dataset using explicit UTC time boundaries."""
    train_boundary = _as_utc_boundary(train_end, "train_end")
    validation_boundary = _as_utc_boundary(
        validation_end,
        "validation_end",
    )

    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be earlier than validation_end.")

    if "tx_datetime" not in dataset.columns:
        raise ValueError("The model dataset must contain tx_datetime.")

    ordered = dataset.copy()
    ordered["tx_datetime"] = _parse_utc_timestamp_column(
        ordered["tx_datetime"],
        "tx_datetime",
    )
    ordered = ordered.sort_values(
        ["tx_datetime", "transaction_id"],
        kind="mergesort",
    )

    train = ordered.loc[ordered["tx_datetime"] < train_boundary].reset_index(drop=True)

    validation = ordered.loc[
        (ordered["tx_datetime"] >= train_boundary) & (ordered["tx_datetime"] < validation_boundary)
    ].reset_index(drop=True)

    test = ordered.loc[ordered["tx_datetime"] >= validation_boundary].reset_index(drop=True)

    if train.empty or validation.empty or test.empty:
        raise ValueError("Training, validation and test partitions must all be non-empty.")

    return ModelDatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )


def _parse_utc_timestamp_column(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")

    if parsed.isna().any():
        raise ValueError(f"{column_name} must contain valid timestamps.")

    return parsed


def _as_utc_boundary(value: datetime, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return timestamp.tz_convert("UTC")
