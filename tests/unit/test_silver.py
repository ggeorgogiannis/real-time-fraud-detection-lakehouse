from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.bronze import SOURCE_COLUMNS
from fraud_lakehouse.silver import (
    BRONZE_METADATA_COLUMNS,
    SOURCE_TO_SILVER_COLUMNS,
    validate_bronze_records,
)

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def _bronze_records(fixture_name: str) -> pd.DataFrame:
    data = pd.read_csv(FIXTURES_DIR / fixture_name)
    data["source_file"] = "2018-04-01.pkl"
    data["source_file_date"] = date(2018, 4, 1)
    data["source_row_number"] = pd.Series(
        range(len(data)),
        dtype="int64",
    )
    data["ingested_at_utc"] = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    return data


def test_validate_bronze_records_converts_valid_rows_to_silver() -> None:
    bronze = _bronze_records("transactions_valid.csv")

    result = validate_bronze_records(bronze)

    assert result.quarantine.empty
    assert result.silver.columns.tolist() == [
        *SOURCE_TO_SILVER_COLUMNS.values(),
        *BRONZE_METADATA_COLUMNS,
    ]
    assert result.silver["transaction_id"].tolist() == [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]
    assert str(result.silver["transaction_id"].dtype) == "int64"
    assert str(result.silver["customer_id"].dtype) == "int64"
    assert str(result.silver["terminal_id"].dtype) == "int64"
    assert str(result.silver["tx_amount"].dtype) == "float64"
    assert str(result.silver["tx_time_seconds"].dtype) == "int64"
    assert str(result.silver["tx_time_days"].dtype) == "int64"
    assert str(result.silver["tx_fraud"].dtype) == "int8"
    assert str(result.silver["tx_fraud_scenario"].dtype) == "int8"
    assert str(result.silver["tx_datetime"].dtype).endswith(", UTC]")


def test_validate_bronze_records_routes_invalid_rows_to_quarantine() -> None:
    bronze = _bronze_records("transactions_invalid.csv")

    result = validate_bronze_records(bronze)

    assert result.silver.empty
    assert result.quarantine.columns.tolist() == [
        *SOURCE_COLUMNS,
        *BRONZE_METADATA_COLUMNS,
        "rejection_reasons",
    ]

    actual_reasons = dict(
        zip(
            result.quarantine["TRANSACTION_ID"],
            result.quarantine["rejection_reasons"],
            strict=True,
        )
    )
    assert actual_reasons == {
        2001: ["missing_required_value"],
        2002: ["negative_transaction_amount"],
        2003: ["invalid_fraud_label"],
        2004: ["invalid_fraud_scenario"],
        2005: ["inconsistent_time_fields"],
        2006: ["invalid_data_type"],
        2007: ["source_date_mismatch"],
    }


def test_validate_bronze_records_retains_first_valid_duplicate() -> None:
    bronze = _bronze_records("transactions_duplicates.csv")

    result = validate_bronze_records(bronze)

    assert result.silver["transaction_id"].tolist() == [3001, 3002, 3003]
    assert result.quarantine["source_row_number"].tolist() == [1, 3]
    assert result.quarantine["rejection_reasons"].tolist() == [
        ["duplicate_transaction_id"],
        ["duplicate_transaction_id"],
    ]


def test_validate_bronze_records_accumulates_rejection_reasons() -> None:
    bronze = _bronze_records("transactions_valid.csv").iloc[[0]].copy()
    bronze.loc[:, "TX_AMOUNT"] = -1.0
    bronze.loc[:, "TX_TIME_DAYS"] = 2

    result = validate_bronze_records(bronze)

    assert result.silver.empty
    assert result.quarantine["rejection_reasons"].iloc[0] == [
        "negative_transaction_amount",
        "inconsistent_time_fields",
    ]


def test_validate_bronze_records_rejects_known_transaction_id() -> None:
    bronze = _bronze_records("transactions_valid.csv")

    result = validate_bronze_records(
        bronze,
        known_transaction_ids={1001},
    )

    assert result.silver["transaction_id"].tolist() == [1002, 1003, 1004, 1005]
    assert result.quarantine["TRANSACTION_ID"].tolist() == [1001]
    assert result.quarantine["rejection_reasons"].iloc[0] == ["duplicate_transaction_id"]


def test_validate_bronze_records_rejects_missing_bronze_column() -> None:
    bronze = _bronze_records("transactions_valid.csv").drop(columns=["TX_AMOUNT"])

    with pytest.raises(
        ValueError,
        match="missing required columns: TX_AMOUNT",
    ):
        validate_bronze_records(bronze)
