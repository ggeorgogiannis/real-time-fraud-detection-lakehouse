from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.silver import (
    ingest_silver_directory,
    ingest_silver_file,
)

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def _write_bronze_file(
    bronze_dir: Path,
    filename: str,
    source_data: pd.DataFrame,
) -> Path:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bronze_file = bronze_dir / filename
    source_date = date.fromisoformat(bronze_file.stem)

    bronze_data = source_data.copy()
    bronze_data["source_file"] = f"{bronze_file.stem}.pkl"
    bronze_data["source_file_date"] = source_date
    bronze_data["source_row_number"] = pd.Series(
        range(len(bronze_data)),
        dtype="int64",
    )
    bronze_data["ingested_at_utc"] = datetime(
        2026,
        9,
        3,
        12,
        0,
        tzinfo=UTC,
    )
    bronze_data.to_parquet(bronze_file, engine="pyarrow", index=False)

    return bronze_file


def test_ingest_silver_file_writes_outputs_and_is_idempotent(
    tmp_path: Path,
) -> None:
    valid_data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    invalid_data = pd.read_csv(FIXTURES_DIR / "transactions_invalid.csv")
    source_data = pd.concat(
        [valid_data, invalid_data],
        ignore_index=True,
    )
    source_data["CUSTOMER_ID"] = source_data["CUSTOMER_ID"].astype("string")
    bronze_file = _write_bronze_file(
        tmp_path / "bronze",
        "2018-04-01.parquet",
        source_data,
    )
    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"

    silver_output, quarantine_output = ingest_silver_file(
        bronze_file,
        silver_dir,
        quarantine_dir,
    )

    silver_data = pd.read_parquet(silver_output)
    quarantine_data = pd.read_parquet(quarantine_output)

    assert silver_data["transaction_id"].tolist() == [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]
    assert len(quarantine_data) == 7

    rejection_reasons = {
        reason for row_reasons in quarantine_data["rejection_reasons"] for reason in row_reasons
    }
    assert rejection_reasons == {
        "missing_required_value",
        "negative_transaction_amount",
        "invalid_fraud_label",
        "invalid_fraud_scenario",
        "inconsistent_time_fields",
        "invalid_data_type",
        "source_date_mismatch",
    }

    original_modification_times = (
        silver_output.stat().st_mtime_ns,
        quarantine_output.stat().st_mtime_ns,
    )

    repeated_outputs = ingest_silver_file(
        bronze_file,
        silver_dir,
        quarantine_dir,
    )

    assert repeated_outputs == (silver_output, quarantine_output)
    assert (
        silver_output.stat().st_mtime_ns,
        quarantine_output.stat().st_mtime_ns,
    ) == original_modification_times


def test_ingest_silver_directory_deduplicates_across_files(
    tmp_path: Path,
) -> None:
    valid_data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")

    first_day = valid_data.iloc[[0]].copy()
    second_day = valid_data.iloc[[0, 1]].copy()
    second_day["TX_DATETIME"] = pd.to_datetime(second_day["TX_DATETIME"]) + pd.Timedelta(days=1)
    second_day["TX_TIME_SECONDS"] += 86400
    second_day["TX_TIME_DAYS"] += 1

    bronze_dir = tmp_path / "bronze"
    _write_bronze_file(
        bronze_dir,
        "2018-04-02.parquet",
        second_day,
    )
    _write_bronze_file(
        bronze_dir,
        "2018-04-01.parquet",
        first_day,
    )

    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"

    outputs = ingest_silver_directory(
        bronze_dir,
        silver_dir,
        quarantine_dir,
    )

    assert [silver_path.name for silver_path, _ in outputs] == [
        "2018-04-01.parquet",
        "2018-04-02.parquet",
    ]

    second_silver = pd.read_parquet(silver_dir / "2018-04-02.parquet")
    second_quarantine = pd.read_parquet(quarantine_dir / "2018-04-02.parquet")

    assert second_silver["transaction_id"].tolist() == [1002]
    assert second_quarantine["TRANSACTION_ID"].tolist() == [1001]
    assert list(second_quarantine["rejection_reasons"].iloc[0]) == ["duplicate_transaction_id"]

    ingest_silver_directory(
        bronze_dir,
        silver_dir,
        quarantine_dir,
    )

    assert len(list(silver_dir.glob("*.parquet"))) == 2
    assert len(list(quarantine_dir.glob("*.parquet"))) == 2


def test_ingest_silver_file_does_not_publish_invalid_schema(
    tmp_path: Path,
) -> None:
    source_data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv").drop(columns=["TX_AMOUNT"])

    bronze_file = _write_bronze_file(
        tmp_path / "bronze",
        "2018-04-01.parquet",
        source_data,
    )
    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"

    with pytest.raises(
        ValueError,
        match="missing required columns: TX_AMOUNT",
    ):
        ingest_silver_file(
            bronze_file,
            silver_dir,
            quarantine_dir,
        )

    assert not silver_dir.exists()
    assert not quarantine_dir.exists()
