from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.bronze import (
    ingest_bronze_directory,
    ingest_bronze_file,
)

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def test_ingest_bronze_file_writes_one_idempotent_parquet(
    tmp_path: Path,
) -> None:
    expected_source = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_file = raw_dir / "2018-04-01.pkl"
    expected_source.to_pickle(source_file)

    bronze_dir = tmp_path / "bronze"
    first_ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    second_ingested_at = datetime(2026, 9, 3, 13, 0, tzinfo=UTC)

    output_path = ingest_bronze_file(
        source_file=source_file,
        bronze_dir=bronze_dir,
        ingested_at_utc=first_ingested_at,
    )
    repeated_output_path = ingest_bronze_file(
        source_file=source_file,
        bronze_dir=bronze_dir,
        ingested_at_utc=second_ingested_at,
    )

    assert output_path == bronze_dir / "2018-04-01.parquet"
    assert repeated_output_path == output_path
    assert sorted(bronze_dir.glob("*.parquet")) == [output_path]

    actual = pd.read_parquet(output_path)
    source_columns = list(expected_source.columns)

    pd.testing.assert_frame_equal(
        actual[source_columns],
        expected_source,
    )
    assert actual["ingested_at_utc"].tolist() == [first_ingested_at] * len(expected_source)


def test_ingest_bronze_directory_processes_files_in_date_order(
    tmp_path: Path,
) -> None:
    source_data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_data.to_pickle(raw_dir / "2018-04-02.pkl")
    source_data.to_pickle(raw_dir / "2018-04-01.pkl")

    bronze_dir = tmp_path / "bronze"
    ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    output_paths = ingest_bronze_directory(
        raw_dir=raw_dir,
        bronze_dir=bronze_dir,
        ingested_at_utc=ingested_at,
    )

    assert [path.name for path in output_paths] == [
        "2018-04-01.parquet",
        "2018-04-02.parquet",
    ]
    assert all(path.exists() for path in output_paths)


def test_ingest_bronze_file_does_not_publish_invalid_schema(
    tmp_path: Path,
) -> None:
    source_data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    source_data = source_data.drop(columns=["TX_AMOUNT"])

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_file = raw_dir / "2018-04-01.pkl"
    source_data.to_pickle(source_file)

    bronze_dir = tmp_path / "bronze"
    output_path = bronze_dir / "2018-04-01.parquet"
    ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    with pytest.raises(
        ValueError,
        match="missing required columns: TX_AMOUNT",
    ):
        ingest_bronze_file(
            source_file=source_file,
            bronze_dir=bronze_dir,
            ingested_at_utc=ingested_at,
        )

    assert not output_path.exists()
