from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.bronze import (
    build_bronze_records,
    discover_source_files,
    load_trusted_source_file,
)

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def test_discover_source_files_returns_pickles_in_date_order(
    tmp_path: Path,
) -> None:
    newest = tmp_path / "2018-04-03.pkl"
    oldest = tmp_path / "2018-04-01.pkl"
    middle = tmp_path / "2018-04-02.pkl"
    ignored = tmp_path / "notes.txt"

    for path in (newest, ignored, oldest, middle):
        path.touch()

    discovered = discover_source_files(tmp_path)

    assert discovered == [oldest, middle, newest]


@pytest.mark.parametrize(
    "filename",
    [
        "2018-02-30.pkl",
        "2018-4-01.pkl",
        "transactions.pkl",
    ],
)
def test_discover_source_files_rejects_invalid_pickle_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    (tmp_path / filename).touch()

    with pytest.raises(ValueError, match="Invalid source filename"):
        discover_source_files(tmp_path)


def test_load_trusted_source_file_preserves_dataframe(
    tmp_path: Path,
) -> None:
    expected = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    source_file = tmp_path / "2018-04-01.pkl"
    expected.to_pickle(source_file)

    actual = load_trusted_source_file(source_file)

    pd.testing.assert_frame_equal(actual, expected)


def test_load_trusted_source_file_rejects_non_dataframe(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "2018-04-01.pkl"
    pd.to_pickle(["not", "a", "dataframe"], source_file)

    with pytest.raises(TypeError, match="must contain a Pandas DataFrame"):
        load_trusted_source_file(source_file)


def test_load_trusted_source_file_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    data = data.drop(columns=["TX_AMOUNT"])
    source_file = tmp_path / "2018-04-01.pkl"
    data.to_pickle(source_file)

    with pytest.raises(
        ValueError,
        match="missing required columns: TX_AMOUNT",
    ):
        load_trusted_source_file(source_file)


def test_load_trusted_source_file_rejects_unexpected_columns(
    tmp_path: Path,
) -> None:
    data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    data["CARD_ID"] = 42
    source_file = tmp_path / "2018-04-01.pkl"
    data.to_pickle(source_file)

    with pytest.raises(
        ValueError,
        match="unexpected columns: CARD_ID",
    ):
        load_trusted_source_file(source_file)


def test_build_bronze_records_adds_metadata_without_changing_source_values(
    tmp_path: Path,
) -> None:
    expected_source = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    source_file = tmp_path / "2018-04-01.pkl"
    expected_source.to_pickle(source_file)
    ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    actual = build_bronze_records(source_file, ingested_at)

    source_columns = list(expected_source.columns)
    pd.testing.assert_frame_equal(
        actual[source_columns],
        expected_source,
    )
    assert actual["source_file"].tolist() == ["2018-04-01.pkl"] * len(expected_source)
    assert actual["source_file_date"].tolist() == [date(2018, 4, 1)] * len(expected_source)
    assert actual["source_row_number"].tolist() == list(range(len(expected_source)))
    assert str(actual["source_row_number"].dtype) == "int64"
    assert actual["ingested_at_utc"].tolist() == [ingested_at] * len(expected_source)


@pytest.mark.parametrize(
    "ingested_at",
    [
        datetime(2026, 9, 3, 12, 0),
        datetime(
            2026,
            9,
            3,
            14,
            0,
            tzinfo=timezone(timedelta(hours=2)),
        ),
    ],
)
def test_build_bronze_records_rejects_non_utc_timestamp(
    tmp_path: Path,
    ingested_at: datetime,
) -> None:
    data = pd.read_csv(FIXTURES_DIR / "transactions_valid.csv")
    source_file = tmp_path / "2018-04-01.pkl"
    data.to_pickle(source_file)

    with pytest.raises(
        ValueError,
        match="ingested_at_utc must be timezone-aware UTC",
    ):
        build_bronze_records(source_file, ingested_at)
