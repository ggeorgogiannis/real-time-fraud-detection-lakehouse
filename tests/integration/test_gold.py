from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.gold import write_gold_tables


def _silver_records() -> pd.DataFrame:
    ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "tx_datetime": pd.to_datetime(
                [
                    "2018-04-01 08:00:00",
                    "2018-04-01 09:00:00",
                    "2018-04-02 10:00:00",
                ],
                utc=True,
            ),
            "customer_id": [1, 1, 2],
            "terminal_id": [10, 10, 11],
            "tx_amount": [10.0, 30.0, 20.0],
            "tx_time_seconds": [28800, 32400, 122400],
            "tx_time_days": [0, 0, 1],
            "tx_fraud": [0, 1, 0],
            "tx_fraud_scenario": [0, 1, 0],
            "source_file": [
                "2018-04-01.pkl",
                "2018-04-01.pkl",
                "2018-04-02.pkl",
            ],
            "source_file_date": [
                date(2018, 4, 1),
                date(2018, 4, 1),
                date(2018, 4, 2),
            ],
            "source_row_number": [0, 1, 0],
            "ingested_at_utc": [ingested_at] * 3,
        }
    )


def _write_silver_files(
    silver_dir: Path,
    silver_data: pd.DataFrame,
) -> None:
    silver_dir.mkdir(parents=True, exist_ok=True)

    for source_date, daily_data in silver_data.groupby("source_file_date"):
        output_path = silver_dir / f"{source_date.isoformat()}.parquet"
        daily_data.to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
        )


def test_write_gold_tables_publishes_reproducible_outputs(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    _write_silver_files(silver_dir, _silver_records())

    output_paths = write_gold_tables(silver_dir, gold_dir)

    assert set(output_paths) == {
        "transaction_features",
        "customer_daily_summary",
        "terminal_daily_summary",
    }
    assert all(path.exists() for path in output_paths.values())

    first_features = pd.read_parquet(output_paths["transaction_features"])
    first_customer_summary = pd.read_parquet(output_paths["customer_daily_summary"])
    first_terminal_summary = pd.read_parquet(output_paths["terminal_daily_summary"])

    assert first_features["transaction_id"].tolist() == [1, 2, 3]
    assert len(first_customer_summary) == 2
    assert len(first_terminal_summary) == 2

    write_gold_tables(silver_dir, gold_dir)

    pd.testing.assert_frame_equal(
        pd.read_parquet(output_paths["transaction_features"]),
        first_features,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_paths["customer_daily_summary"]),
        first_customer_summary,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_paths["terminal_daily_summary"]),
        first_terminal_summary,
    )


def test_write_gold_tables_does_not_publish_invalid_input(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    invalid_data = _silver_records().drop(columns=["tx_amount"])
    _write_silver_files(silver_dir, invalid_data)

    with pytest.raises(
        ValueError,
        match="missing required columns: tx_amount",
    ):
        write_gold_tables(silver_dir, gold_dir)

    assert not gold_dir.exists()
