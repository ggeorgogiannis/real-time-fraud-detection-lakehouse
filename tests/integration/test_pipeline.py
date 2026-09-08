from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_lakehouse.pipeline import run_batch_pipeline

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def test_batch_pipeline_produces_reproducible_outputs(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    valid_data = pd.read_csv(
        FIXTURES_DIR / "transactions_valid.csv",
        dtype="string",
        keep_default_na=False,
    )
    invalid_data = pd.read_csv(
        FIXTURES_DIR / "transactions_invalid.csv",
        dtype="string",
        keep_default_na=False,
    )

    source_data = pd.concat(
        [valid_data, invalid_data],
        ignore_index=True,
    )
    source_data.to_pickle(raw_dir / "2018-04-01.pkl")

    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    first_result = run_batch_pipeline(
        raw_dir,
        output_dir,
        timestamp,
    )

    assert len(first_result.bronze_files) == 1
    assert len(first_result.silver_files) == 1
    assert len(first_result.quarantine_files) == 1
    assert set(first_result.gold_tables) == {
        "transaction_features",
        "customer_daily_summary",
        "terminal_daily_summary",
    }

    silver_data = pd.read_parquet(first_result.silver_files[0])
    quarantine_data = pd.read_parquet(first_result.quarantine_files[0])

    assert len(silver_data) == 5
    assert len(quarantine_data) == 7

    first_gold_tables = {
        name: pd.read_parquet(path) for name, path in first_result.gold_tables.items()
    }

    second_result = run_batch_pipeline(
        raw_dir,
        output_dir,
        timestamp,
    )

    for name, first_table in first_gold_tables.items():
        second_table = pd.read_parquet(second_result.gold_tables[name])
        pd.testing.assert_frame_equal(first_table, second_table)
