from pathlib import Path

import pandas as pd
import pytest

from fraud_lakehouse.analytics import build_analytics_database


def test_build_analytics_database_requires_silver_files(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    silver_dir.mkdir()
    gold_dir.mkdir()

    with pytest.raises(ValueError, match="No Silver Parquet files"):
        build_analytics_database(
            silver_dir,
            gold_dir,
            tmp_path / "analytics.duckdb",
        )


def test_build_analytics_database_requires_all_gold_files(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    silver_dir.mkdir()
    gold_dir.mkdir()

    pd.DataFrame({"transaction_id": [1]}).to_parquet(
        silver_dir / "2018-04-01.parquet",
        index=False,
    )

    with pytest.raises(FileNotFoundError, match="Missing Gold Parquet files"):
        build_analytics_database(
            silver_dir,
            gold_dir,
            tmp_path / "analytics.duckdb",
        )
