from pathlib import Path

import duckdb
import pandas as pd

from fraud_lakehouse.analytics import (
    ANALYTICS_VIEW_NAMES,
    build_analytics_database,
)


def test_build_analytics_database_creates_queryable_views(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    database_path = tmp_path / "analytics" / "fraud_lakehouse.duckdb"

    silver_dir.mkdir()
    gold_dir.mkdir()

    pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "tx_fraud": [0, 1],
        }
    ).to_parquet(
        silver_dir / "2018-04-01.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "transaction_hour": [8, 9],
        }
    ).to_parquet(
        gold_dir / "transaction_features.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "customer_id": [101],
            "transaction_count": [2],
        }
    ).to_parquet(
        gold_dir / "customer_daily_summary.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "terminal_id": [201, 202],
            "transaction_count": [1, 1],
        }
    ).to_parquet(
        gold_dir / "terminal_daily_summary.parquet",
        index=False,
    )

    assert build_analytics_database(silver_dir, gold_dir, database_path) == database_path
    assert build_analytics_database(silver_dir, gold_dir, database_path) == database_path

    with duckdb.connect(str(database_path), read_only=True) as connection:
        available_views = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}

        assert set(ANALYTICS_VIEW_NAMES).issubset(available_views)
        assert connection.execute("SELECT COUNT(*) FROM silver_transactions").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM gold_transaction_features").fetchone() == (
            2,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM gold_customer_daily_summary"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM gold_terminal_daily_summary"
        ).fetchone() == (2,)
