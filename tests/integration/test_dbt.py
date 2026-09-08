from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from dbt.cli.main import dbtRunner

from fraud_lakehouse.analytics import build_analytics_database

PROJECT_ROOT = Path(__file__).parents[2]
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt"


def test_dbt_build_creates_valid_analytical_mart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    database_path = tmp_path / "fraud_lakehouse.duckdb"

    silver_dir.mkdir()
    gold_dir.mkdir()

    pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "tx_datetime": pd.to_datetime(
                [
                    "2018-04-01 08:00:00+00:00",
                    "2018-04-01 09:00:00+00:00",
                    "2018-04-01 10:00:00+00:00",
                ]
            ),
            "customer_id": [101, 101, 102],
            "terminal_id": [201, 202, 201],
            "tx_amount": [10.0, 20.0, 30.0],
            "tx_fraud": [0, 1, 0],
        }
    ).to_parquet(
        silver_dir / "2018-04-01.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
        }
    ).to_parquet(
        gold_dir / "transaction_features.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_date": [
                date(2018, 4, 1),
                date(2018, 4, 1),
            ],
            "customer_id": [101, 102],
        }
    ).to_parquet(
        gold_dir / "customer_daily_summary.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_date": [
                date(2018, 4, 1),
                date(2018, 4, 1),
            ],
            "terminal_id": [201, 202],
        }
    ).to_parquet(
        gold_dir / "terminal_daily_summary.parquet",
        index=False,
    )

    build_analytics_database(
        silver_dir,
        gold_dir,
        database_path,
    )

    monkeypatch.setenv(
        "FRAUD_LAKEHOUSE_DUCKDB_PATH",
        str(database_path),
    )

    result = dbtRunner().invoke(
        [
            "build",
            "--project-dir",
            str(DBT_PROJECT_DIR),
            "--profiles-dir",
            str(DBT_PROJECT_DIR),
        ]
    )

    assert result.success, str(result.exception)

    with duckdb.connect(str(database_path)) as connection:
        daily_fraud = connection.execute(
            """
            SELECT
                transaction_date,
                transaction_count,
                fraud_count,
                fraud_rate
            FROM analytics_marts.fct_daily_fraud
            """
        ).fetchone()

    assert daily_fraud == (
        date(2018, 4, 1),
        3,
        1,
        pytest.approx(1 / 3),
    )
