from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from dbt.cli.main import dbtRunner

from fraud_lakehouse.analytics import build_analytics_database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt"


def test_dbt_build_creates_valid_analytical_marts(
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
            "transaction_id": [1, 2, 3, 4],
            "tx_datetime": pd.to_datetime(
                [
                    "2018-04-01 08:00:00+00:00",
                    "2018-04-01 09:00:00+00:00",
                    "2018-04-02 01:30:00+02:00",
                    "2018-04-02 10:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [101, 101, 102, 101],
            "terminal_id": [201, 202, 201, 201],
            "tx_amount": [10.0, 20.0, 30.0, 40.0],
            "tx_fraud": [0, 1, 0, 1],
        }
    ).to_parquet(
        silver_dir / "2018-04-01.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "transaction_date": [
                date(2018, 4, 1),
                date(2018, 4, 1),
                date(2018, 4, 1),
                date(2018, 4, 2),
            ],
            "tx_amount": [10.0, 20.0, 30.0, 40.0],
            "tx_fraud": [0, 1, 0, 1],
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
                date(2018, 4, 2),
            ],
            "customer_id": [101, 102, 101],
            "transaction_count": [2, 1, 1],
            "total_amount": [30.0, 30.0, 40.0],
            "average_amount": [15.0, 30.0, 40.0],
            "maximum_amount": [20.0, 30.0, 40.0],
            "fraud_count": [1, 0, 1],
            "fraud_rate": [0.5, 0.0, 1.0],
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
                date(2018, 4, 2),
            ],
            "terminal_id": [201, 202, 201],
            "transaction_count": [2, 1, 1],
            "unique_customers": [2, 1, 1],
            "total_amount": [40.0, 20.0, 40.0],
            "average_amount": [20.0, 20.0, 40.0],
            "fraud_count": [0, 1, 1],
            "fraud_rate": [0.0, 1.0, 1.0],
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
        daily_rows = connection.execute(
            """
            select
                transaction_date,
                transaction_count,
                total_amount,
                fraud_count,
                fraud_rate
            from analytics_marts.fct_daily_fraud
            order by transaction_date
            """
        ).fetchall()

        customer_rows = connection.execute(
            """
            select
                transaction_date,
                customer_id,
                rolling_7d_transaction_count,
                rolling_7d_fraud_count,
                rolling_7d_fraud_rate
            from analytics_marts.fct_customer_daily_risk
            order by transaction_date, customer_id
            """
        ).fetchall()

        terminal_rows = connection.execute(
            """
            select
                transaction_date,
                terminal_id,
                rolling_7d_transaction_count,
                rolling_7d_fraud_count,
                rolling_7d_fraud_rate
            from analytics_marts.fct_terminal_daily_risk
            order by transaction_date, terminal_id
            """
        ).fetchall()

        overview_rows = connection.execute(
            """
            select
                transaction_date,
                transaction_count,
                fraud_count,
                active_customer_count,
                customers_with_fraud,
                active_terminal_count,
                terminals_with_fraud
            from analytics_marts.rpt_daily_fraud_overview
            order by transaction_date
            """
        ).fetchall()

    assert daily_rows[0][:4] == (date(2018, 4, 1), 3, 60.0, 1)
    assert daily_rows[0][4] == pytest.approx(1 / 3)
    assert daily_rows[1] == (date(2018, 4, 2), 1, 40.0, 1, 1.0)

    assert customer_rows[:2] == [
        (date(2018, 4, 1), 101, 2, 1, 0.5),
        (date(2018, 4, 1), 102, 1, 0, 0.0),
    ]
    assert customer_rows[2][:4] == (date(2018, 4, 2), 101, 3, 2)
    assert customer_rows[2][4] == pytest.approx(2 / 3)

    assert terminal_rows[:2] == [
        (date(2018, 4, 1), 201, 2, 0, 0.0),
        (date(2018, 4, 1), 202, 1, 1, 1.0),
    ]
    assert terminal_rows[2][:4] == (date(2018, 4, 2), 201, 3, 1)
    assert terminal_rows[2][4] == pytest.approx(1 / 3)

    assert overview_rows == [
        (date(2018, 4, 1), 3, 1, 2, 1, 2, 1),
        (date(2018, 4, 2), 1, 1, 1, 1, 1, 1),
    ]
