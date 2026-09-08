from datetime import UTC, date, datetime

import pandas as pd
import pytest

from fraud_lakehouse.gold import build_gold_tables


def _silver_records() -> pd.DataFrame:
    ingested_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    return pd.DataFrame(
        {
            "transaction_id": [4, 2, 1, 3],
            "tx_datetime": pd.to_datetime(
                [
                    "2018-04-02 08:00:00",
                    "2018-04-01 09:00:00",
                    "2018-04-01 08:00:00",
                    "2018-04-01 10:00:00",
                ],
                utc=True,
            ),
            "customer_id": [1, 1, 1, 2],
            "terminal_id": [11, 10, 10, 11],
            "tx_amount": [50.0, 30.0, 10.0, 20.0],
            "tx_time_seconds": [115200, 32400, 28800, 36000],
            "tx_time_days": [1, 0, 0, 0],
            "tx_fraud": [0, 1, 0, 0],
            "tx_fraud_scenario": [0, 1, 0, 0],
            "source_file": [
                "2018-04-02.pkl",
                "2018-04-01.pkl",
                "2018-04-01.pkl",
                "2018-04-01.pkl",
            ],
            "source_file_date": [
                date(2018, 4, 2),
                date(2018, 4, 1),
                date(2018, 4, 1),
                date(2018, 4, 1),
            ],
            "source_row_number": [0, 1, 0, 2],
            "ingested_at_utc": [ingested_at] * 4,
        }
    )


def test_build_gold_tables_creates_time_aware_features() -> None:
    silver = _silver_records()
    original = silver.copy(deep=True)

    result = build_gold_tables(silver)
    features = result.transaction_features

    pd.testing.assert_frame_equal(silver, original)

    assert features["transaction_id"].tolist() == [1, 2, 3, 4]
    assert features["transaction_hour"].tolist() == [8, 9, 10, 8]
    assert features["day_of_week"].tolist() == [6, 6, 6, 0]
    assert features["is_weekend"].tolist() == [1, 1, 1, 0]
    assert features["is_night"].tolist() == [0, 0, 0, 0]
    assert features["customer_previous_transaction_count"].tolist() == [0, 1, 0, 2]
    assert features["terminal_previous_transaction_count"].tolist() == [0, 1, 0, 1]

    customer_one = features[features["customer_id"] == 1].reset_index(drop=True)

    assert pd.isna(customer_one.loc[0, "customer_previous_mean_amount"])
    assert customer_one.loc[1:, "customer_previous_mean_amount"].tolist() == pytest.approx(
        [10.0, 20.0]
    )
    assert customer_one.loc[1:, "amount_to_customer_previous_mean"].tolist() == pytest.approx(
        [3.0, 2.5]
    )


def test_build_gold_tables_creates_daily_summaries() -> None:
    result = build_gold_tables(_silver_records())

    customer_summary = result.customer_daily_summary
    customer_one_first_day = customer_summary[
        (customer_summary["transaction_date"] == date(2018, 4, 1))
        & (customer_summary["customer_id"] == 1)
    ].iloc[0]

    assert customer_one_first_day["transaction_count"] == 2
    assert customer_one_first_day["total_amount"] == pytest.approx(40.0)
    assert customer_one_first_day["average_amount"] == pytest.approx(20.0)
    assert customer_one_first_day["maximum_amount"] == pytest.approx(30.0)
    assert customer_one_first_day["fraud_count"] == 1
    assert customer_one_first_day["fraud_rate"] == pytest.approx(0.5)

    terminal_summary = result.terminal_daily_summary
    terminal_ten = terminal_summary[terminal_summary["terminal_id"] == 10].iloc[0]

    assert terminal_ten["transaction_count"] == 2
    assert terminal_ten["unique_customers"] == 1
    assert terminal_ten["total_amount"] == pytest.approx(40.0)
    assert terminal_ten["fraud_count"] == 1
    assert terminal_ten["fraud_rate"] == pytest.approx(0.5)


def test_build_gold_tables_rejects_invalid_silver_schema() -> None:
    silver = _silver_records().drop(columns=["tx_amount"])

    with pytest.raises(
        ValueError,
        match="missing required columns: tx_amount",
    ):
        build_gold_tables(silver)


def test_build_gold_tables_rejects_duplicate_transaction_ids() -> None:
    silver = pd.concat(
        [_silver_records(), _silver_records().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate transaction IDs",
    ):
        build_gold_tables(silver)
