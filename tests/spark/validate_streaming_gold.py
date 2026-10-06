import math
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.streaming_gold import (
    build_customer_risk_windows,
    build_terminal_risk_windows,
)

CUSTOMER_COLUMNS = [
    "window_start",
    "window_end",
    "customer_id",
    "transaction_count",
    "total_amount",
    "average_amount",
    "maximum_amount",
    "approx_unique_terminals",
    "night_transaction_count",
]

TERMINAL_COLUMNS = [
    "window_start",
    "window_end",
    "terminal_id",
    "transaction_count",
    "approx_unique_customers",
    "total_amount",
    "average_amount",
    "maximum_amount",
    "night_transaction_count",
]


def _silver_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField(
                "transaction_id",
                T.LongType(),
                nullable=False,
            ),
            T.StructField(
                "tx_datetime",
                T.TimestampType(),
                nullable=False,
            ),
            T.StructField(
                "customer_id",
                T.LongType(),
                nullable=False,
            ),
            T.StructField(
                "terminal_id",
                T.LongType(),
                nullable=False,
            ),
            T.StructField(
                "tx_amount",
                T.DoubleType(),
                nullable=False,
            ),
            T.StructField(
                "tx_fraud",
                T.ByteType(),
                nullable=False,
            ),
            T.StructField(
                "tx_fraud_scenario",
                T.ByteType(),
                nullable=False,
            ),
        ]
    )


def _assert_close(
    actual: float,
    expected: float,
) -> None:
    assert math.isclose(
        actual,
        expected,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _assert_rejects_empty_interval(
    silver_events,
) -> None:
    try:
        build_customer_risk_windows(
            silver_events,
            window_duration=" ",
        )
    except ValueError as error:
        assert str(error) == "window_duration must not be empty"
    else:
        raise AssertionError("Empty customer window duration was accepted")

    try:
        build_terminal_risk_windows(
            silver_events,
            watermark_delay="",
        )
    except ValueError as error:
        assert str(error) == "watermark_delay must not be empty"
    else:
        raise AssertionError("Empty terminal watermark delay was accepted")


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-gold-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        silver_events = spark.createDataFrame(
            [
                (
                    1001,
                    datetime(2018, 4, 1, 8, 5),
                    101,
                    201,
                    25.0,
                    0,
                    0,
                ),
                (
                    1002,
                    datetime(2018, 4, 1, 8, 20),
                    101,
                    202,
                    75.0,
                    1,
                    1,
                ),
                (
                    1003,
                    datetime(2018, 4, 1, 8, 45),
                    102,
                    201,
                    100.0,
                    1,
                    2,
                ),
                (
                    1004,
                    datetime(2018, 4, 1, 9, 5),
                    101,
                    201,
                    50.0,
                    0,
                    0,
                ),
                (
                    1005,
                    datetime(2018, 4, 1, 2, 15),
                    101,
                    203,
                    10.0,
                    1,
                    3,
                ),
            ],
            schema=_silver_schema(),
        )

        customer_windows = build_customer_risk_windows(
            silver_events,
        )
        terminal_windows = build_terminal_risk_windows(
            silver_events,
        )

        assert customer_windows.columns == CUSTOMER_COLUMNS
        assert terminal_windows.columns == TERMINAL_COLUMNS

        assert all("fraud" not in column.lower() for column in customer_windows.columns)
        assert all("fraud" not in column.lower() for column in terminal_windows.columns)

        customer_rows = {
            (
                row["window_start"],
                row["customer_id"],
            ): row.asDict()
            for row in customer_windows.collect()
        }

        assert len(customer_rows) == 4

        customer_101_morning = customer_rows[
            (
                datetime(2018, 4, 1, 8, 0),
                101,
            )
        ]
        assert customer_101_morning["window_end"] == datetime(
            2018,
            4,
            1,
            9,
            0,
        )
        assert customer_101_morning["transaction_count"] == 2
        _assert_close(
            customer_101_morning["total_amount"],
            100.0,
        )
        _assert_close(
            customer_101_morning["average_amount"],
            50.0,
        )
        _assert_close(
            customer_101_morning["maximum_amount"],
            75.0,
        )
        assert customer_101_morning["approx_unique_terminals"] == 2
        assert customer_101_morning["night_transaction_count"] == 0

        customer_102_morning = customer_rows[
            (
                datetime(2018, 4, 1, 8, 0),
                102,
            )
        ]
        assert customer_102_morning["transaction_count"] == 1
        _assert_close(
            customer_102_morning["total_amount"],
            100.0,
        )
        assert customer_102_morning["approx_unique_terminals"] == 1
        assert customer_102_morning["night_transaction_count"] == 0

        customer_101_next_hour = customer_rows[
            (
                datetime(2018, 4, 1, 9, 0),
                101,
            )
        ]
        assert customer_101_next_hour["transaction_count"] == 1
        _assert_close(
            customer_101_next_hour["total_amount"],
            50.0,
        )

        customer_101_night = customer_rows[
            (
                datetime(2018, 4, 1, 2, 0),
                101,
            )
        ]
        assert customer_101_night["transaction_count"] == 1
        assert customer_101_night["night_transaction_count"] == 1
        _assert_close(
            customer_101_night["total_amount"],
            10.0,
        )

        terminal_rows = {
            (
                row["window_start"],
                row["terminal_id"],
            ): row.asDict()
            for row in terminal_windows.collect()
        }

        assert len(terminal_rows) == 4

        terminal_201_morning = terminal_rows[
            (
                datetime(2018, 4, 1, 8, 0),
                201,
            )
        ]
        assert terminal_201_morning["window_end"] == datetime(
            2018,
            4,
            1,
            9,
            0,
        )
        assert terminal_201_morning["transaction_count"] == 2
        assert terminal_201_morning["approx_unique_customers"] == 2
        _assert_close(
            terminal_201_morning["total_amount"],
            125.0,
        )
        _assert_close(
            terminal_201_morning["average_amount"],
            62.5,
        )
        _assert_close(
            terminal_201_morning["maximum_amount"],
            100.0,
        )
        assert terminal_201_morning["night_transaction_count"] == 0

        terminal_201_next_hour = terminal_rows[
            (
                datetime(2018, 4, 1, 9, 0),
                201,
            )
        ]
        assert terminal_201_next_hour["transaction_count"] == 1
        assert terminal_201_next_hour["approx_unique_customers"] == 1
        _assert_close(
            terminal_201_next_hour["total_amount"],
            50.0,
        )

        terminal_202_morning = terminal_rows[
            (
                datetime(2018, 4, 1, 8, 0),
                202,
            )
        ]
        assert terminal_202_morning["transaction_count"] == 1
        assert terminal_202_morning["approx_unique_customers"] == 1
        _assert_close(
            terminal_202_morning["total_amount"],
            75.0,
        )

        terminal_203_night = terminal_rows[
            (
                datetime(2018, 4, 1, 2, 0),
                203,
            )
        ]
        assert terminal_203_night["transaction_count"] == 1
        assert terminal_203_night["approx_unique_customers"] == 1
        assert terminal_203_night["night_transaction_count"] == 1
        _assert_close(
            terminal_203_night["total_amount"],
            10.0,
        )

        _assert_rejects_empty_interval(silver_events)

        print("Spark streaming Gold aggregation validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
