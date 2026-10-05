from datetime import UTC, date, datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.streaming_events import build_transaction_event
from fraud_lakehouse.streaming_transformations import (
    parse_kafka_transaction_events,
    project_transaction_events,
    transaction_event_schema,
)


def _transaction() -> dict[str, object]:
    return {
        "TRANSACTION_ID": 1001,
        "TX_DATETIME": pd.Timestamp("2018-04-01 08:00:00"),
        "CUSTOMER_ID": 101,
        "TERMINAL_ID": 201,
        "TX_AMOUNT": 25.5,
        "TX_TIME_SECONDS": 28800,
        "TX_TIME_DAYS": 0,
        "TX_FRAUD": 0,
        "TX_FRAUD_SCENARIO": 0,
    }


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-parsing-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        event = build_transaction_event(
            transaction=_transaction(),
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=0,
            produced_at_utc=datetime(
                2026,
                10,
                5,
                13,
                0,
                tzinfo=UTC,
            ),
        )

        kafka_schema = T.StructType(
            [
                T.StructField("key", T.BinaryType(), nullable=True),
                T.StructField("value", T.BinaryType(), nullable=False),
                T.StructField("topic", T.StringType(), nullable=False),
                T.StructField("partition", T.IntegerType(), nullable=False),
                T.StructField("offset", T.LongType(), nullable=False),
                T.StructField("timestamp", T.TimestampType(), nullable=False),
            ]
        )

        kafka_records = spark.createDataFrame(
            [
                (
                    event.key,
                    event.value,
                    "transactions.raw.v1",
                    1,
                    42,
                    datetime(2026, 10, 5, 13, 0),
                ),
                (
                    b"malformed",
                    b"{not-valid-json",
                    "transactions.raw.v1",
                    2,
                    43,
                    datetime(2026, 10, 5, 13, 0, 1),
                ),
            ],
            schema=kafka_schema,
        )

        parsed = parse_kafka_transaction_events(kafka_records)
        projected = project_transaction_events(parsed)
        rows = projected.orderBy("kafka_offset").collect()

        assert len(transaction_event_schema().fields) == 7
        assert len(rows) == 2

        valid = rows[0].asDict(recursive=True)
        assert valid["kafka_key"] == "1001"
        assert valid["kafka_topic"] == "transactions.raw.v1"
        assert valid["kafka_partition"] == 1
        assert valid["kafka_offset"] == 42
        assert valid["schema_version"] == "1.0"
        assert valid["source_file"] == "2018-04-01.pkl"
        assert valid["source_file_date"] == "2018-04-01"
        assert valid["source_row_number"] == 0
        assert valid["TRANSACTION_ID"] == "1001"
        assert valid["TX_DATETIME"] == "2018-04-01T08:00:00Z"
        assert valid["TX_AMOUNT"] == "25.5"

        malformed = rows[1].asDict(recursive=True)
        assert malformed["kafka_key"] == "malformed"
        assert malformed["raw_event"] == "{not-valid-json"
        assert malformed["schema_version"] is None
        assert malformed["event_id"] is None
        assert malformed["TRANSACTION_ID"] is None

        print("Spark streaming parsing validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
