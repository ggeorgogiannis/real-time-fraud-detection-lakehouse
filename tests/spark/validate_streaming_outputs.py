from datetime import UTC, date, datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.bronze import SOURCE_COLUMNS
from fraud_lakehouse.streaming_events import build_transaction_event
from fraud_lakehouse.streaming_transformations import (
    parse_kafka_transaction_events,
    project_streaming_bronze,
    project_streaming_quarantine,
    project_streaming_silver,
    project_transaction_events,
    select_quarantined_transaction_events,
    select_valid_transaction_events,
    type_transaction_events,
    validate_transaction_events,
)

BRONZE_COLUMNS = [
    "kafka_key",
    "raw_event",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "ingested_at_utc",
]

SILVER_COLUMNS = [
    "transaction_id",
    "tx_datetime",
    "customer_id",
    "terminal_id",
    "tx_amount",
    "tx_time_seconds",
    "tx_time_days",
    "tx_fraud",
    "tx_fraud_scenario",
    "source_file",
    "source_file_date",
    "source_row_number",
    "ingested_at_utc",
    "schema_version",
    "event_id",
    "produced_at_utc",
    "kafka_key",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
]

QUARANTINE_COLUMNS = [
    *SOURCE_COLUMNS,
    "source_file",
    "source_file_date",
    "source_row_number",
    "ingested_at_utc",
    "schema_version",
    "event_id",
    "produced_at_utc",
    "kafka_key",
    "raw_event",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "rejection_reasons",
]


def _transaction(
    *,
    transaction_id: int,
    transaction_amount: float,
) -> dict[str, object]:
    return {
        "TRANSACTION_ID": transaction_id,
        "TX_DATETIME": pd.Timestamp("2018-04-01 08:00:00"),
        "CUSTOMER_ID": 101,
        "TERMINAL_ID": 201,
        "TX_AMOUNT": transaction_amount,
        "TX_TIME_SECONDS": 28800,
        "TX_TIME_DAYS": 0,
        "TX_FRAUD": 0,
        "TX_FRAUD_SCENARIO": 0,
    }


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-output-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        valid_event = build_transaction_event(
            transaction=_transaction(
                transaction_id=1001,
                transaction_amount=25.5,
            ),
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
        invalid_event = build_transaction_event(
            transaction=_transaction(
                transaction_id=1002,
                transaction_amount=-10.0,
            ),
            source_file="2018-04-01.pkl",
            source_file_date=date(2018, 4, 1),
            source_row_number=1,
            produced_at_utc=datetime(
                2026,
                10,
                5,
                13,
                0,
                1,
                tzinfo=UTC,
            ),
        )

        kafka_schema = T.StructType(
            [
                T.StructField(
                    "key",
                    T.BinaryType(),
                    nullable=True,
                ),
                T.StructField(
                    "value",
                    T.BinaryType(),
                    nullable=False,
                ),
                T.StructField(
                    "topic",
                    T.StringType(),
                    nullable=False,
                ),
                T.StructField(
                    "partition",
                    T.IntegerType(),
                    nullable=False,
                ),
                T.StructField(
                    "offset",
                    T.LongType(),
                    nullable=False,
                ),
                T.StructField(
                    "timestamp",
                    T.TimestampType(),
                    nullable=False,
                ),
            ]
        )

        valid_kafka_timestamp = datetime(
            2026,
            10,
            5,
            13,
            0,
        )
        invalid_kafka_timestamp = datetime(
            2026,
            10,
            5,
            13,
            0,
            1,
        )

        kafka_records = spark.createDataFrame(
            [
                (
                    valid_event.key,
                    valid_event.value,
                    "transactions.raw.v1",
                    0,
                    42,
                    valid_kafka_timestamp,
                ),
                (
                    invalid_event.key,
                    invalid_event.value,
                    "transactions.raw.v1",
                    1,
                    43,
                    invalid_kafka_timestamp,
                ),
            ],
            schema=kafka_schema,
        )

        parsed = parse_kafka_transaction_events(kafka_records)
        projected = project_transaction_events(parsed)
        typed = type_transaction_events(projected)
        validated = validate_transaction_events(typed)

        bronze = project_streaming_bronze(parsed)
        silver = project_streaming_silver(select_valid_transaction_events(validated))
        quarantine = project_streaming_quarantine(select_quarantined_transaction_events(validated))

        assert bronze.columns == BRONZE_COLUMNS
        assert silver.columns == SILVER_COLUMNS
        assert quarantine.columns == QUARANTINE_COLUMNS

        bronze_rows = bronze.orderBy("kafka_offset").collect()
        assert len(bronze_rows) == 2
        assert bronze_rows[0]["kafka_key"] == "1001"
        assert bronze_rows[0]["raw_event"] == valid_event.value.decode("utf-8")
        assert bronze_rows[0]["kafka_topic"] == "transactions.raw.v1"
        assert bronze_rows[0]["kafka_partition"] == 0
        assert bronze_rows[0]["kafka_offset"] == 42
        assert bronze_rows[0]["kafka_timestamp"] == valid_kafka_timestamp
        assert bronze_rows[0]["ingested_at_utc"] == valid_kafka_timestamp

        silver_rows = silver.collect()
        assert len(silver_rows) == 1

        silver_row = silver_rows[0].asDict(recursive=True)
        assert silver_row["transaction_id"] == 1001
        assert silver_row["tx_datetime"] == datetime(2018, 4, 1, 8, 0)
        assert silver_row["customer_id"] == 101
        assert silver_row["terminal_id"] == 201
        assert silver_row["tx_amount"] == 25.5
        assert silver_row["tx_time_seconds"] == 28800
        assert silver_row["tx_time_days"] == 0
        assert silver_row["tx_fraud"] == 0
        assert silver_row["tx_fraud_scenario"] == 0
        assert silver_row["source_file"] == "2018-04-01.pkl"
        assert silver_row["source_file_date"] == date(2018, 4, 1)
        assert silver_row["source_row_number"] == 0
        assert silver_row["ingested_at_utc"] == valid_kafka_timestamp
        assert silver_row["schema_version"] == "1.0"
        assert silver_row["event_id"] == ("717d4707-cdeb-5bb1-968a-ddf53b5f49ce")
        assert silver_row["produced_at_utc"] == datetime(
            2026,
            10,
            5,
            13,
            0,
        )
        assert silver_row["kafka_key"] == "1001"
        assert silver_row["kafka_topic"] == "transactions.raw.v1"
        assert silver_row["kafka_partition"] == 0
        assert silver_row["kafka_offset"] == 42
        assert silver_row["kafka_timestamp"] == valid_kafka_timestamp

        quarantine_rows = quarantine.collect()
        assert len(quarantine_rows) == 1

        quarantine_row = quarantine_rows[0].asDict(recursive=True)
        assert quarantine_row["TRANSACTION_ID"] == "1002"
        assert quarantine_row["TX_AMOUNT"] == "-10.0"
        assert quarantine_row["source_file"] == "2018-04-01.pkl"
        assert quarantine_row["source_file_date"] == "2018-04-01"
        assert quarantine_row["source_row_number"] == 1
        assert quarantine_row["ingested_at_utc"] == invalid_kafka_timestamp
        assert quarantine_row["kafka_key"] == "1002"
        assert quarantine_row["kafka_partition"] == 1
        assert quarantine_row["kafka_offset"] == 43
        assert quarantine_row["rejection_reasons"] == ["negative_transaction_amount"]

        assert isinstance(
            silver.schema["tx_datetime"].dataType,
            T.TimestampType,
        )
        assert isinstance(
            silver.schema["source_file_date"].dataType,
            T.DateType,
        )
        assert isinstance(
            silver.schema["transaction_id"].dataType,
            T.LongType,
        )
        assert isinstance(
            quarantine.schema["rejection_reasons"].dataType,
            T.ArrayType,
        )

        print("Spark streaming output projection validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
