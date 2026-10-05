import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.streaming_events import build_transaction_event
from fraud_lakehouse.streaming_pipeline import (
    StreamingPipelinePaths,
    build_streaming_output_frames,
    start_streaming_output_queries,
    stop_streaming_queries,
)

KAFKA_SCHEMA = T.StructType(
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


def _build_event(
    *,
    transaction_id: int,
    transaction_amount: float,
    source_row_number: int,
    produced_at_utc: datetime,
):
    return build_transaction_event(
        transaction=_transaction(
            transaction_id=transaction_id,
            transaction_amount=transaction_amount,
        ),
        source_file="2018-04-01.pkl",
        source_file_date=date(2018, 4, 1),
        source_row_number=source_row_number,
        produced_at_utc=produced_at_utc,
    )


def _write_kafka_batch(
    spark: SparkSession,
    input_directory: Path,
    records: list[tuple[object, ...]],
) -> None:
    (
        spark.createDataFrame(
            records,
            schema=KAFKA_SCHEMA,
        )
        .write.mode("append")
        .parquet(str(input_directory))
    )


def _process_all(
    queries,
) -> None:
    for query in queries:
        query.processAllAvailable()


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-pipeline-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_directory = temporary_path / "input"
            input_directory.mkdir()

            paths = StreamingPipelinePaths(
                output_root=temporary_path / "outputs",
                checkpoint_root=temporary_path / "checkpoints",
            )

            try:
                StreamingPipelinePaths(
                    output_root=temporary_path / "same",
                    checkpoint_root=temporary_path / "same",
                )
            except ValueError as error:
                assert str(error) == (
                    "output_root and checkpoint_root must be different directories"
                )
            else:
                raise AssertionError("Expected matching pipeline roots to be rejected")

            source = (
                spark.readStream.schema(KAFKA_SCHEMA)
                .option(
                    "maxFilesPerTrigger",
                    1,
                )
                .parquet(str(input_directory))
            )
            outputs = build_streaming_output_frames(
                source,
                watermark_delay="1 day",
            )

            valid_event = _build_event(
                transaction_id=1001,
                transaction_amount=25.5,
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
            invalid_event = _build_event(
                transaction_id=1002,
                transaction_amount=-10.0,
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

            first_queries = start_streaming_output_queries(
                outputs,
                paths=paths,
                trigger_interval="1 second",
            )

            try:
                _write_kafka_batch(
                    spark,
                    input_directory,
                    [
                        (
                            valid_event.key,
                            valid_event.value,
                            "transactions.raw.v1",
                            0,
                            42,
                            datetime(2026, 10, 5, 13, 0),
                        ),
                        (
                            valid_event.key,
                            valid_event.value,
                            "transactions.raw.v1",
                            1,
                            43,
                            datetime(2026, 10, 5, 13, 0, 1),
                        ),
                        (
                            invalid_event.key,
                            invalid_event.value,
                            "transactions.raw.v1",
                            2,
                            44,
                            datetime(2026, 10, 5, 13, 0, 2),
                        ),
                    ],
                )
                _process_all(first_queries)
            finally:
                stop_streaming_queries(first_queries)

            first_bronze = spark.read.parquet(str(paths.bronze_output))
            first_silver = spark.read.parquet(str(paths.silver_output))
            first_quarantine = spark.read.parquet(str(paths.quarantine_output))

            assert first_bronze.count() == 3
            assert first_silver.count() == 1
            assert first_quarantine.count() == 1

            assert first_silver.select("transaction_id").collect()[0][0] == 1001
            assert first_quarantine.select("rejection_reasons").collect()[0][0] == [
                "negative_transaction_amount"
            ]

            assert (paths.bronze_checkpoint / "metadata").is_file()
            assert (paths.silver_checkpoint / "metadata").is_file()
            assert (paths.quarantine_checkpoint / "metadata").is_file()

            new_event = _build_event(
                transaction_id=1003,
                transaction_amount=75.0,
                source_row_number=2,
                produced_at_utc=datetime(
                    2026,
                    10,
                    5,
                    13,
                    0,
                    2,
                    tzinfo=UTC,
                ),
            )

            restarted_queries = start_streaming_output_queries(
                outputs,
                paths=paths,
                trigger_interval="1 second",
            )

            try:
                _write_kafka_batch(
                    spark,
                    input_directory,
                    [
                        (
                            valid_event.key,
                            valid_event.value,
                            "transactions.raw.v1",
                            0,
                            45,
                            datetime(2026, 10, 5, 13, 0, 3),
                        ),
                        (
                            new_event.key,
                            new_event.value,
                            "transactions.raw.v1",
                            1,
                            46,
                            datetime(2026, 10, 5, 13, 0, 4),
                        ),
                    ],
                )
                _process_all(restarted_queries)
            finally:
                stop_streaming_queries(restarted_queries)

            final_bronze = spark.read.parquet(str(paths.bronze_output))
            final_silver = spark.read.parquet(str(paths.silver_output))
            final_quarantine = spark.read.parquet(str(paths.quarantine_output))

            assert final_bronze.count() == 5
            assert final_quarantine.count() == 1

            silver_transaction_ids = [
                row["transaction_id"]
                for row in final_silver.select("transaction_id").orderBy("transaction_id").collect()
            ]
            assert silver_transaction_ids == [
                1001,
                1003,
            ]

            assert final_bronze.select("kafka_offset").distinct().count() == 5
            assert final_silver.select("transaction_id").distinct().count() == 2

            print("Spark streaming pipeline persistence validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
