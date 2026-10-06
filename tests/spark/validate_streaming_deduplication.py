import json
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import types as T

from fraud_lakehouse.streaming_transformations import (
    deduplicate_transaction_events,
)

STREAM_SCHEMA = T.StructType(
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
            "description",
            T.StringType(),
            nullable=False,
        ),
    ]
)


def _write_batch(
    input_directory: Path,
    batch_name: str,
    records: list[dict[str, object]],
) -> None:
    """Write one newline-delimited JSON file as a streaming micro-batch."""
    batch_path = input_directory / f"{batch_name}.json"
    batch_content = "\n".join(
        json.dumps(
            record,
            separators=(",", ":"),
            sort_keys=True,
        )
        for record in records
    )
    batch_path.write_text(
        f"{batch_content}\n",
        encoding="utf-8",
    )


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-deduplication-validation")
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
            checkpoint_directory = temporary_path / "checkpoint"
            input_directory.mkdir()

            source = (
                spark.readStream.schema(STREAM_SCHEMA)
                .option(
                    "maxFilesPerTrigger",
                    1,
                )
                .json(str(input_directory))
            )

            try:
                deduplicate_transaction_events(
                    source,
                    watermark_delay=" ",
                )
            except ValueError as error:
                assert str(error) == "watermark_delay must not be empty"
            else:
                raise AssertionError("Expected an empty watermark delay to be rejected")

            deduplicated = deduplicate_transaction_events(
                source,
                watermark_delay="1 day",
            )

            assert deduplicated.isStreaming

            query = (
                deduplicated.writeStream.format("memory")
                .queryName("deduplicated_transactions")
                .outputMode("append")
                .option(
                    "checkpointLocation",
                    str(checkpoint_directory),
                )
                .start()
            )

            try:
                _write_batch(
                    input_directory,
                    "batch-001",
                    [
                        {
                            "transaction_id": 1001,
                            "tx_datetime": "2018-04-01T08:00:00Z",
                            "description": "first transaction",
                        },
                        {
                            "transaction_id": 1002,
                            "tx_datetime": "2018-04-01T09:00:00Z",
                            "description": "second transaction",
                        },
                    ],
                )
                query.processAllAvailable()

                _write_batch(
                    input_directory,
                    "batch-002",
                    [
                        {
                            "transaction_id": 1001,
                            "tx_datetime": "2018-04-01T08:05:00Z",
                            "description": "duplicate transaction",
                        },
                        {
                            "transaction_id": 1003,
                            "tx_datetime": "2018-04-01T10:00:00Z",
                            "description": "third transaction",
                        },
                    ],
                )
                query.processAllAvailable()

                current_rows = (
                    spark.table("deduplicated_transactions")
                    .select(
                        "transaction_id",
                        "description",
                    )
                    .orderBy("transaction_id")
                    .collect()
                )

                assert [
                    (
                        row["transaction_id"],
                        row["description"],
                    )
                    for row in current_rows
                ] == [
                    (
                        1001,
                        "first transaction",
                    ),
                    (
                        1002,
                        "second transaction",
                    ),
                    (
                        1003,
                        "third transaction",
                    ),
                ]

                _write_batch(
                    input_directory,
                    "batch-003",
                    [
                        {
                            "transaction_id": 2000,
                            "tx_datetime": "2018-04-04T00:00:00Z",
                            "description": "watermark advancement transaction",
                        }
                    ],
                )
                query.processAllAvailable()

                _write_batch(
                    input_directory,
                    "batch-004",
                    [
                        {
                            "transaction_id": 9999,
                            "tx_datetime": "2018-04-01T12:00:00Z",
                            "description": "late transaction",
                        }
                    ],
                )
                query.processAllAvailable()

                final_rows = (
                    spark.table("deduplicated_transactions")
                    .select(
                        "transaction_id",
                        "description",
                    )
                    .orderBy("transaction_id")
                    .collect()
                )

                assert [
                    (
                        row["transaction_id"],
                        row["description"],
                    )
                    for row in final_rows
                ] == [
                    (
                        1001,
                        "first transaction",
                    ),
                    (
                        1002,
                        "second transaction",
                    ),
                    (
                        1003,
                        "third transaction",
                    ),
                    (
                        2000,
                        "watermark advancement transaction",
                    ),
                ]

                assert all(row["transaction_id"] != 9999 for row in final_rows)

                print("Spark streaming deduplication validation passed")
            finally:
                query.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
