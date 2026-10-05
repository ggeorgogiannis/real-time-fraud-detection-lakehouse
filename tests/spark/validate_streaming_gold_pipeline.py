import math
import tempfile
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

from fraud_lakehouse.streaming_gold_pipeline import (
    StreamingGoldPaths,
    build_streaming_gold_frames,
    read_streaming_silver,
    start_streaming_gold_queries,
    streaming_silver_schema,
)
from fraud_lakehouse.streaming_pipeline import stop_streaming_queries


def _silver_record(
    *,
    transaction_id: int,
    tx_datetime: datetime,
    customer_id: int,
    terminal_id: int,
    tx_amount: float,
    source_row_number: int,
) -> tuple[object, ...]:
    source_date = tx_datetime.date()
    tx_time_seconds = tx_datetime.hour * 3600 + tx_datetime.minute * 60 + tx_datetime.second

    metadata_timestamp = datetime(
        2026,
        10,
        5,
        13,
        0,
        source_row_number,
    )

    return (
        transaction_id,
        tx_datetime,
        customer_id,
        terminal_id,
        tx_amount,
        tx_time_seconds,
        0,
        0,
        0,
        f"{source_date.isoformat()}.pkl",
        source_date,
        source_row_number,
        metadata_timestamp,
        "1.0",
        f"event-{transaction_id}",
        metadata_timestamp,
        str(transaction_id),
        "transactions.raw.v1",
        source_row_number % 3,
        source_row_number,
        metadata_timestamp,
    )


def _write_silver_batch(
    spark: SparkSession,
    input_directory: Path,
    records: list[tuple[object, ...]],
) -> None:
    (
        spark.createDataFrame(
            records,
            schema=streaming_silver_schema(),
        )
        .coalesce(1)
        .write.mode("append")
        .parquet(str(input_directory))
    )


def _process_all(
    queries,
) -> None:
    for query in queries:
        query.processAllAvailable()


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


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("streaming-gold-pipeline-validation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.streaming.noDataMicroBatches.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            paths = StreamingGoldPaths(
                silver_input=temporary_path / "silver",
                output_root=temporary_path / "gold",
                checkpoint_root=temporary_path / "checkpoints",
            )

            try:
                StreamingGoldPaths(
                    silver_input=temporary_path / "silver",
                    output_root=temporary_path / "same",
                    checkpoint_root=temporary_path / "same",
                )
            except ValueError as error:
                assert str(error) == (
                    "output_root and checkpoint_root must be different directories"
                )
            else:
                raise AssertionError("Expected matching Gold roots to be rejected")

            try:
                read_streaming_silver(
                    spark,
                    silver_path=paths.silver_input,
                    max_files_per_trigger=0,
                )
            except ValueError as error:
                assert str(error) == ("max_files_per_trigger must be at least 1")
            else:
                raise AssertionError("Expected invalid maximum files per trigger to be rejected")

            silver_stream = read_streaming_silver(
                spark,
                silver_path=paths.silver_input,
                max_files_per_trigger=1,
            )
            outputs = build_streaming_gold_frames(
                silver_stream,
                window_duration="1 hour",
                watermark_delay="10 minutes",
            )

            first_queries = start_streaming_gold_queries(
                outputs,
                paths=paths,
                trigger_interval="1 second",
            )

            try:
                _write_silver_batch(
                    spark,
                    paths.silver_input,
                    [
                        _silver_record(
                            transaction_id=1001,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                8,
                                5,
                            ),
                            customer_id=101,
                            terminal_id=201,
                            tx_amount=25.0,
                            source_row_number=0,
                        ),
                        _silver_record(
                            transaction_id=1002,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                8,
                                20,
                            ),
                            customer_id=101,
                            terminal_id=202,
                            tx_amount=75.0,
                            source_row_number=1,
                        ),
                        _silver_record(
                            transaction_id=1003,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                8,
                                45,
                            ),
                            customer_id=102,
                            terminal_id=201,
                            tx_amount=100.0,
                            source_row_number=2,
                        ),
                    ],
                )
                _process_all(first_queries)

                _write_silver_batch(
                    spark,
                    paths.silver_input,
                    [
                        _silver_record(
                            transaction_id=1004,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                9,
                                11,
                            ),
                            customer_id=101,
                            terminal_id=201,
                            tx_amount=50.0,
                            source_row_number=3,
                        )
                    ],
                )
                _process_all(first_queries)

                _write_silver_batch(
                    spark,
                    paths.silver_input,
                    [
                        _silver_record(
                            transaction_id=1005,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                9,
                                12,
                            ),
                            customer_id=101,
                            terminal_id=201,
                            tx_amount=60.0,
                            source_row_number=4,
                        )
                    ],
                )
                _process_all(first_queries)
            finally:
                stop_streaming_queries(first_queries)

            first_customer = spark.read.parquet(str(paths.customer_output))
            first_terminal = spark.read.parquet(str(paths.terminal_output))

            assert first_customer.count() == 2
            assert first_terminal.count() == 2

            customer_rows = {
                (
                    row["window_start"],
                    row["customer_id"],
                ): row.asDict()
                for row in first_customer.collect()
            }

            customer_101 = customer_rows[
                (
                    datetime(2018, 4, 1, 8, 0),
                    101,
                )
            ]
            assert customer_101["window_end"] == datetime(
                2018,
                4,
                1,
                9,
                0,
            )
            assert customer_101["transaction_count"] == 2
            assert customer_101["approx_unique_terminals"] == 2
            _assert_close(
                customer_101["total_amount"],
                100.0,
            )
            _assert_close(
                customer_101["average_amount"],
                50.0,
            )
            _assert_close(
                customer_101["maximum_amount"],
                75.0,
            )

            customer_102 = customer_rows[
                (
                    datetime(2018, 4, 1, 8, 0),
                    102,
                )
            ]
            assert customer_102["transaction_count"] == 1
            _assert_close(
                customer_102["total_amount"],
                100.0,
            )

            terminal_rows = {
                (
                    row["window_start"],
                    row["terminal_id"],
                ): row.asDict()
                for row in first_terminal.collect()
            }

            terminal_201 = terminal_rows[
                (
                    datetime(2018, 4, 1, 8, 0),
                    201,
                )
            ]
            assert terminal_201["transaction_count"] == 2
            assert terminal_201["approx_unique_customers"] == 2
            _assert_close(
                terminal_201["total_amount"],
                125.0,
            )
            _assert_close(
                terminal_201["average_amount"],
                62.5,
            )
            _assert_close(
                terminal_201["maximum_amount"],
                100.0,
            )

            terminal_202 = terminal_rows[
                (
                    datetime(2018, 4, 1, 8, 0),
                    202,
                )
            ]
            assert terminal_202["transaction_count"] == 1
            assert terminal_202["approx_unique_customers"] == 1
            _assert_close(
                terminal_202["total_amount"],
                75.0,
            )

            assert (paths.customer_checkpoint / "metadata").is_file()
            assert (paths.terminal_checkpoint / "metadata").is_file()

            assert all("fraud" not in column.lower() for column in first_customer.columns)
            assert all("fraud" not in column.lower() for column in first_terminal.columns)

            restarted_queries = start_streaming_gold_queries(
                outputs,
                paths=paths,
                trigger_interval="1 second",
            )

            try:
                _write_silver_batch(
                    spark,
                    paths.silver_input,
                    [
                        _silver_record(
                            transaction_id=1006,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                10,
                                11,
                            ),
                            customer_id=103,
                            terminal_id=203,
                            tx_amount=40.0,
                            source_row_number=5,
                        )
                    ],
                )
                _process_all(restarted_queries)

                _write_silver_batch(
                    spark,
                    paths.silver_input,
                    [
                        _silver_record(
                            transaction_id=1007,
                            tx_datetime=datetime(
                                2018,
                                4,
                                1,
                                10,
                                12,
                            ),
                            customer_id=103,
                            terminal_id=203,
                            tx_amount=45.0,
                            source_row_number=6,
                        )
                    ],
                )
                _process_all(restarted_queries)
            finally:
                stop_streaming_queries(restarted_queries)

            final_customer = spark.read.parquet(str(paths.customer_output))
            final_terminal = spark.read.parquet(str(paths.terminal_output))

            assert final_customer.count() == 3
            assert final_terminal.count() == 3

            final_customer_rows = {
                (
                    row["window_start"],
                    row["customer_id"],
                ): row.asDict()
                for row in final_customer.collect()
            }

            customer_101_next_hour = final_customer_rows[
                (
                    datetime(2018, 4, 1, 9, 0),
                    101,
                )
            ]
            assert customer_101_next_hour["transaction_count"] == 2
            assert customer_101_next_hour["approx_unique_terminals"] == 1
            _assert_close(
                customer_101_next_hour["total_amount"],
                110.0,
            )
            _assert_close(
                customer_101_next_hour["average_amount"],
                55.0,
            )
            _assert_close(
                customer_101_next_hour["maximum_amount"],
                60.0,
            )

            final_terminal_rows = {
                (
                    row["window_start"],
                    row["terminal_id"],
                ): row.asDict()
                for row in final_terminal.collect()
            }

            terminal_201_next_hour = final_terminal_rows[
                (
                    datetime(2018, 4, 1, 9, 0),
                    201,
                )
            ]
            assert terminal_201_next_hour["transaction_count"] == 2
            assert terminal_201_next_hour["approx_unique_customers"] == 1
            _assert_close(
                terminal_201_next_hour["total_amount"],
                110.0,
            )

            assert (
                final_customer.select(
                    "window_start",
                    "customer_id",
                )
                .distinct()
                .count()
                == 3
            )
            assert (
                final_terminal.select(
                    "window_start",
                    "terminal_id",
                )
                .distinct()
                .count()
                == 3
            )

            assert paths.silver_input.is_dir()
            assert paths.customer_output.is_dir()
            assert paths.terminal_output.is_dir()

            print("Spark streaming Gold pipeline persistence validation passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
