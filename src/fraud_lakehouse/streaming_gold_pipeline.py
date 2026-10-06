from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T
from pyspark.sql.streaming import StreamingQuery

from fraud_lakehouse.streaming_gold import (
    DEFAULT_GOLD_WATERMARK_DELAY,
    DEFAULT_GOLD_WINDOW_DURATION,
    build_customer_risk_windows,
    build_terminal_risk_windows,
)
from fraud_lakehouse.streaming_pipeline import (
    DEFAULT_TRIGGER_INTERVAL,
    start_parquet_query,
)


@dataclass(frozen=True, slots=True)
class StreamingGoldFrames:
    """Customer and terminal streaming risk aggregates."""

    customer_risk: DataFrame
    terminal_risk: DataFrame


@dataclass(frozen=True, slots=True)
class StreamingGoldPaths:
    """Silver input and persistent Gold output locations."""

    silver_input: Path
    output_root: Path
    checkpoint_root: Path

    def __post_init__(self) -> None:
        if self.output_root.resolve() == self.checkpoint_root.resolve():
            raise ValueError("output_root and checkpoint_root must be different directories")

    @property
    def customer_output(self) -> Path:
        return self.output_root / "customer_risk_windows"

    @property
    def terminal_output(self) -> Path:
        return self.output_root / "terminal_risk_windows"

    @property
    def customer_checkpoint(self) -> Path:
        return self.checkpoint_root / "customer_risk_windows"

    @property
    def terminal_checkpoint(self) -> Path:
        return self.checkpoint_root / "terminal_risk_windows"


def streaming_silver_schema() -> T.StructType:
    """Return the persisted streaming Silver schema."""
    return T.StructType(
        [
            T.StructField(
                "transaction_id",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "tx_datetime",
                T.TimestampType(),
                nullable=True,
            ),
            T.StructField(
                "customer_id",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "terminal_id",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "tx_amount",
                T.DoubleType(),
                nullable=True,
            ),
            T.StructField(
                "tx_time_seconds",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "tx_time_days",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "tx_fraud",
                T.ByteType(),
                nullable=True,
            ),
            T.StructField(
                "tx_fraud_scenario",
                T.ByteType(),
                nullable=True,
            ),
            T.StructField(
                "source_file",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "source_file_date",
                T.DateType(),
                nullable=True,
            ),
            T.StructField(
                "source_row_number",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "ingested_at_utc",
                T.TimestampType(),
                nullable=True,
            ),
            T.StructField(
                "schema_version",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "event_id",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "produced_at_utc",
                T.TimestampType(),
                nullable=True,
            ),
            T.StructField(
                "kafka_key",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "kafka_topic",
                T.StringType(),
                nullable=True,
            ),
            T.StructField(
                "kafka_partition",
                T.IntegerType(),
                nullable=True,
            ),
            T.StructField(
                "kafka_offset",
                T.LongType(),
                nullable=True,
            ),
            T.StructField(
                "kafka_timestamp",
                T.TimestampType(),
                nullable=True,
            ),
        ]
    )


def read_streaming_silver(
    spark: SparkSession,
    *,
    silver_path: Path,
    max_files_per_trigger: int | None = None,
) -> DataFrame:
    """Read persisted Silver Parquet files as a streaming source."""
    if max_files_per_trigger is not None and max_files_per_trigger < 1:
        raise ValueError("max_files_per_trigger must be at least 1")

    silver_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    reader = spark.readStream.schema(
        streaming_silver_schema(),
    ).format("parquet")

    if max_files_per_trigger is not None:
        reader = reader.option(
            "maxFilesPerTrigger",
            max_files_per_trigger,
        )

    return reader.load(str(silver_path))


def build_streaming_gold_frames(
    silver_events: DataFrame,
    *,
    window_duration: str = DEFAULT_GOLD_WINDOW_DURATION,
    watermark_delay: str = DEFAULT_GOLD_WATERMARK_DELAY,
) -> StreamingGoldFrames:
    """Build customer and terminal event-time risk aggregates."""
    return StreamingGoldFrames(
        customer_risk=build_customer_risk_windows(
            silver_events,
            window_duration=window_duration,
            watermark_delay=watermark_delay,
        ),
        terminal_risk=build_terminal_risk_windows(
            silver_events,
            window_duration=window_duration,
            watermark_delay=watermark_delay,
        ),
    )


def start_streaming_gold_queries(
    outputs: StreamingGoldFrames,
    *,
    paths: StreamingGoldPaths,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
) -> tuple[StreamingQuery, StreamingQuery]:
    """Start restart-safe customer and terminal Gold queries."""
    paths.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_specs = (
        (
            outputs.customer_risk,
            "fraud_streaming_customer_risk",
            paths.customer_output,
            paths.customer_checkpoint,
        ),
        (
            outputs.terminal_risk,
            "fraud_streaming_terminal_risk",
            paths.terminal_output,
            paths.terminal_checkpoint,
        ),
    )

    queries: list[StreamingQuery] = []

    try:
        for frame, query_name, output_path, checkpoint_path in query_specs:
            query = start_parquet_query(
                frame,
                query_name=query_name,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                trigger_interval=trigger_interval,
            )
            queries.append(query)
    except Exception:
        for query in queries:
            query.stop()
        raise

    return (
        queries[0],
        queries[1],
    )


def start_streaming_gold_pipeline(
    spark: SparkSession,
    *,
    paths: StreamingGoldPaths,
    window_duration: str = DEFAULT_GOLD_WINDOW_DURATION,
    watermark_delay: str = DEFAULT_GOLD_WATERMARK_DELAY,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
    max_files_per_trigger: int | None = None,
) -> tuple[StreamingQuery, StreamingQuery]:
    """Start the persisted-Silver-to-Gold streaming pipeline."""
    silver_events = read_streaming_silver(
        spark,
        silver_path=paths.silver_input,
        max_files_per_trigger=max_files_per_trigger,
    )
    outputs = build_streaming_gold_frames(
        silver_events,
        window_duration=window_duration,
        watermark_delay=watermark_delay,
    )

    return start_streaming_gold_queries(
        outputs,
        paths=paths,
        trigger_interval=trigger_interval,
    )
