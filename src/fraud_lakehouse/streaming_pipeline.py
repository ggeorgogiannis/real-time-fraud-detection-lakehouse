from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.streaming import StreamingQuery

from fraud_lakehouse.streaming_transformations import (
    DEFAULT_TRANSACTION_WATERMARK_DELAY,
    deduplicate_transaction_events,
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

DEFAULT_STARTING_OFFSETS = "earliest"
DEFAULT_TRIGGER_INTERVAL = "10 seconds"


@dataclass(frozen=True, slots=True)
class StreamingOutputFrames:
    """Bronze, Silver and quarantine streaming DataFrames."""

    bronze: DataFrame
    silver: DataFrame
    quarantine: DataFrame


@dataclass(frozen=True, slots=True)
class StreamingPipelinePaths:
    """Persistent output and checkpoint locations."""

    output_root: Path
    checkpoint_root: Path

    def __post_init__(self) -> None:
        if self.output_root.resolve() == self.checkpoint_root.resolve():
            raise ValueError("output_root and checkpoint_root must be different directories")

    @property
    def bronze_output(self) -> Path:
        return self.output_root / "bronze"

    @property
    def silver_output(self) -> Path:
        return self.output_root / "silver"

    @property
    def quarantine_output(self) -> Path:
        return self.output_root / "quarantine"

    @property
    def bronze_checkpoint(self) -> Path:
        return self.checkpoint_root / "bronze"

    @property
    def silver_checkpoint(self) -> Path:
        return self.checkpoint_root / "silver"

    @property
    def quarantine_checkpoint(self) -> Path:
        return self.checkpoint_root / "quarantine"


def read_kafka_transaction_stream(
    spark: SparkSession,
    *,
    bootstrap_servers: str,
    topic: str,
    starting_offsets: str = DEFAULT_STARTING_OFFSETS,
) -> DataFrame:
    """Create the raw Kafka transaction stream."""
    if not bootstrap_servers.strip():
        raise ValueError("bootstrap_servers must not be empty")

    if not topic.strip():
        raise ValueError("topic must not be empty")

    if not starting_offsets.strip():
        raise ValueError("starting_offsets must not be empty")

    return (
        spark.readStream.format("kafka")
        .option(
            "kafka.bootstrap.servers",
            bootstrap_servers,
        )
        .option(
            "subscribe",
            topic,
        )
        .option(
            "startingOffsets",
            starting_offsets,
        )
        .option(
            "failOnDataLoss",
            "true",
        )
        .load()
    )


def build_streaming_output_frames(
    kafka_records: DataFrame,
    *,
    watermark_delay: str = DEFAULT_TRANSACTION_WATERMARK_DELAY,
) -> StreamingOutputFrames:
    """Build Bronze, Silver and quarantine transformations."""
    parsed = parse_kafka_transaction_events(kafka_records)
    projected = project_transaction_events(parsed)
    typed = type_transaction_events(projected)
    validated = validate_transaction_events(typed)

    valid_events = select_valid_transaction_events(validated)
    quarantined_events = select_quarantined_transaction_events(validated)
    deduplicated_events = deduplicate_transaction_events(
        valid_events,
        watermark_delay=watermark_delay,
    )

    return StreamingOutputFrames(
        bronze=project_streaming_bronze(parsed),
        silver=project_streaming_silver(deduplicated_events),
        quarantine=project_streaming_quarantine(quarantined_events),
    )


def _start_parquet_query(
    frame: DataFrame,
    *,
    query_name: str,
    output_path: Path,
    checkpoint_path: Path,
    trigger_interval: str,
) -> StreamingQuery:
    if not trigger_interval.strip():
        raise ValueError("trigger_interval must not be empty")

    return (
        frame.writeStream.queryName(query_name)
        .format("parquet")
        .outputMode("append")
        .option(
            "checkpointLocation",
            str(checkpoint_path),
        )
        .trigger(processingTime=trigger_interval)
        .start(str(output_path))
    )


def start_streaming_output_queries(
    outputs: StreamingOutputFrames,
    *,
    paths: StreamingPipelinePaths,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
) -> tuple[StreamingQuery, StreamingQuery, StreamingQuery]:
    """Start independent restart-safe Parquet queries for each output."""
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
            outputs.bronze,
            "fraud_streaming_bronze",
            paths.bronze_output,
            paths.bronze_checkpoint,
        ),
        (
            outputs.silver,
            "fraud_streaming_silver",
            paths.silver_output,
            paths.silver_checkpoint,
        ),
        (
            outputs.quarantine,
            "fraud_streaming_quarantine",
            paths.quarantine_output,
            paths.quarantine_checkpoint,
        ),
    )

    queries: list[StreamingQuery] = []

    try:
        for frame, query_name, output_path, checkpoint_path in query_specs:
            query = _start_parquet_query(
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
        queries[2],
    )


def start_streaming_pipeline(
    spark: SparkSession,
    *,
    bootstrap_servers: str,
    topic: str,
    paths: StreamingPipelinePaths,
    starting_offsets: str = DEFAULT_STARTING_OFFSETS,
    watermark_delay: str = DEFAULT_TRANSACTION_WATERMARK_DELAY,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
) -> tuple[StreamingQuery, StreamingQuery, StreamingQuery]:
    """Build and start the Kafka-to-Parquet streaming pipeline."""
    kafka_records = read_kafka_transaction_stream(
        spark,
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        starting_offsets=starting_offsets,
    )
    outputs = build_streaming_output_frames(
        kafka_records,
        watermark_delay=watermark_delay,
    )

    return start_streaming_output_queries(
        outputs,
        paths=paths,
        trigger_interval=trigger_interval,
    )


def stop_streaming_queries(
    queries: tuple[StreamingQuery, ...],
) -> None:
    """Stop every active query in the supplied collection."""
    for query in queries:
        if query.isActive:
            query.stop()
