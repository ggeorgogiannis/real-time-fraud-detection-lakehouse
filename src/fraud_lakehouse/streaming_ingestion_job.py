import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery

from fraud_lakehouse.streaming_events import TRANSACTION_TOPIC
from fraud_lakehouse.streaming_pipeline import (
    DEFAULT_STARTING_OFFSETS,
    DEFAULT_TRIGGER_INTERVAL,
    StreamingPipelinePaths,
    start_streaming_pipeline,
    stop_streaming_queries,
)
from fraud_lakehouse.streaming_transformations import (
    DEFAULT_TRANSACTION_WATERMARK_DELAY,
)

LOGGER = logging.getLogger(__name__)
LOG_LEVELS = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the streaming ingestion job argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Consume transaction events from Kafka and persist streaming "
            "Bronze, Silver and quarantine outputs."
        )
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="kafka:19092",
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--topic",
        default=TRANSACTION_TOPIC,
        help="Kafka topic containing raw transaction events.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/opt/spark/data/streaming/lakehouse"),
        help="Root directory for streaming lakehouse outputs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/opt/spark/data/streaming/checkpoints/ingestion"),
        help="Root directory for ingestion query checkpoints.",
    )
    parser.add_argument(
        "--starting-offsets",
        default=DEFAULT_STARTING_OFFSETS,
        help="Kafka startingOffsets value used for a new checkpoint.",
    )
    parser.add_argument(
        "--watermark-delay",
        default=DEFAULT_TRANSACTION_WATERMARK_DELAY,
        help="Maximum event-time delay retained for deduplication.",
    )
    parser.add_argument(
        "--trigger-interval",
        default=DEFAULT_TRIGGER_INTERVAL,
        help="Structured Streaming processing-time trigger interval.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
    )
    return parser


def create_spark_session() -> SparkSession:
    """Create the Spark session used by the ingestion job."""
    return (
        SparkSession.builder.appName("fraud-lakehouse-streaming-ingestion")
        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )
        .getOrCreate()
    )


def run_ingestion_job(
    args: argparse.Namespace,
) -> None:
    """Run the Kafka-to-streaming-lakehouse job until termination."""
    spark = create_spark_session()
    spark.sparkContext.setLogLevel(args.log_level)

    queries: tuple[StreamingQuery, ...] = ()

    try:
        paths = StreamingPipelinePaths(
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
        )
        queries = start_streaming_pipeline(
            spark,
            bootstrap_servers=args.bootstrap_servers,
            topic=args.topic,
            paths=paths,
            starting_offsets=args.starting_offsets,
            watermark_delay=args.watermark_delay,
            trigger_interval=args.trigger_interval,
        )

        LOGGER.info(
            ("streaming_ingestion_started topic=%s output_root=%s checkpoint_root=%s"),
            args.topic,
            args.output_root,
            args.checkpoint_root,
        )
        spark.streams.awaitAnyTermination()
    finally:
        stop_streaming_queries(queries)
        spark.stop()


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the streaming ingestion command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_ingestion_job(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
