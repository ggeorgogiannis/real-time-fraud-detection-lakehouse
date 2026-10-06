import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery

from fraud_lakehouse.streaming_gold import (
    DEFAULT_GOLD_WATERMARK_DELAY,
    DEFAULT_GOLD_WINDOW_DURATION,
)
from fraud_lakehouse.streaming_gold_pipeline import (
    StreamingGoldPaths,
    start_streaming_gold_pipeline,
)
from fraud_lakehouse.streaming_pipeline import (
    DEFAULT_TRIGGER_INTERVAL,
    stop_streaming_queries,
)

LOGGER = logging.getLogger(__name__)
LOG_LEVELS = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the streaming Gold job argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Read persisted streaming Silver transactions and create "
            "customer and terminal risk windows."
        )
    )
    parser.add_argument(
        "--silver-input",
        type=Path,
        default=Path("/opt/spark/data/streaming/lakehouse/silver"),
        help="Directory containing streaming Silver Parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/opt/spark/data/streaming/lakehouse/gold"),
        help="Root directory for streaming Gold outputs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/opt/spark/data/streaming/checkpoints/gold"),
        help="Root directory for Gold query checkpoints.",
    )
    parser.add_argument(
        "--window-duration",
        default=DEFAULT_GOLD_WINDOW_DURATION,
        help="Duration of each event-time risk window.",
    )
    parser.add_argument(
        "--watermark-delay",
        default=DEFAULT_GOLD_WATERMARK_DELAY,
        help="Maximum accepted event-time delay for Gold aggregation.",
    )
    parser.add_argument(
        "--trigger-interval",
        default=DEFAULT_TRIGGER_INTERVAL,
        help="Structured Streaming processing-time trigger interval.",
    )
    parser.add_argument(
        "--max-files-per-trigger",
        type=int,
        default=None,
        help="Optional maximum Silver files processed per trigger.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
    )
    return parser


def create_spark_session() -> SparkSession:
    """Create the Spark session used by the Gold job."""
    return (
        SparkSession.builder.appName("fraud-lakehouse-streaming-gold")
        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )
        .getOrCreate()
    )


def run_gold_job(
    args: argparse.Namespace,
) -> None:
    """Run the persisted-Silver-to-Gold job until termination."""
    spark = create_spark_session()
    spark.sparkContext.setLogLevel(args.log_level)

    queries: tuple[StreamingQuery, ...] = ()

    try:
        paths = StreamingGoldPaths(
            silver_input=args.silver_input,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
        )
        queries = start_streaming_gold_pipeline(
            spark,
            paths=paths,
            window_duration=args.window_duration,
            watermark_delay=args.watermark_delay,
            trigger_interval=args.trigger_interval,
            max_files_per_trigger=args.max_files_per_trigger,
        )

        LOGGER.info(
            ("streaming_gold_started silver_input=%s output_root=%s checkpoint_root=%s"),
            args.silver_input,
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
    """Run the streaming Gold command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_gold_job(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
