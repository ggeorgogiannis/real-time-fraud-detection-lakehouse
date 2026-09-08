import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from fraud_lakehouse.pipeline import run_batch_pipeline

LOGGER = logging.getLogger(__name__)


def _parse_utc_datetime(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected an ISO 8601 datetime, for example 2026-09-08T12:00:00+00:00."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("The ingestion datetime must include a timezone.")

    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-lakehouse",
        description="Run the local fraud-detection lakehouse pipeline.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Bronze, Silver and Gold batch pipeline.",
    )
    run_parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing daily transaction CSV files.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for generated lakehouse outputs.",
    )
    run_parser.add_argument(
        "--ingested-at-utc",
        type=_parse_utc_datetime,
        help="Optional timezone-aware ingestion timestamp.",
    )
    run_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        result = run_batch_pipeline(
            raw_dir=arguments.raw_dir,
            output_dir=arguments.output_dir,
            ingested_at_utc=arguments.ingested_at_utc,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("pipeline_failed error=%s", exc)
        return 1

    LOGGER.info(
        ("pipeline_completed bronze_files=%d silver_files=%d quarantine_files=%d gold_tables=%d"),
        len(result.bronze_files),
        len(result.silver_files),
        len(result.quarantine_files),
        len(result.gold_tables),
    )

    return 0
