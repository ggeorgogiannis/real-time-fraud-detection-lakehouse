from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fraud_lakehouse.bronze import ingest_bronze_directory
from fraud_lakehouse.gold import write_gold_tables
from fraud_lakehouse.silver import ingest_silver_directory


@dataclass(frozen=True)
class PipelineDirectories:
    raw: Path
    bronze: Path
    silver: Path
    quarantine: Path
    gold: Path

    @classmethod
    def from_roots(cls, raw_dir: Path, output_dir: Path) -> "PipelineDirectories":
        return cls(
            raw=raw_dir,
            bronze=output_dir / "bronze",
            silver=output_dir / "silver",
            quarantine=output_dir / "quarantine",
            gold=output_dir / "gold",
        )


@dataclass(frozen=True)
class BatchPipelineResult:
    bronze_files: tuple[Path, ...]
    silver_files: tuple[Path, ...]
    quarantine_files: tuple[Path, ...]
    gold_tables: Mapping[str, Path]


def run_batch_pipeline(
    raw_dir: Path,
    output_dir: Path,
    ingested_at_utc: datetime | None = None,
) -> BatchPipelineResult:
    """Run the complete Bronze, Silver and Gold batch pipeline."""
    timestamp = ingested_at_utc or datetime.now(UTC)

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("ingested_at_utc must be timezone-aware.")

    timestamp = timestamp.astimezone(UTC)
    directories = PipelineDirectories.from_roots(raw_dir, output_dir)

    bronze_files = tuple(
        ingest_bronze_directory(
            directories.raw,
            directories.bronze,
            timestamp,
        )
    )

    silver_outputs = ingest_silver_directory(
        directories.bronze,
        directories.silver,
        directories.quarantine,
    )

    silver_files = tuple(silver_file for silver_file, _ in silver_outputs)
    quarantine_files = tuple(quarantine_file for _, quarantine_file in silver_outputs)

    gold_tables = write_gold_tables(
        directories.silver,
        directories.gold,
    )

    return BatchPipelineResult(
        bronze_files=bronze_files,
        silver_files=silver_files,
        quarantine_files=quarantine_files,
        gold_tables=dict(gold_tables),
    )
