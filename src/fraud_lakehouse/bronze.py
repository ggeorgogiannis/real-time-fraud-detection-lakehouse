import re
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

_SOURCE_FILENAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.pkl$")


SOURCE_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


def _parse_source_file_date(source_file: Path) -> date:
    if _SOURCE_FILENAME_PATTERN.fullmatch(source_file.name) is None:
        raise ValueError(f"Invalid source filename: {source_file.name}")

    try:
        return date.fromisoformat(source_file.stem)
    except ValueError as error:
        raise ValueError(f"Invalid source filename: {source_file.name}") from error


def discover_source_files(raw_dir: Path) -> list[Path]:
    """Return valid daily Pickle files in chronological order."""
    source_files = (path for path in raw_dir.iterdir() if path.is_file() and path.suffix == ".pkl")

    return sorted(source_files, key=_parse_source_file_date)


def load_trusted_source_file(source_file: Path) -> pd.DataFrame:
    """Load a trusted source Pickle and validate its file-level schema."""
    data = pd.read_pickle(source_file)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Source file must contain a Pandas DataFrame: {source_file.name}")

    required_columns = set(SOURCE_COLUMNS)
    actual_columns = set(data.columns)

    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        details = ", ".join(missing_columns)
        raise ValueError(f"Source file is missing required columns: {details}")

    unexpected_columns = sorted(str(column) for column in actual_columns - required_columns)
    if unexpected_columns:
        details = ", ".join(unexpected_columns)
        raise ValueError(f"Source file contains unexpected columns: {details}")

    return data


def _validate_ingested_at_utc(ingested_at_utc: datetime) -> None:
    if ingested_at_utc.tzinfo is None or ingested_at_utc.utcoffset() != timedelta(0):
        raise ValueError("ingested_at_utc must be timezone-aware UTC")


def build_bronze_records(
    source_file: Path,
    ingested_at_utc: datetime,
) -> pd.DataFrame:
    """Load source records and add Bronze ingestion metadata."""
    _validate_ingested_at_utc(ingested_at_utc)
    source_file_date = _parse_source_file_date(source_file)
    source_data = load_trusted_source_file(source_file)
    bronze_data = source_data.copy()

    bronze_data["source_file"] = source_file.name
    bronze_data["source_file_date"] = source_file_date
    bronze_data["source_row_number"] = pd.Series(
        range(len(bronze_data)),
        index=bronze_data.index,
        dtype="int64",
    )
    bronze_data["ingested_at_utc"] = ingested_at_utc

    return bronze_data


def ingest_bronze_file(
    source_file: Path,
    bronze_dir: Path,
    ingested_at_utc: datetime,
) -> Path:
    """Persist one trusted source file to an idempotent Bronze Parquet file."""
    _validate_ingested_at_utc(ingested_at_utc)
    output_path = bronze_dir / f"{source_file.stem}.parquet"

    if output_path.exists():
        return output_path

    bronze_data = build_bronze_records(
        source_file=source_file,
        ingested_at_utc=ingested_at_utc,
    )
    bronze_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=bronze_dir) as temporary_directory:
        temporary_path = Path(temporary_directory) / output_path.name
        bronze_data.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
        )
        temporary_path.replace(output_path)

    return output_path


def ingest_bronze_directory(
    raw_dir: Path,
    bronze_dir: Path,
    ingested_at_utc: datetime,
) -> list[Path]:
    """Discover and ingest a deterministic batch of trusted source files."""
    _validate_ingested_at_utc(ingested_at_utc)

    return [
        ingest_bronze_file(
            source_file=source_file,
            bronze_dir=bronze_dir,
            ingested_at_utc=ingested_at_utc,
        )
        for source_file in discover_source_files(raw_dir)
    ]
