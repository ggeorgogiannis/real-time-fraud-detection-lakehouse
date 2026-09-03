import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fraud_lakehouse.bronze import SOURCE_COLUMNS

SOURCE_TO_SILVER_COLUMNS = {
    "TRANSACTION_ID": "transaction_id",
    "TX_DATETIME": "tx_datetime",
    "CUSTOMER_ID": "customer_id",
    "TERMINAL_ID": "terminal_id",
    "TX_AMOUNT": "tx_amount",
    "TX_TIME_SECONDS": "tx_time_seconds",
    "TX_TIME_DAYS": "tx_time_days",
    "TX_FRAUD": "tx_fraud",
    "TX_FRAUD_SCENARIO": "tx_fraud_scenario",
}

BRONZE_METADATA_COLUMNS = (
    "source_file",
    "source_file_date",
    "source_row_number",
    "ingested_at_utc",
)

_INTEGER_SOURCE_COLUMNS = (
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


@dataclass(frozen=True, slots=True)
class SilverValidationResult:
    """Valid Silver records and rejected Bronze records."""

    silver: pd.DataFrame
    quarantine: pd.DataFrame


def _add_reason(
    reasons: list[list[str]],
    invalid_mask: pd.Series,
    reason: str,
) -> None:
    for position, is_invalid in enumerate(invalid_mask.fillna(False).tolist()):
        if is_invalid and reason not in reasons[position]:
            reasons[position].append(reason)


def _validate_bronze_schema(bronze_data: pd.DataFrame) -> None:
    required_columns = set(SOURCE_COLUMNS) | set(BRONZE_METADATA_COLUMNS)
    missing_columns = sorted(required_columns - set(bronze_data.columns))

    if missing_columns:
        details = ", ".join(missing_columns)
        raise ValueError(f"Bronze data is missing required columns: {details}")


def validate_bronze_records(
    bronze_data: pd.DataFrame,
    known_transaction_ids: Collection[int] = (),
) -> SilverValidationResult:
    """Convert valid Bronze records and route invalid records to quarantine."""
    _validate_bronze_schema(bronze_data)

    bronze = bronze_data.reset_index(drop=True).copy()
    reasons: list[list[str]] = [[] for _ in range(len(bronze))]

    metadata_missing = bronze[list(BRONZE_METADATA_COLUMNS)].isna().any(axis=1)
    if metadata_missing.any():
        raise ValueError("Bronze ingestion metadata must not contain missing values")

    source_dates = pd.to_datetime(
        bronze["source_file_date"],
        errors="coerce",
    )
    source_row_numbers = pd.to_numeric(
        bronze["source_row_number"],
        errors="coerce",
    )
    ingested_at = pd.to_datetime(
        bronze["ingested_at_utc"],
        errors="coerce",
        utc=True,
    )

    invalid_metadata = (
        source_dates.isna()
        | source_row_numbers.isna()
        | source_row_numbers.isin([float("inf"), float("-inf")])
        | source_row_numbers.mod(1).ne(0)
        | source_row_numbers.lt(0)
        | ingested_at.isna()
    )
    if invalid_metadata.any():
        raise ValueError("Bronze ingestion metadata contains invalid values")

    missing_required_value = bronze[list(SOURCE_COLUMNS)].isna().any(axis=1)
    _add_reason(reasons, missing_required_value, "missing_required_value")

    converted_integers: dict[str, pd.Series] = {}
    valid_integer_types: dict[str, pd.Series] = {}
    invalid_data_type = pd.Series(False, index=bronze.index, dtype="bool")

    for column in _INTEGER_SOURCE_COLUMNS:
        converted = pd.to_numeric(bronze[column], errors="coerce")
        is_finite = ~converted.isin([float("inf"), float("-inf")])
        is_whole = converted.notna() & converted.mod(1).eq(0)
        is_valid = bronze[column].notna() & is_finite & is_whole

        converted_integers[column] = converted
        valid_integer_types[column] = is_valid
        invalid_data_type |= bronze[column].notna() & ~is_valid

    transaction_amount = pd.to_numeric(bronze["TX_AMOUNT"], errors="coerce")
    valid_amount_type = (
        bronze["TX_AMOUNT"].notna()
        & transaction_amount.notna()
        & ~transaction_amount.isin([float("inf"), float("-inf")])
    )
    invalid_data_type |= bronze["TX_AMOUNT"].notna() & ~valid_amount_type

    transaction_datetime = pd.to_datetime(
        bronze["TX_DATETIME"],
        errors="coerce",
        utc=True,
    )
    valid_datetime_type = bronze["TX_DATETIME"].notna() & transaction_datetime.notna()
    invalid_data_type |= bronze["TX_DATETIME"].notna() & ~valid_datetime_type

    negative_identifier = pd.Series(False, index=bronze.index, dtype="bool")
    for column in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"):
        negative_identifier |= valid_integer_types[column] & converted_integers[column].lt(0)

    invalid_data_type |= negative_identifier
    _add_reason(reasons, invalid_data_type, "invalid_data_type")

    negative_amount = valid_amount_type & transaction_amount.lt(0)
    _add_reason(reasons, negative_amount, "negative_transaction_amount")

    fraud_label = converted_integers["TX_FRAUD"]
    valid_fraud_label = valid_integer_types["TX_FRAUD"] & fraud_label.isin([0, 1])
    invalid_fraud_label = valid_integer_types["TX_FRAUD"] & ~fraud_label.isin([0, 1])
    _add_reason(reasons, invalid_fraud_label, "invalid_fraud_label")

    fraud_scenario = converted_integers["TX_FRAUD_SCENARIO"]
    valid_fraud_scenario = valid_integer_types["TX_FRAUD_SCENARIO"] & (
        fraud_scenario.isin([0, 1, 2, 3])
    )
    invalid_fraud_scenario = valid_integer_types["TX_FRAUD_SCENARIO"] & (
        ~fraud_scenario.isin([0, 1, 2, 3])
    )

    inconsistent_fraud_fields = (
        valid_fraud_label
        & valid_fraud_scenario
        & (
            ((fraud_scenario == 0) & (fraud_label != 0))
            | (fraud_scenario.isin([1, 2, 3]) & (fraud_label != 1))
        )
    )

    _add_reason(
        reasons,
        invalid_fraud_scenario | inconsistent_fraud_fields,
        "invalid_fraud_scenario",
    )

    time_seconds = converted_integers["TX_TIME_SECONDS"]
    time_days = converted_integers["TX_TIME_DAYS"]
    valid_time_fields = valid_integer_types["TX_TIME_SECONDS"] & valid_integer_types["TX_TIME_DAYS"]
    inconsistent_time_fields = valid_time_fields & (
        time_seconds.lt(0) | time_days.lt(0) | time_days.ne(time_seconds.floordiv(86400))
    )
    _add_reason(reasons, inconsistent_time_fields, "inconsistent_time_fields")

    source_date_mismatch = valid_datetime_type & (
        transaction_datetime.dt.date != source_dates.dt.date
    )
    _add_reason(reasons, source_date_mismatch, "source_date_mismatch")

    seen_transaction_ids = {int(value) for value in known_transaction_ids}

    for position in range(len(bronze)):
        if reasons[position]:
            continue

        transaction_id = int(converted_integers["TRANSACTION_ID"].iloc[position])
        if transaction_id in seen_transaction_ids:
            reasons[position].append("duplicate_transaction_id")
        else:
            seen_transaction_ids.add(transaction_id)

    valid_mask = pd.Series(
        [not row_reasons for row_reasons in reasons],
        index=bronze.index,
        dtype="bool",
    )

    silver = pd.DataFrame(index=bronze.index[valid_mask])
    silver["transaction_id"] = converted_integers["TRANSACTION_ID"][valid_mask].astype("int64")
    silver["tx_datetime"] = transaction_datetime[valid_mask]
    silver["customer_id"] = converted_integers["CUSTOMER_ID"][valid_mask].astype("int64")
    silver["terminal_id"] = converted_integers["TERMINAL_ID"][valid_mask].astype("int64")
    silver["tx_amount"] = transaction_amount[valid_mask].astype("float64")
    silver["tx_time_seconds"] = converted_integers["TX_TIME_SECONDS"][valid_mask].astype("int64")
    silver["tx_time_days"] = converted_integers["TX_TIME_DAYS"][valid_mask].astype("int64")
    silver["tx_fraud"] = converted_integers["TX_FRAUD"][valid_mask].astype("int8")
    silver["tx_fraud_scenario"] = converted_integers["TX_FRAUD_SCENARIO"][valid_mask].astype("int8")
    silver["source_file"] = bronze.loc[valid_mask, "source_file"].astype("string")
    silver["source_file_date"] = source_dates[valid_mask].dt.date
    silver["source_row_number"] = source_row_numbers[valid_mask].astype("int64")
    silver["ingested_at_utc"] = ingested_at[valid_mask]
    silver = silver.reset_index(drop=True)

    quarantine = bronze.loc[~valid_mask].copy()
    quarantine["rejection_reasons"] = pd.Series(
        [reasons[position] for position in range(len(bronze)) if not valid_mask.iloc[position]],
        index=quarantine.index,
        dtype="object",
    )
    quarantine = quarantine.reset_index(drop=True)

    return SilverValidationResult(
        silver=silver,
        quarantine=quarantine,
    )


_BRONZE_FILENAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


def _parse_bronze_file_date(bronze_file: Path) -> date:
    if not _BRONZE_FILENAME_PATTERN.fullmatch(bronze_file.name):
        raise ValueError(f"Invalid Bronze filename: {bronze_file.name}")

    try:
        return date.fromisoformat(bronze_file.stem)
    except ValueError as error:
        raise ValueError(f"Invalid Bronze filename: {bronze_file.name}") from error


def discover_bronze_files(bronze_dir: Path) -> list[Path]:
    """Return dated Bronze Parquet files in chronological order."""
    bronze_files = (
        path for path in bronze_dir.iterdir() if path.is_file() and path.suffix == ".parquet"
    )
    return sorted(bronze_files, key=_parse_bronze_file_date)


def _read_transaction_ids(silver_file: Path) -> set[int]:
    silver_data = pd.read_parquet(
        silver_file,
        columns=["transaction_id"],
    )
    return {int(transaction_id) for transaction_id in silver_data["transaction_id"].tolist()}


def ingest_silver_file(
    bronze_file: Path,
    silver_dir: Path,
    quarantine_dir: Path,
    known_transaction_ids: Collection[int] = (),
) -> tuple[Path, Path]:
    """Validate one Bronze file and publish Silver and Quarantine outputs."""
    _parse_bronze_file_date(bronze_file)

    if silver_dir.resolve() == quarantine_dir.resolve():
        raise ValueError("Silver and Quarantine directories must be different")

    silver_output = silver_dir / bronze_file.name
    quarantine_output = quarantine_dir / bronze_file.name

    if silver_output.exists() and quarantine_output.exists():
        return silver_output, quarantine_output

    if silver_output.exists() and not quarantine_output.exists():
        raise RuntimeError(f"Silver output exists without Quarantine output: {bronze_file.name}")

    bronze_data = pd.read_parquet(bronze_file)
    validation_result = validate_bronze_records(
        bronze_data,
        known_transaction_ids=known_transaction_ids,
    )

    silver_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    with (
        TemporaryDirectory(dir=silver_dir) as temporary_silver_dir,
        TemporaryDirectory(dir=quarantine_dir) as temporary_quarantine_dir,
    ):
        temporary_silver_output = Path(temporary_silver_dir) / silver_output.name
        temporary_quarantine_output = Path(temporary_quarantine_dir) / quarantine_output.name

        validation_result.silver.to_parquet(
            temporary_silver_output,
            engine="pyarrow",
            index=False,
        )
        validation_result.quarantine.to_parquet(
            temporary_quarantine_output,
            engine="pyarrow",
            index=False,
        )

        temporary_quarantine_output.replace(quarantine_output)
        temporary_silver_output.replace(silver_output)

    return silver_output, quarantine_output


def ingest_silver_directory(
    bronze_dir: Path,
    silver_dir: Path,
    quarantine_dir: Path,
) -> list[tuple[Path, Path]]:
    """Process Bronze files chronologically with dataset-wide deduplication."""
    known_transaction_ids: set[int] = set()

    if silver_dir.exists():
        for silver_file in sorted(silver_dir.glob("*.parquet")):
            known_transaction_ids.update(_read_transaction_ids(silver_file))

    outputs: list[tuple[Path, Path]] = []

    for bronze_file in discover_bronze_files(bronze_dir):
        output_paths = ingest_silver_file(
            bronze_file,
            silver_dir,
            quarantine_dir,
            known_transaction_ids=known_transaction_ids,
        )
        outputs.append(output_paths)
        known_transaction_ids.update(_read_transaction_ids(output_paths[0]))

    return outputs
