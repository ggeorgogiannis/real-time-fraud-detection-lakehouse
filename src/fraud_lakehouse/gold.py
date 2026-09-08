import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fraud_lakehouse.silver import (
    BRONZE_METADATA_COLUMNS,
    SOURCE_TO_SILVER_COLUMNS,
)

SILVER_COLUMNS = (
    *SOURCE_TO_SILVER_COLUMNS.values(),
    *BRONZE_METADATA_COLUMNS,
)

GOLD_OUTPUT_FILENAMES = {
    "transaction_features": "transaction_features.parquet",
    "customer_daily_summary": "customer_daily_summary.parquet",
    "terminal_daily_summary": "terminal_daily_summary.parquet",
}

_SILVER_FILENAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


@dataclass(frozen=True, slots=True)
class GoldTables:
    """Analytical tables created from validated Silver records."""

    transaction_features: pd.DataFrame
    customer_daily_summary: pd.DataFrame
    terminal_daily_summary: pd.DataFrame


def _validate_silver_schema(silver_data: pd.DataFrame) -> None:
    missing_columns = sorted(set(SILVER_COLUMNS) - set(silver_data.columns))
    if missing_columns:
        details = ", ".join(missing_columns)
        raise ValueError(f"Silver data is missing required columns: {details}")

    if silver_data["transaction_id"].duplicated().any():
        raise ValueError("Silver data contains duplicate transaction IDs")


def _parse_silver_file_date(silver_file: Path) -> date:
    if not _SILVER_FILENAME_PATTERN.fullmatch(silver_file.name):
        raise ValueError(f"Invalid Silver filename: {silver_file.name}")

    try:
        return date.fromisoformat(silver_file.stem)
    except ValueError as error:
        raise ValueError(f"Invalid Silver filename: {silver_file.name}") from error


def discover_silver_files(silver_dir: Path) -> list[Path]:
    """Return daily Silver Parquet files in chronological order."""
    silver_files = (
        path for path in silver_dir.iterdir() if path.is_file() and path.suffix == ".parquet"
    )
    return sorted(silver_files, key=_parse_silver_file_date)


def load_silver_directory(silver_dir: Path) -> pd.DataFrame:
    """Load and combine validated daily Silver files."""
    silver_files = discover_silver_files(silver_dir)
    if not silver_files:
        raise ValueError("No Silver Parquet files found")

    daily_frames: list[pd.DataFrame] = []

    for silver_file in silver_files:
        daily_data = pd.read_parquet(silver_file)
        _validate_silver_schema(daily_data)
        daily_frames.append(daily_data)

    combined_data = pd.concat(daily_frames, ignore_index=True)
    _validate_silver_schema(combined_data)
    return combined_data


def build_gold_tables(silver_data: pd.DataFrame) -> GoldTables:
    """Create transaction features and daily analytical summaries."""
    _validate_silver_schema(silver_data)

    features = silver_data.copy()
    features["tx_datetime"] = pd.to_datetime(
        features["tx_datetime"],
        errors="coerce",
        utc=True,
    )
    features["tx_amount"] = pd.to_numeric(
        features["tx_amount"],
        errors="coerce",
    )

    if features["tx_datetime"].isna().any():
        raise ValueError("Silver data contains invalid transaction timestamps")

    invalid_amount = features["tx_amount"].isna() | features["tx_amount"].isin(
        [float("inf"), float("-inf")]
    )
    if invalid_amount.any():
        raise ValueError("Silver data contains invalid transaction amounts")

    features = features.sort_values(
        ["tx_datetime", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    features["transaction_date"] = features["tx_datetime"].dt.date
    features["transaction_hour"] = features["tx_datetime"].dt.hour.astype("int8")
    features["day_of_week"] = features["tx_datetime"].dt.dayofweek.astype("int8")
    features["is_weekend"] = (features["day_of_week"] >= 5).astype("int8")
    features["is_night"] = (
        features["transaction_hour"]
        .between(
            0,
            5,
        )
        .astype("int8")
    )

    customer_group = features.groupby(
        "customer_id",
        sort=False,
    )
    customer_previous_count = customer_group.cumcount().astype("int64")
    customer_cumulative_amount = customer_group["tx_amount"].cumsum()
    customer_previous_total = customer_cumulative_amount - features["tx_amount"]
    customer_previous_mean = customer_previous_total.div(
        customer_previous_count.where(customer_previous_count > 0)
    ).astype("float64")

    features["customer_previous_transaction_count"] = customer_previous_count
    features["customer_previous_mean_amount"] = customer_previous_mean
    features["amount_to_customer_previous_mean"] = (
        features["tx_amount"].div(customer_previous_mean.where(customer_previous_mean > 0))
    ).astype("float64")

    features["terminal_previous_transaction_count"] = (
        features.groupby(
            "terminal_id",
            sort=False,
        )
        .cumcount()
        .astype("int64")
    )

    customer_daily_summary = (
        features.groupby(
            ["transaction_date", "customer_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            transaction_count=("transaction_id", "size"),
            total_amount=("tx_amount", "sum"),
            average_amount=("tx_amount", "mean"),
            maximum_amount=("tx_amount", "max"),
            fraud_count=("tx_fraud", "sum"),
        )
        .sort_values(
            ["transaction_date", "customer_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    customer_daily_summary["transaction_count"] = customer_daily_summary[
        "transaction_count"
    ].astype("int64")
    customer_daily_summary["fraud_count"] = customer_daily_summary["fraud_count"].astype("int64")
    customer_daily_summary["fraud_rate"] = (
        customer_daily_summary["fraud_count"] / customer_daily_summary["transaction_count"]
    ).astype("float64")

    terminal_daily_summary = (
        features.groupby(
            ["transaction_date", "terminal_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            transaction_count=("transaction_id", "size"),
            unique_customers=("customer_id", "nunique"),
            total_amount=("tx_amount", "sum"),
            average_amount=("tx_amount", "mean"),
            fraud_count=("tx_fraud", "sum"),
        )
        .sort_values(
            ["transaction_date", "terminal_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    terminal_daily_summary["transaction_count"] = terminal_daily_summary[
        "transaction_count"
    ].astype("int64")
    terminal_daily_summary["unique_customers"] = terminal_daily_summary["unique_customers"].astype(
        "int64"
    )
    terminal_daily_summary["fraud_count"] = terminal_daily_summary["fraud_count"].astype("int64")
    terminal_daily_summary["fraud_rate"] = (
        terminal_daily_summary["fraud_count"] / terminal_daily_summary["transaction_count"]
    ).astype("float64")

    return GoldTables(
        transaction_features=features,
        customer_daily_summary=customer_daily_summary,
        terminal_daily_summary=terminal_daily_summary,
    )


def write_gold_tables(
    silver_dir: Path,
    gold_dir: Path,
) -> dict[str, Path]:
    """Build and atomically publish the Gold analytical tables."""
    silver_data = load_silver_directory(silver_dir)
    gold_tables = build_gold_tables(silver_data)

    gold_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        table_name: gold_dir / filename for table_name, filename in GOLD_OUTPUT_FILENAMES.items()
    }

    with TemporaryDirectory(dir=gold_dir) as temporary_dir:
        temporary_paths: dict[str, Path] = {}

        for table_name, output_path in output_paths.items():
            temporary_path = Path(temporary_dir) / output_path.name
            table = getattr(gold_tables, table_name)
            table.to_parquet(
                temporary_path,
                engine="pyarrow",
                index=False,
            )
            temporary_paths[table_name] = temporary_path

        for table_name in (
            "customer_daily_summary",
            "terminal_daily_summary",
            "transaction_features",
        ):
            temporary_paths[table_name].replace(output_paths[table_name])

    return output_paths
