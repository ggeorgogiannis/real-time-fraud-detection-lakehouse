from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from fraud_lakehouse.gold import discover_silver_files

GOLD_VIEWS = {
    "gold_transaction_features": "transaction_features.parquet",
    "gold_customer_daily_summary": "customer_daily_summary.parquet",
    "gold_terminal_daily_summary": "terminal_daily_summary.parquet",
}

ANALYTICS_VIEW_NAMES = (
    "silver_transactions",
    *GOLD_VIEWS,
)


def _quote_sql_string(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) * 2)}'"


def _parquet_file_list(paths: Sequence[Path]) -> str:
    quoted_paths = ", ".join(_quote_sql_string(path.resolve().as_posix()) for path in paths)
    return f"[{quoted_paths}]"


def _create_parquet_view(
    connection: duckdb.DuckDBPyConnection,
    view_name: str,
    parquet_files: Sequence[Path],
) -> None:
    if not parquet_files:
        raise ValueError(f"No Parquet files were provided for view {view_name!r}.")

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT *
        FROM read_parquet(
            {_parquet_file_list(parquet_files)},
            union_by_name = true
        )
        """
    )


def _validated_silver_files(silver_dir: Path) -> list[Path]:
    if not silver_dir.is_dir():
        raise FileNotFoundError(f"Silver directory does not exist: {silver_dir}")

    silver_files = discover_silver_files(silver_dir)

    if not silver_files:
        raise ValueError(f"No Silver Parquet files found in: {silver_dir}")

    return silver_files


def _validated_gold_files(gold_dir: Path) -> dict[str, Path]:
    if not gold_dir.is_dir():
        raise FileNotFoundError(f"Gold directory does not exist: {gold_dir}")

    gold_files = {view_name: gold_dir / filename for view_name, filename in GOLD_VIEWS.items()}

    missing_files = [path for path in gold_files.values() if not path.is_file()]

    if missing_files:
        missing_names = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(f"Missing Gold Parquet files: {missing_names}")

    return gold_files


def build_analytics_database(
    silver_dir: Path,
    gold_dir: Path,
    database_path: Path,
) -> Path:
    """Create a persistent DuckDB database backed by Parquet views."""
    silver_files = _validated_silver_files(silver_dir)
    gold_files = _validated_gold_files(gold_dir)

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        prefix="analytics-",
        dir=database_path.parent,
    ) as temporary_dir:
        temporary_database = Path(temporary_dir) / database_path.name

        with duckdb.connect(str(temporary_database)) as connection:
            _create_parquet_view(
                connection,
                "silver_transactions",
                silver_files,
            )

            for view_name, gold_file in gold_files.items():
                _create_parquet_view(
                    connection,
                    view_name,
                    [gold_file],
                )

        temporary_database.replace(database_path)

    return database_path
