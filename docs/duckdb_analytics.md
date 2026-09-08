# DuckDB Analytics

## Purpose

DuckDB provides a local SQL interface over the validated Silver and analytical Gold Parquet datasets.

The DuckDB database stores persistent view definitions. The Parquet files remain the source of truth and are queried directly rather than copied into database tables.

## Analytical Views

| View                          | Source                           | Grain                                     |
| ----------------------------- | -------------------------------- | ----------------------------------------- |
| `silver_transactions`         | Daily Silver Parquet files       | One row per validated transaction         |
| `gold_transaction_features`   | `transaction_features.parquet`   | One row per transaction                   |
| `gold_customer_daily_summary` | `customer_daily_summary.parquet` | One row per transaction date and customer |
| `gold_terminal_daily_summary` | `terminal_daily_summary.parquet` | One row per transaction date and terminal |

## Build the Database

Run the Bronze, Silver and Gold pipeline first:

```bash
fraud-lakehouse run \
  --raw-dir data/raw \
  --output-dir data
```

Create the analytical database:

```bash
fraud-lakehouse build-analytics \
  --silver-dir data/silver \
  --gold-dir data/gold \
  --database-path data/analytics/fraud_lakehouse.duckdb
```

The database is rebuilt through a temporary file before replacing the published database. This prevents an incomplete database from being exposed if creation fails.

## Query the Database

The database can be queried through Python:

```python
import duckdb

with duckdb.connect(
    "data/analytics/fraud_lakehouse.duckdb",
    read_only=True,
) as connection:
    fraud_by_day = connection.execute(
        """
        SELECT
            transaction_date,
            SUM(fraud_count) AS fraud_count,
            SUM(transaction_count) AS transaction_count
        FROM gold_customer_daily_summary
        GROUP BY transaction_date
        ORDER BY transaction_date
        """
    ).fetchdf()

print(fraud_by_day)
```

Because the views reference the Parquet files directly, the Silver and Gold files must remain available at their original paths.
