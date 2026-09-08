# dbt Analytics

## Purpose

The dbt project manages tested SQL transformations on top of the DuckDB analytical database.

DuckDB exposes the Silver and Gold Parquet datasets as persistent views. dbt treats those views as sources and builds documented staging models and analytical marts.

## Project Structure

```text
dbt/
├── dbt_project.yml
├── profiles.yml
├── models/
│   ├── staging/
│   └── marts/
├── tests/
└── macros/
```

## Sources

The `lakehouse` source contains:

| Source                        | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| `silver_transactions`         | Validated and deduplicated Silver transactions |
| `gold_transaction_features`   | Transaction-level fraud features               |
| `gold_customer_daily_summary` | Daily customer metrics                         |
| `gold_terminal_daily_summary` | Daily terminal metrics                         |

Source tests verify identifiers, required values and accepted fraud labels.

## Staging Models

The staging layer creates dbt-managed views over the DuckDB sources:

* `stg_transactions`
* `stg_transaction_features`
* `stg_customer_daily_summary`
* `stg_terminal_daily_summary`

These models are created in the `analytics_staging` schema.

## Analytical Marts

### `fct_daily_fraud`

The first mart contains one row per transaction date with:

* Transaction count
* Total transaction amount
* Fraud count
* Fraud rate

The model is materialized as a table in the `analytics_marts` schema.

## Data Tests

The dbt project verifies:

* Required source and mart values
* Unique transaction identifiers
* Accepted fraud labels
* Unique customer-day and terminal-day records
* Fraud rates between zero and one
* Reconciliation of daily transaction and fraud counts

## Running dbt

Build the DuckDB database first:

```bash
fraud-lakehouse build-analytics \
  --silver-dir data/silver \
  --gold-dir data/gold \
  --database-path data/analytics/fraud_lakehouse.duckdb
```

Validate the dbt configuration:

```bash
dbt debug \
  --project-dir dbt \
  --profiles-dir dbt
```

Build all models and execute all data tests:

```bash
dbt build \
  --project-dir dbt \
  --profiles-dir dbt
```

A different database can be selected without changing the tracked profile:

```bash
FRAUD_LAKEHOUSE_DUCKDB_PATH=/absolute/path/to/database.duckdb \
  dbt build \
  --project-dir dbt \
  --profiles-dir dbt
```

The automated Python integration suite also runs the complete dbt build against an isolated temporary database.
