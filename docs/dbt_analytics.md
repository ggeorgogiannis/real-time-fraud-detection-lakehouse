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

### `fct_customer_daily_risk`

Provides one row per customer and transaction date. It preserves the daily customer metrics produced by Gold and adds rolling seven-calendar-day transaction, amount, fraud-count and fraud-rate indicators.

### `fct_terminal_daily_risk`

Provides one row per terminal and transaction date. It preserves daily terminal activity and adds rolling seven-calendar-day transaction, amount, fraud-count and fraud-rate indicators.

### `rpt_daily_fraud_overview`

Provides one reporting-ready row per transaction date. It combines the daily fraud totals with:

* Active customer count.
* Customers associated with fraud.
* Active terminal count.
* Terminals associated with fraud.

## Analytical and Machine-Learning Boundary

The customer and terminal risk marts contain aggregated fraud labels. They are designed for retrospective analysis, operational reporting and data-quality verification.

These models must not be used directly as machine-learning input features because doing so would expose the prediction target and create data leakage. Phase 3 will create separate time-aware feature datasets using only information available before each transaction.

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
