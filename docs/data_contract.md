# Transaction Data Contract

## Purpose

This document defines the input contract for transaction data entering the batch lakehouse. It provides a shared reference for ingestion, validation, quarantine handling and automated tests.

Contract version: `1.0`

## Source

The dataset is published by the [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).

Source files are Pandas Pickle files containing one day of synthetic transactions.

Because Pickle files can execute arbitrary Python code during deserialization, the pipeline must only load files obtained from the trusted project source. Arbitrary user-provided Pickle files must not be processed.

## File Contract

Raw files are stored locally using this structure:

```text
data/raw/YYYY-MM-DD.pkl
```

Each file must satisfy these rules:

* The filename must contain a valid calendar date using the `YYYY-MM-DD.pkl` format.
* The file must deserialize into a Pandas DataFrame.
* All required source columns must be present.
* Unexpected columns are treated as schema drift and require review.
* Every `TX_DATETIME` value must belong to the date represented by the filename.
* File discovery order must not affect pipeline results.

Downloaded files are never modified in place.

## Transaction Schema

Bronze preserves the original source column names and values. Silver converts the names to lowercase snake case and applies the canonical types and validation rules below.

| Source/Bronze column | Silver column       | Canonical type | Required | Validation                                 |
| -------------------- | ------------------- | -------------- | -------- | ------------------------------------------ |
| `TRANSACTION_ID`     | `transaction_id`    | `int64`        | Yes      | Non-negative and unique across the dataset |
| `TX_DATETIME`        | `tx_datetime`       | Timestamp      | Yes      | Valid timestamp interpreted as UTC         |
| `CUSTOMER_ID`        | `customer_id`       | `int64`        | Yes      | Non-negative                               |
| `TERMINAL_ID`        | `terminal_id`       | `int64`        | Yes      | Non-negative                               |
| `TX_AMOUNT`          | `tx_amount`         | `float64`      | Yes      | Finite and greater than or equal to zero   |
| `TX_TIME_SECONDS`    | `tx_time_seconds`   | `int64`        | Yes      | Non-negative                               |
| `TX_TIME_DAYS`       | `tx_time_days`      | `int64`        | Yes      | Non-negative                               |
| `TX_FRAUD`           | `tx_fraud`          | `int8`         | Yes      | Either `0` or `1`                          |
| `TX_FRAUD_SCENARIO`  | `tx_fraud_scenario` | `int8`         | Yes      | One of `0`, `1`, `2` or `3`                |

The source timestamps do not include timezone information. The project interprets them as UTC so that later batch and streaming components use one consistent time standard.

## Cross-Field Rules

A valid transaction must also satisfy these relationships:

* `tx_time_days` must equal `tx_time_seconds // 86400`.
* Fraud scenario `0` must have `tx_fraud = 0`.
* Fraud scenarios `1`, `2` and `3` must have `tx_fraud = 1`.
* The calendar date of `tx_datetime` must match the date in `source_file_date`.

## Bronze Metadata

Bronze records add the following ingestion metadata without changing the source values:

| Column              | Type      | Description                                          |
| ------------------- | --------- | ---------------------------------------------------- |
| `source_file`       | String    | Name of the raw file containing the record           |
| `source_file_date`  | Date      | Date parsed from the source filename                 |
| `source_row_number` | `int64`   | Zero-based position of the record in the source file |
| `ingested_at_utc`   | Timestamp | UTC time at which the record was ingested            |

## Validation Outcomes

Every discovered record has one of two outcomes:

1. Valid records are written to Silver using the canonical column names and types.
2. Invalid records are written to Quarantine with their original values, source metadata and one or more rejection reasons.

Example rejection reasons include:

* `missing_required_value`
* `invalid_data_type`
* `negative_transaction_amount`
* `invalid_fraud_label`
* `invalid_fraud_scenario`
* `inconsistent_time_fields`
* `source_date_mismatch`
* `duplicate_transaction_id`

A file that cannot be read or does not contain the required schema fails ingestion as a file-level error. It must not produce a partial Bronze or Silver output.

## Duplicate and Idempotency Rules

`transaction_id` is the transaction business key.

Within newly discovered data:

* The first valid occurrence of a transaction ID is retained.
* Later occurrences of the same ID are sent to Quarantine.
* A repeated pipeline run over unchanged input must not create additional Bronze, Silver or Quarantine records.
* A reused transaction ID containing different values is treated as a conflicting duplicate and must never silently overwrite an existing transaction.

## Contract Evolution

Changes to required columns, canonical types, validation rules or key definitions require:

* An updated contract version.
* Automated test changes.
* A documented migration or compatibility decision.
* Review through a pull request.
