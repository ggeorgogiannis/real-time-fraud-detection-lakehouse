# Gold Data Models

## Purpose

The Gold layer converts validated Silver transactions into feature-ready and analytical datasets. Gold models contain only records accepted by the Silver layer.

## Input

Gold processing reads daily Parquet files from the Silver layer. Files are processed in chronological filename order, and transactions are ordered by `tx_datetime` and then `transaction_id` before historical features are calculated.

## Outputs

Gold processing publishes three Parquet tables:

| Table                  | File                             | Grain                                     |
| ---------------------- | -------------------------------- | ----------------------------------------- |
| Transaction features   | `transaction_features.parquet`   | One row per transaction                   |
| Customer daily summary | `customer_daily_summary.parquet` | One row per transaction date and customer |
| Terminal daily summary | `terminal_daily_summary.parquet` | One row per transaction date and terminal |

## Transaction Features

The transaction feature table retains the canonical Silver columns and adds:

| Column                                | Description                                                   |
| ------------------------------------- | ------------------------------------------------------------- |
| `transaction_date`                    | UTC calendar date of the transaction                          |
| `transaction_hour`                    | Transaction hour from 0 to 23                                 |
| `day_of_week`                         | Day number where Monday is 0 and Sunday is 6                  |
| `is_weekend`                          | 1 for Saturday or Sunday; otherwise 0                         |
| `is_night`                            | 1 for transactions between 00:00 and 05:59 UTC; otherwise 0   |
| `customer_previous_transaction_count` | Number of earlier transactions for the customer               |
| `customer_previous_mean_amount`       | Mean amount of the customer's earlier transactions            |
| `amount_to_customer_previous_mean`    | Current amount divided by the customer's previous mean amount |
| `terminal_previous_transaction_count` | Number of earlier transactions at the terminal                |

Historical customer and terminal features use only preceding records. The current transaction is excluded to prevent information leakage.

Transactions with the same timestamp are ordered by `transaction_id` to produce deterministic results. The previous mean and amount ratio are null for a customer's first transaction. The ratio is also null when the previous mean is zero.

## Customer Daily Summary

| Column              | Description                                  |
| ------------------- | -------------------------------------------- |
| `transaction_date`  | Transaction date                             |
| `customer_id`       | Customer identifier                          |
| `transaction_count` | Number of transactions                       |
| `total_amount`      | Sum of transaction amounts                   |
| `average_amount`    | Mean transaction amount                      |
| `maximum_amount`    | Largest transaction amount                   |
| `fraud_count`       | Number of fraudulent transactions            |
| `fraud_rate`        | Proportion of transactions labelled as fraud |

## Terminal Daily Summary

| Column              | Description                                  |
| ------------------- | -------------------------------------------- |
| `transaction_date`  | Transaction date                             |
| `terminal_id`       | Terminal identifier                          |
| `transaction_count` | Number of transactions                       |
| `unique_customers`  | Number of distinct customers                 |
| `total_amount`      | Sum of transaction amounts                   |
| `average_amount`    | Mean transaction amount                      |
| `fraud_count`       | Number of fraudulent transactions            |
| `fraud_rate`        | Proportion of transactions labelled as fraud |

## Reproducibility and Publication

The input schema and duplicate transaction identifiers are validated before publication. All three outputs are written to temporary files before the final paths are replaced.

Repeated runs over unchanged Silver data produce the same records and ordering.
