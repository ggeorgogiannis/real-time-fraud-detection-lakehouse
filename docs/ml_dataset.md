# Machine-Learning Dataset Contract

## Purpose

The machine-learning dataset provides reproducible, point-in-time transaction features for training and evaluating fraud-detection models.

Each row represents one transaction. Features must contain only information available when that transaction occurred.

## Source

The dataset is built from the Gold `transaction_features` table.

This table already contains transaction attributes and historical features calculated from strictly earlier transactions.

## Target

| Column     | Description                            |
| ---------- | -------------------------------------- |
| `tx_fraud` | Binary fraud label used as the target. |

The target must never appear in the feature matrix.

## Model Features

| Feature                               | Description                                              |
| ------------------------------------- | -------------------------------------------------------- |
| `tx_amount`                           | Amount of the current transaction.                       |
| `transaction_hour`                    | UTC hour in which the transaction occurred.              |
| `day_of_week`                         | UTC day of the week.                                     |
| `is_weekend`                          | Indicates whether the transaction occurred on a weekend. |
| `is_night`                            | Indicates whether the transaction occurred at night.     |
| `customer_previous_transaction_count` | Number of earlier transactions from the customer.        |
| `customer_previous_mean_amount`       | Mean amount of the customer’s earlier transactions.      |
| `amount_to_customer_previous_mean`    | Current amount relative to the customer’s previous mean. |
| `terminal_previous_transaction_count` | Number of earlier transactions at the terminal.          |

Customer and terminal identifiers are retained as metadata but are not baseline model features. Treating high-cardinality identifiers as numeric features could create misleading relationships and poor generalization.

## Metadata

The following columns identify, order or trace records but are excluded from the feature matrix:

* `transaction_id`
* `tx_datetime`
* `transaction_date`
* `customer_id`
* `terminal_id`
* `source_file`
* `source_file_date`
* `source_row_number`
* `ingested_at_utc`

## Leakage Exclusions

The following values must not be used as model features:

* `tx_fraud`, because it is the prediction target.
* `tx_fraud_scenario`, because it describes the mechanism used to generate the fraud label.
* Fraud counts, fraud rates and other target-derived aggregates from analytical reporting marts.
* Features calculated using the current transaction’s label or any future transaction.

Historical feature calculations must use only transactions that occurred strictly before the transaction being scored.

## Temporal Splitting

The dataset is divided chronologically using two explicit UTC boundaries:

* Training rows have `tx_datetime < train_end`.
* Validation rows have `train_end <= tx_datetime < validation_end`.
* Test rows have `tx_datetime >= validation_end`.

Random train-test splitting is not permitted because it can expose future transaction patterns during training.

All preprocessing and model fitting must use the training partition only. The validation and test partitions must remain unseen during fitting.

## Ordering and Uniqueness

Rows are ordered by:

1. `tx_datetime`
2. `transaction_id`

Each `transaction_id` must be unique.

The training, validation and test partitions must be non-empty and must not overlap.

## Missing and Invalid Values

Missing historical mean and ratio values are permitted when a customer has no previous transactions. Their treatment will be defined by the model preprocessing pipeline.

Infinite numeric values are not permitted. Binary indicator and target columns must contain only `0` or `1`.

## Modeling Boundary

This dataset is intended for leakage-safe model development.

The dbt customer and terminal risk marts are intended for retrospective reporting because they contain aggregated fraud outcomes. They must not be joined into the training dataset.
