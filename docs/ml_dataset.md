# Machine-Learning Dataset Contract

## Purpose

The machine-learning dataset provides reproducible, point-in-time transaction features for training and evaluating fraud-detection models.

Each row represents one transaction. Features must contain only information available when that transaction occurred.

## Source

The dataset is built from the Gold `transaction_features.parquet` table.

This table contains transaction attributes and historical features calculated from strictly earlier transactions.

## Target

| Column     | Description                            |
| ---------- | -------------------------------------- |
| `tx_fraud` | Binary fraud label used as the target. |

The target must never appear in the feature matrix.

## Model Features

| Feature                               | Description                                               |
| ------------------------------------- | --------------------------------------------------------- |
| `tx_amount`                           | Amount of the current transaction.                        |
| `transaction_hour`                    | UTC hour in which the transaction occurred.               |
| `day_of_week`                         | UTC day of the week.                                      |
| `is_weekend`                          | Indicates whether the transaction occurred on a weekend.  |
| `is_night`                            | Indicates whether the transaction occurred at night.      |
| `customer_previous_transaction_count` | Number of earlier transactions from the customer.         |
| `customer_previous_mean_amount`       | Mean amount of the customer's earlier transactions.       |
| `amount_to_customer_previous_mean`    | Current amount relative to the customer's previous mean.  |
| `terminal_previous_transaction_count` | Number of earlier transactions processed by the terminal. |

Customer and terminal identifiers are retained as metadata but are not baseline model features. Treating high-cardinality identifiers as numeric features could create misleading relationships and poor generalization.

## Metadata Columns

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
* Features calculated using the current transaction's label or any future transaction.

Historical feature calculations must use only transactions that occurred strictly before the transaction being scored.

## Validation Rules

Before splitting, the dataset builder verifies that:

* All required metadata, feature and target columns are present.
* Every `transaction_id` is unique.
* Required values are not missing.
* Timestamps and dates are valid.
* `transaction_date` matches the UTC date of `tx_datetime`.
* The target and binary indicators contain only `0` or `1`.
* Transaction hours and days of the week are within their valid ranges.
* Historical transaction counts are non-negative.
* Numeric model features do not contain infinite values.

Missing historical mean and ratio values are permitted only when a customer has no previous transactions. Their treatment must be defined by the model preprocessing pipeline.

## Temporal Splitting

The dataset is divided chronologically using two explicit timezone-aware boundaries:

* Training rows have `tx_datetime < train_end`.
* Validation rows have `train_end <= tx_datetime < validation_end`.
* Test rows have `tx_datetime >= validation_end`.

The supplied boundaries are normalized to UTC. The training boundary must be earlier than the validation boundary, and all three partitions must be non-empty.

Random train-test splitting is not permitted because it can expose future transaction patterns during training.

All preprocessing and model fitting must use the training partition only. The validation and test partitions must remain unseen during fitting.

## Ordering and Uniqueness

Rows are ordered by:

1. `tx_datetime`
2. `transaction_id`

This produces deterministic ordering when multiple transactions have the same timestamp.

The training, validation and test partitions do not overlap.

## Building the Dataset

Run the batch pipeline first so that the Gold transaction features are available. Then build the chronological model dataset:

```bash
fraud-lakehouse build-ml-dataset \
  --transaction-features-path data/gold/transaction_features.parquet \
  --output-dir data/ml \
  --train-end "2018-08-01T00:00:00Z" \
  --validation-end "2018-09-01T00:00:00Z"
```

The boundary values should be chosen according to the time range of the input dataset. Each boundary must include a timezone.

## Output Artifacts

The command creates or replaces:

| Artifact                | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `train.parquet`         | Transactions before the training boundary.            |
| `validation.parquet`    | Transactions within the validation period.            |
| `test.parquet`          | Transactions on or after the validation boundary.     |
| `dataset_metadata.json` | Dataset contract, boundaries and partition summaries. |

The metadata file records:

* The metadata schema version.
* The ordered model feature list.
* The target column.
* The normalized UTC split boundaries.
* Each partition's filename, row count and fraud count.

Partition files are written to temporary paths before publication. The metadata file is published last so that it represents a complete set of model dataset artifacts.

Generated model datasets remain local and are excluded from Git.

## Modeling Boundary

This dataset is intended for leakage-safe model development.

The dbt customer and terminal risk marts are intended for retrospective reporting because they contain aggregated fraud outcomes. They must not be joined into the training dataset.
