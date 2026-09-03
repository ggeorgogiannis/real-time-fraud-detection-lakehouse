# Transaction Test Fixtures

This directory contains small, deterministic datasets used by automated tests.

The fixtures use CSV rather than Pickle so their contents remain safe to open, human-readable and reviewable through Git. Tests that exercise raw ingestion will convert these files into temporary Pickle files at runtime.

These files are test specifications. They are not training data and are not intended to reproduce the statistical distribution of the complete dataset.

## Fixture Inventory

| File                          | Purpose                                                                                 | Expected outcome                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `transactions_valid.csv`      | Valid schema, time relationships, legitimate transactions and all three fraud scenarios | Five Silver records and no quarantined records                      |
| `transactions_invalid.csv`    | Missing, malformed and contract-violating values                                        | Seven quarantined records containing the expected rejection reasons |
| `transactions_duplicates.csv` | Exact and conflicting duplicate transaction IDs                                         | Three Silver records and two quarantined duplicate records          |

## Valid Transactions

`transactions_valid.csv` contains transaction IDs `1001` through `1005`.

It covers:

* Two legitimate transactions.
* Fraud scenarios `1`, `2` and `3`.
* Multiple transactions for the same customer.
* Positive integer and decimal transaction amounts.
* Consistent transaction timestamps, seconds and day values.

All rows represent `2018-04-01`.

## Invalid Transactions

Each row in `transactions_invalid.csv` targets a specific contract rule.

| Transaction ID | Primary expected rejection reason |
| -------------- | --------------------------------- |
| `2001`         | `missing_required_value`          |
| `2002`         | `negative_transaction_amount`     |
| `2003`         | `invalid_fraud_label`             |
| `2004`         | `invalid_fraud_scenario`          |
| `2005`         | `inconsistent_time_fields`        |
| `2006`         | `invalid_data_type`               |
| `2007`         | `source_date_mismatch`            |

Tests should verify that the expected reason is present. A record may contain additional rejection reasons when it violates more than one related rule.

Transaction `2007` must be written to a temporary raw file named `2018-04-01.pkl`. Its transaction timestamp belongs to `2018-04-02`, intentionally creating the source-date mismatch.

## Duplicate Transactions

`transactions_duplicates.csv` contains:

* Two identical occurrences of transaction ID `3001`.
* Two occurrences of transaction ID `3002` with different amounts.
* One unique control transaction with ID `3003`.

The first valid occurrence of each transaction ID should be retained. Later occurrences should be quarantined with `duplicate_transaction_id`.

## Runtime Conversion

Tests that require the source Pickle format should:

1. Read the appropriate CSV fixture.
2. Create the required Pandas DataFrame.
3. Write it to a temporary directory supplied by pytest.
4. Name the temporary file using the source contract, such as `2018-04-01.pkl`.
5. Run the pipeline against the temporary input.
6. Allow pytest to remove the temporary file after the test.

Generated Pickle files must not be committed to the repository.

## Maintenance Rules

When a fixture changes:

* Keep it as small as possible.
* Preserve deterministic row ordering.
* Update this README when expected behavior changes.
* Update the associated automated tests.
* Review the change against `docs/data_contract.md`.
