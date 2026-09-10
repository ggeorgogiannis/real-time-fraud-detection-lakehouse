# Real-Time Fraud Detection Lakehouse

A local-first data engineering project for processing transaction data and detecting fraud.

The project begins as a tested batch pipeline and will gradually evolve into a streaming lakehouse. Each new tool will be introduced only when the existing implementation creates a clear reason to use it.

## Current Status

Phase 1 is complete. The first executable local batch pipeline was published as [v0.1.0](https://github.com/ggeorgogiannis/real-time-fraud-detection-lakehouse/releases/tag/v0.1.0).

The batch pipeline can:

* Discover and ingest trusted daily transaction files.
* Preserve source records and ingestion metadata in Bronze.
* Validate, type-convert, deduplicate and quarantine records in Silver.
* Create historical transaction features and daily customer and terminal summaries in Gold.
* Publish Parquet outputs safely and reproducibly.
* Run through a documented command-line interface.
* Verify its behaviour through automated unit and integration tests.

Phase 2 is complete and was published as [v0.2.0](https://github.com/ggeorgogiannis/real-time-fraud-detection-lakehouse/releases/tag/v0.2.0). DuckDB exposes the Silver and Gold datasets through persistent SQL views, while dbt builds tested staging models, analytical marts and reporting-ready fraud summaries.

Phase 3 is underway. The project now includes leakage-safe chronological model datasets, training-only preprocessing, a dummy prior baseline, class-balanced logistic regression and an imbalance-aware XGBoost classifier. All three models are evaluated on validation and test periods using metrics designed for imbalanced classification and are published with their fitted preprocessing pipelines.

The next milestone is running the three-model workflow on the complete dataset, comparing validation performance and selecting an appropriate fraud-classification threshold.

## Why This Project

Fraud detection is often presented only as a classification problem. In practice, the model depends on a larger data system that must handle ingestion, validation, historical features, reproducibility and monitoring.

This project focuses on that complete workflow. Its purpose is to explore how the components fit together and document the engineering decisions made during development.

## Current Architecture

`Daily transaction files -> Bronze -> Silver -> Gold`

`Gold -> DuckDB -> dbt -> Analytical marts and reporting models`

`Gold transaction features -> Chronological ML datasets -> Fraud models`

| Component   | Responsibility                                                          |
| ----------- | ----------------------------------------------------------------------- |
| Bronze      | Store ingested transactions with minimal changes and ingestion metadata |
| Silver      | Apply schema checks, type conversions, deduplication and quality rules  |
| Gold        | Create point-in-time features, customer summaries and analytical tables |
| Quarantine  | Preserve rejected records together with their validation failures       |
| DuckDB      | Expose Silver and Gold Parquet datasets through persistent SQL views    |
| dbt         | Manage tested SQL transformations, staging models and analytical marts  |
| ML datasets | Create validated chronological training, validation and test partitions |

The completed batch pipeline provides the common foundation for the analytical and machine-learning workflows. It will also serve as the reference implementation for the later Kafka and Spark streaming pipeline.

## Dataset

The project is based on the methodology and synthetic transaction simulator documented in the [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook).

The transaction files are obtained from the handbook's [simulated-data-raw repository](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).

The data is organized into daily files and includes transaction timestamps, customer identifiers, terminal identifiers, transaction amounts and fraud labels. This structure supports incremental ingestion and time-aware fraud analysis.

The dataset is downloaded separately from its official source and is not redistributed through this repository.

## Data Storage Policy

Downloaded data and generated pipeline outputs remain on the local machine.

They are excluded from Git because:

* Transaction files are input data rather than source code.
* Generated Parquet files can be reproduced by running the pipeline.
* Binary files cannot be reviewed through meaningful line-by-line Git differences.
* Large files make cloning and repository history unnecessarily heavy.
* Generated files can become inconsistent with the code that created them.
* Dataset distribution should continue through the original publisher.

The repository includes small synthetic fixtures for automated tests. These fixtures cover legitimate transactions, fraud labels, duplicates and invalid records without requiring the complete dataset.

## Local-First Development

The complete platform will run locally without paid cloud services or subscriptions.

The planned stack includes:

| Area                       | Tools                               |
| -------------------------- | ----------------------------------- |
| Batch processing           | Python, Pandas, PyArrow and Parquet |
| Analytical transformations | DuckDB and dbt Core                 |
| Machine learning           | scikit-learn and XGBoost            |
| Local infrastructure       | Docker and Docker Compose           |
| Orchestration              | Apache Airflow                      |
| Distributed processing     | Apache Spark                        |
| Event streaming            | Apache Kafka                        |
| Experiment tracking        | MLflow                              |
| Visualization              | Streamlit or Apache Superset        |
| Container orchestration    | kind or Minikube                    |

The stack may change as the project develops. Significant changes will be documented together with the reasoning behind them.

## Development Plan

### Phase 1: Batch Pipeline

Implement Bronze, Silver and Gold processing with Python and Parquet. Add schema validation, quarantine handling, idempotency and automated tests.

Completed in [v0.1.0](https://github.com/ggeorgogiannis/real-time-fraud-detection-lakehouse/releases/tag/v0.1.0).

### Phase 2: Analytical Lakehouse

Phase 2 is complete. DuckDB provides persistent SQL views over the Silver and Gold Parquet outputs, while dbt builds staging views, a daily fraud fact table, seven-day customer and terminal risk marts, and a reporting-ready daily fraud overview. The analytical layer is covered by source, schema, uniqueness, range, reconciliation and Python integration tests.

The risk marts include observed fraud labels for analytical reporting and must not be used directly as machine-learning features. Phase 3 will introduce leakage-safe, time-aware model datasets and baseline fraud models.

Completed in [v0.2.0](https://github.com/ggeorgogiannis/real-time-fraud-detection-lakehouse/releases/tag/v0.2.0).

### Phase 3: Fraud Detection

Phase 3 is underway. The project includes a leakage-safe dataset contract, chronological training, validation and test partitions, training-only preprocessing, and reproducible model artifacts.

A dummy prior classifier provides the non-informative reference, while class-balanced logistic regression provides an interpretable statistical baseline. An imbalance-aware XGBoost classifier adds nonlinear modeling and feature interactions using a class-weight ratio calculated exclusively from the training partition.

Evaluation reports average precision as the primary metric together with ROC AUC, precision, recall, F1 and confusion-matrix counts. The next steps are to run the workflow on the complete chronological dataset, compare validation performance, select an operating threshold and investigate calibration and validation-based tuning.

### Phase 4: Local Platform

Containerize the services and orchestrate scheduled batch runs with Apache Airflow.

### Phase 5: Streaming Pipeline

Simulate live transactions through Kafka and process them with Spark Structured Streaming.

### Phase 6: MLOps and Deployment

Track experiments with MLflow, add monitoring and deploy the completed platform to a local Kubernetes cluster.

## Development Practices

The project will use:

* Short-lived Git branches and pull requests.
* Conventional commit messages.
* Type hints and focused docstrings.
* Structured logging.
* Unit and integration tests.
* Automated linting and testing.
* Configuration separated from application logic.
* Documented architectural decisions.
* Reproducible setup and execution commands.

Comments will explain business rules and non-obvious decisions rather than restating the code.

## Repository Progress

- [x] Create the public GitHub repository.
- [x] Configure a project-local Python 3.11 environment.
- [x] Define the initial repository structure.
- [x] Document the transaction data contract.
- [x] Add small synthetic test fixtures.
- [x] Implement Bronze ingestion.
- [x] Implement Silver validation and quarantine handling.
- [x] Implement Gold features and analytical tables.
- [x] Add continuous integration.
- [x] Publish the first executable release.
- [x] Add the DuckDB analytical database and views.
- [x] Create the dbt project and local DuckDB profile.
- [x] Add dbt staging models and data tests.
- [x] Add the daily fraud analytical mart.
- [x] Add seven-day customer and terminal risk marts.
- [x] Add the reporting-ready daily fraud overview.
- [x] Automate dbt data and integration tests.
- [x] Create leakage-safe model datasets.
- [x] Train and evaluate baseline fraud models.
- [x] Add an imbalance-aware XGBoost fraud model.

## Running the Project

The project requires Python 3.11.

Create and activate the project environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -e ".[dev]"
```

### Run the Batch Pipeline

Place daily transaction files in `data/raw`. Input files must use the `YYYY-MM-DD.pkl` naming convention.

Run the complete Bronze, Silver and Gold pipeline:

```bash
fraud-lakehouse run \
  --raw-dir data/raw \
  --output-dir data
```

The command creates or updates:

* `data/bronze`
* `data/silver`
* `data/quarantine`
* `data/gold`

An explicit ingestion timestamp can be supplied when a reproducible run is required:

```bash
fraud-lakehouse run \
  --raw-dir data/raw \
  --output-dir data \
  --ingested-at-utc "2026-09-08T12:00:00Z"
```

### Build the Machine-Learning Dataset

After running the batch pipeline, create chronological training, validation and test datasets from the Gold transaction features:

```bash
fraud-lakehouse build-ml-dataset \
  --transaction-features-path data/gold/transaction_features.parquet \
  --output-dir data/ml \
  --train-end "2018-08-01T00:00:00Z" \
  --validation-end "2018-09-01T00:00:00Z"
```

Choose boundaries that leave non-empty training, validation and test periods for the available data. Both boundaries must include a timezone.

The command creates or replaces:

* `data/ml/train.parquet`
* `data/ml/validation.parquet`
* `data/ml/test.parquet`
* `data/ml/dataset_metadata.json`

The metadata file records the model features, target column, normalized UTC boundaries, row counts and fraud counts. See [`docs/ml_dataset.md`](docs/ml_dataset.md) for the feature contract, leakage exclusions, validation rules and split semantics.

### Train the Fraud Models

After creating the chronological model dataset, train and evaluate the fraud classifiers:

```bash
fraud-lakehouse train-baselines \
  --dataset-dir data/ml \
  --output-dir data/models \
  --threshold 0.5
```

The command creates or replaces:

* `data/models/dummy_prior.joblib`
* `data/models/logistic_regression.joblib`
* `data/models/xgboost.joblib`
* `data/models/metrics.json`

Each model artifact contains the fitted training preprocessor, estimator, feature contract and classification threshold. The metrics file contains separate validation and test results for all three models together with the installed scikit-learn and XGBoost versions.

See [`docs/baseline_models.md`](docs/baseline_models.md) for the preprocessing strategy, model configurations, class-imbalance handling, evaluation metrics and current limitations.

### Build the Analytical Database

Build the DuckDB analytical database after running the batch pipeline:

```bash
fraud-lakehouse build-analytics \
  --silver-dir data/silver \
  --gold-dir data/gold \
  --database-path data/analytics/fraud_lakehouse.duckdb
```

The database exposes persistent SQL views over the Silver and Gold Parquet datasets. See [`docs/duckdb_analytics.md`](docs/duckdb_analytics.md) for the view definitions and query examples.

### Run the dbt Models

Run the dbt analytical models and data tests:

```bash
dbt build \
  --project-dir dbt \
  --profiles-dir dbt
```

dbt creates staging views in the `analytics_staging` schema and analytical marts in the `analytics_marts` schema. See [`docs/dbt_analytics.md`](docs/dbt_analytics.md) for the model structure, schemas and test coverage.

### Run the Automated Checks

Run the complete Python formatting, linting and test suite:

```bash
ruff format --check .
ruff check .
python -m pytest
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
