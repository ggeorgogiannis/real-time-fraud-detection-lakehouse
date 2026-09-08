# Real-Time Fraud Detection Lakehouse

A local-first data engineering project for processing transaction data and detecting fraud.

The project begins as a tested batch pipeline and will gradually evolve into a streaming lakehouse. Each new tool will be introduced only when the existing implementation creates a clear reason to use it.

## Current Status

Phase 1 is complete. The first executable local batch pipeline was published as [v0.1.0](https://github.com/ggeorgogiannis/real-time-fraud-detection-lakehouse/releases/tag/v0.1.0).

The pipeline can:

* Discover and ingest trusted daily transaction files.
* Preserve source records and ingestion metadata in Bronze.
* Validate, type-convert, deduplicate and quarantine records in Silver.
* Create historical transaction features and daily customer and terminal summaries in Gold.
* Publish Parquet outputs safely and reproducibly.
* Run through a documented command-line interface.
* Verify its behaviour through automated unit and integration tests.

Phase 2 is now underway. DuckDB integration, the initial dbt project, staging models, the daily fraud mart and automated dbt data tests are implemented. The next milestone is expanding the analytical marts and reporting models.


## Why This Project

Fraud detection is often presented only as a classification problem. In practice, the model depends on a larger data system that must handle ingestion, validation, historical features, reproducibility and monitoring.

This project focuses on that complete workflow. Its purpose is to explore how the components fit together and document the engineering decisions made during development.

## Current Architecture

`Daily transaction files -> Bronze -> Silver -> Gold -> DuckDB -> dbt -> Fraud analysis and model datasets`

| Component  | Responsibility                                                          |
| ---------- | ----------------------------------------------------------------------- |
| Bronze     | Store ingested transactions with minimal changes and ingestion metadata |
| Silver     | Apply schema checks, type conversions, deduplication and quality rules  |
| Gold       | Create fraud features, customer summaries and analytical tables         |
| Quarantine | Preserve rejected records together with their validation failures       |
| DuckDB     | Expose Silver and Gold Parquet datasets through persistent SQL views    |
| dbt        | Manage tested SQL transformations, staging models and analytical marts  |

The completed batch pipeline provides a reliable foundation for the DuckDB and dbt analytical layer. It will also serve as the reference implementation for the later Kafka and Spark streaming pipeline.

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

Create time-aware features, train baseline models and evaluate them using metrics appropriate for imbalanced data.

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
- [ ] Create leakage-safe model datasets.
- [ ] Train and evaluate baseline fraud models.

## Running the Project

The batch pipeline requires Python 3.11.

Create and activate the project environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -e ".[dev]"
```

Place daily transaction files in `data/raw`. Input files must use the `YYYY-MM-DD.pkl` naming convention.

Run the complete pipeline:

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

An explicit ingestion timestamp can be supplied when a reproducible test run is required:

```bash
fraud-lakehouse run \
  --raw-dir data/raw \
  --output-dir data \
  --ingested-at-utc "2026-09-08T12:00:00Z"
```

Run the automated checks with:

```bash
ruff format --check .
ruff check .
python -m pytest
```

Build the DuckDB analytical database after running the batch pipeline:

```bash
fraud-lakehouse build-analytics \
  --silver-dir data/silver \
  --gold-dir data/gold \
  --database-path data/analytics/fraud_lakehouse.duckdb
```

Run the dbt analytical models and data tests:

```bash
dbt build \
  --project-dir dbt \
  --profiles-dir dbt
```

dbt creates staging views in the `analytics_staging` schema and analytical marts in the `analytics_marts` schema. See [`docs/dbt_analytics.md`](docs/dbt_analytics.md) for the model structure, schemas and test coverage.

The database exposes persistent SQL views over the Silver and Gold Parquet datasets. See [`docs/duckdb_analytics.md`](docs/duckdb_analytics.md) for the view definitions and query examples.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
