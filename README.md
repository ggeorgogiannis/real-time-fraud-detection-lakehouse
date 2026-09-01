# Real-Time Fraud Detection Lakehouse

A local-first data engineering project for processing transaction data and detecting fraud.

The project starts with a tested batch pipeline and will gradually evolve into a streaming lakehouse. Each tool will be introduced when the existing implementation provides a clear reason to use it.

## Current Status

Work is currently focused on the project foundation and the first batch-processing milestone.

The first milestone is a local pipeline that can:

* Discover daily transaction files.
* Ingest transactions without creating duplicate records.
* Preserve source data in a Bronze layer.
* Validate and quarantine records in a Silver layer.
* Create fraud features and analytical tables in a Gold layer.
* Produce reproducible outputs from a documented command.
* Verify pipeline behaviour through automated tests.

## Why This Project

Fraud detection is often presented only as a classification problem. In practice, the model depends on a larger data system that must handle ingestion, validation, historical features, reproducibility and monitoring.

This project focuses on that complete workflow. Its purpose is to explore how the components fit together and document the engineering decisions made during development.

## Initial Architecture

`Daily transaction files -> Bronze -> Silver -> Gold -> Fraud analysis and model datasets`

| Layer      | Responsibility                                                              |
| ---------- | --------------------------------------------------------------------------- |
| Bronze     | Store ingested transactions with minimal changes and ingestion metadata     |
| Silver     | Apply schema checks, type conversions, deduplication and data-quality rules |
| Gold       | Create fraud features, customer summaries and analytical tables             |
| Quarantine | Preserve rejected records together with their validation failures           |

The batch implementation will be completed before streaming components are introduced. It will provide a reliable reference pipeline against which the later Kafka and Spark implementation can be tested.

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

The repository will include small synthetic fixtures for automated tests. These fixtures will cover legitimate transactions, fraud labels, duplicates and invalid records without requiring the complete dataset.

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

### Phase 2: Analytical Lakehouse

Introduce DuckDB and dbt Core for SQL transformations, data tests and analytical models.

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

## Running the Project

Setup and execution instructions will be added with the first working batch pipeline.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
