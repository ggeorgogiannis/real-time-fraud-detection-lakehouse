# Local Data Directory

This directory contains datasets and generated pipeline outputs used during local development.

Only this README is tracked by Git. Transaction data, Parquet outputs and other generated artifacts remain on the local machine.

## Directory Structure

| Directory    | Contents                                                            |
| ------------ | ------------------------------------------------------------------- |
| `raw`        | Daily transaction files downloaded from the official dataset source |
| `bronze`     | Ingested transactions with source and ingestion metadata            |
| `silver`     | Validated, standardized and deduplicated transactions               |
| `gold`       | Fraud features, analytical tables and model-ready datasets          |
| `quarantine` | Records rejected by Silver-layer validation                         |

## Data Source

The project follows the simulator and methodology documented in the [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook).

The source transaction files are available from the [simulated-data-raw repository](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).

Download instructions and dataset verification steps will be added with the first ingestion implementation.

## Version-Control Policy

The data directories are excluded from Git because they contain downloaded or reproducible artifacts.

Small synthetic datasets used by automated tests are stored separately under `tests/fixtures`.
