# Containerized Execution

## Purpose

The application container provides a reproducible Python 3.11 runtime for the fraud lakehouse command-line interface. It allows the batch pipeline to run without depending on the host Python environment or locally installed project dependencies.

The image includes the complete `fraud-lakehouse` CLI. Docker Compose provides the standard local interface for building the image, mounting transaction data and running the pipeline.

## Image Design

The image:

- Uses the official Python 3.11 slim image.
- Installs the OpenMP runtime required by XGBoost.
- Installs the project from `pyproject.toml`.
- Exposes `fraud-lakehouse` as its entry point.
- Runs as a non-root user.
- Excludes local data, tests, development environments and generated artifacts from the build context.

The default container user has UID and GID `1000`, matching the default user created by most WSL and Linux installations.

## Data Mounts

Docker Compose mounts the local data directory into the container.

| Host path  | Container path  | Access     | Purpose                                      |
| ---------- | --------------- | ---------- | -------------------------------------------- |
| `data/`    | `/app/data`     | Read/write | Persist generated lakehouse outputs locally |
| `data/raw` | `/app/data/raw` | Read-only  | Protect downloaded source transaction files |

The nested read-only mount prevents the container from modifying the source transaction files. Bronze, Silver, Gold and quarantine outputs remain on the host and continue to be excluded from Git.

## Build the Image

Build the application image:

```bash
docker compose build batch
```

On Linux systems where the local user does not have UID and GID `1000`, build with the host identifiers:

```bash
APP_UID="$(id -u)" APP_GID="$(id -g)" docker compose build batch
```

## Run the Batch Pipeline

Place trusted daily transaction files in `data/raw`. Their filenames must follow the `YYYY-MM-DD.pkl` contract.

Run the Bronze, Silver and Gold pipeline:

```bash
docker compose run --rm batch
```

The container writes the generated datasets to the existing directories under `data/`. The `--rm` option removes the one-off container after the pipeline finishes.

The batch pipeline remains idempotent when run through Docker. Previously ingested source files are not duplicated, and outputs are published through the same safe replacement process used by direct local execution.

## Run Other CLI Commands

The service command can be replaced to access any command included in the application image:

```bash
docker compose run --rm batch --help
```

For example, build the DuckDB analytical database from the containerized CLI:

```bash
docker compose run --rm batch \
  build-analytics \
  --silver-dir /app/data/silver \
  --gold-dir /app/data/gold \
  --database-path /app/data/analytics/fraud_lakehouse.duckdb
```

## Stop Compose Resources

One-off `docker compose run --rm` executions do not leave application containers running. Remove any Compose-created network resources with:

```bash
docker compose down
```

## Automated Validation

Continuous integration performs the following container checks on every pull request and push to `main`:

- Validate the resolved Compose configuration.
- Build the application image.
- Start the packaged CLI.
- Verify that the container does not run as root.

The Python formatting, linting and automated test suite continues to run in a separate CI job.