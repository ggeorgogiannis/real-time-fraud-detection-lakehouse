# Airflow Orchestration

## Purpose

Apache Airflow schedules, executes and monitors the local fraud lakehouse batch workflow. The Airflow environment uses the same project package and local data directories as the existing command-line and Docker workflows.

The orchestration layer remains local-first. It does not require a cloud account, managed Airflow service or external storage.

## Runtime Architecture

The Airflow stack is defined in `compose.airflow.yaml`.

| Service | Responsibility |
| --- | --- |
| PostgreSQL | Store Airflow metadata, task states and DAG-run history |
| API server | Provide the Airflow web interface and execution API |
| Scheduler | Create scheduled DAG runs and submit eligible tasks |
| DAG processor | Discover, parse and serialize DAG definitions |
| Airflow init | Apply database migrations and create the local administrator |
| LocalExecutor | Execute task processes inside the scheduler container |

LocalExecutor is used because the current project runs on one development machine. This avoids adding Redis, Celery workers and other distributed components before the workload requires them.

Task parallelism is limited to four processes to keep local resource consumption predictable.

## Airflow Image

`Dockerfile.airflow` extends the official Apache Airflow Python 3.11 image. It installs:

- The operating-system runtime required by XGBoost.
- The project runtime dependencies from `pyproject.toml`.
- The `fraud-lakehouse` package and command-line interface.

The image uses the standard non-root Airflow user. Docker Compose runs Airflow with the host user identifier configured through `AIRFLOW_UID` so bind-mounted logs and data remain accessible from the host.

## Local Environment

Copy the tracked environment template:

```bash
cp .env.example .env
```

Set `AIRFLOW_UID` to the current Linux or WSL user identifier:

```bash
id -u
```

For example:

```dotenv
AIRFLOW_UID=1000
```

Replace the example administrator password and JWT secret in `.env` before starting Airflow.

A random JWT secret can be generated with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The local `.env` file is excluded from Git. `.env.example` contains only development placeholders and is safe to commit.

## Data and Log Mounts

Docker Compose mounts the following local directories into the Airflow containers:

| Host path | Container path | Access | Purpose |
| --- | --- | --- | --- |
| `airflow/dags/` | `/opt/airflow/dags` | Read-only | Provide DAG definitions |
| `airflow/logs/` | `/opt/airflow/logs` | Read/write | Persist task and scheduler logs |
| `data/` | `/opt/airflow/data` | Read/write | Share lakehouse inputs and outputs |

The generated Airflow logs and lakehouse datasets remain excluded from Git.

## Build the Airflow Image

Build the Airflow services:

```bash
docker compose \
  --file compose.airflow.yaml \
  build
```

## Initialize Airflow

Run the initialization service before starting Airflow for the first time:

```bash
docker compose \
  --file compose.airflow.yaml \
  up airflow-init
```

The initialization service:

- Waits for PostgreSQL to become healthy.
- Applies Airflow metadata database migrations.
- Creates the local administrator configured in `.env`.
- Exits after initialization completes.

## Start Airflow

Start the API server, scheduler and DAG processor:

```bash
docker compose \
  --file compose.airflow.yaml \
  up --detach \
  airflow-api-server \
  airflow-scheduler \
  airflow-dag-processor
```

Check the container states:

```bash
docker compose \
  --file compose.airflow.yaml \
  ps
```

The Airflow interface is available at:

```text
http://localhost:8080
```

Sign in with the administrator username and password stored in the local `.env` file.

## Daily Lakehouse DAG

The DAG is defined in:

```text
airflow/dags/fraud_lakehouse_daily.py
```

Its identifier is:

```text
fraud_lakehouse_daily
```

The DAG runs daily at `02:00 UTC` and contains two ordered tasks:

1. `run_batch_pipeline`
2. `build_analytics_database`

The first task executes Bronze ingestion, Silver validation and Gold transformations. The second task creates the DuckDB analytical database after the batch pipeline succeeds.

The DAG uses:

- A start date of `2026-01-01`.
- `catchup=False`.
- A maximum of one active DAG run.
- Two task retries.
- A five-minute retry delay.
- Explicit execution timeouts.

Scheduled runs use the Airflow data-interval end as the reproducible ingestion timestamp. Manual runs without a data interval fall back to the DAG run's `run_after` timestamp.

## Enable the DAG

New DAGs may initially be paused. Enable the lakehouse DAG with:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow dags unpause \
  fraud_lakehouse_daily
```

The DAG can also be enabled from the Airflow web interface.

## Trigger a Manual Run

Trigger the DAG from the command line:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow dags trigger \
  fraud_lakehouse_daily
```

A manual run does not require a logical date. The DAG uses the run's `run_after` timestamp when no data interval is available.

List recent runs:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow dags list-runs \
  fraud_lakehouse_daily \
  --output table
```

## Validate the Airflow Configuration

Validate the resolved Compose configuration:

```bash
docker compose \
  --file compose.airflow.yaml \
  config > /dev/null
```

Confirm the installed Airflow version:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow version
```

Check for DAG import errors:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow dags list-import-errors
```

List the DAG tasks:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow tasks list \
  fraud_lakehouse_daily
```

## Run a Controlled DAG Test

A full executor-backed DAG test can be run with an explicit logical date:

```bash
docker compose \
  --file compose.airflow.yaml \
  exec airflow-scheduler \
  airflow dags test \
  fraud_lakehouse_daily \
  2026-09-13T02:00:00+00:00 \
  --use-executor
```

This command executes both tasks against the mounted local data. It should therefore be treated as a real pipeline execution rather than a parse-only test.

## Verify the Analytical Database

After a successful DAG run, query the generated DuckDB database from the host:

```bash
python -c "import duckdb; con = duckdb.connect('data/analytics/fraud_lakehouse.duckdb', read_only=True); print(con.execute('SHOW TABLES').fetchall()); con.close()"
```

The analytical database exposes these views:

- `silver_transactions`
- `gold_transaction_features`
- `gold_customer_daily_summary`
- `gold_terminal_daily_summary`

The views store relative Parquet paths so the database remains queryable from both the host and the Airflow containers.

## Monitoring

Inspect recent service logs:

```bash
docker compose \
  --file compose.airflow.yaml \
  logs \
  --tail 100 \
  airflow-api-server \
  airflow-scheduler \
  airflow-dag-processor
```

Follow scheduler logs:

```bash
docker compose \
  --file compose.airflow.yaml \
  logs --follow \
  airflow-scheduler
```

Stop following logs with `Ctrl+C`.

Task status, retries, execution duration and logs are also available through the Airflow web interface.

## Execution API

Airflow 3 task processes communicate with the API server through the execution API. The internal service URL is configured as:

```text
http://airflow-api-server:8080/execution/
```

All Airflow runtime services must be able to resolve and connect to `airflow-api-server` through the Compose network.

## Stop Airflow

Stop the running Airflow services without removing their containers:

```bash
docker compose \
  --file compose.airflow.yaml \
  stop
```

Stop and remove the containers and network:

```bash
docker compose \
  --file compose.airflow.yaml \
  down
```

The PostgreSQL metadata volume is retained by the standard `down` command.

To remove the metadata volume as well:

```bash
docker compose \
  --file compose.airflow.yaml \
  down --volumes
```

The final command permanently removes the local Airflow metadata database, including users, DAG-run history and task states.

## Local Development Scope

This Airflow configuration is intended for local development, portfolio demonstrations and reproducible testing.

It is not a production deployment. A production environment would require additional security, secret management, monitoring, backup, scaling and availability controls.
