# MLOps and Local Kubernetes Deployment

## Purpose

Phase 6 turns the approved fraud policy into a locally deployable and observable service.

The implementation provides:

- An MLflow tracking server and model registry backed by PostgreSQL.
- Registration of the approved fraud policy and its evaluation evidence.
- A stable `champion` model alias for deployment.
- A non-root FastAPI scoring service.
- Prometheus metrics and a provisioned Grafana dashboard.
- Docker Compose execution for local development.
- Deployment to a local Kubernetes cluster created with kind.
- Persistent storage and restart-safe recovery.
- Automated unit, container, configuration and manifest validation.

The entire platform runs locally without paid cloud services.

## Runtime Architecture

The approved policy consists of:

- The fitted preprocessing and model artifact.
- The selected decision threshold.
- The daily customer-card alert capacity.
- Validation-period evaluation results.
- Locked test-period evaluation results.

The registration workflow packages these components as one MLflow PyFunc model and assigns the approved version the `champion` alias.

```text
Approved model artifact
        |
        v
Threshold and evaluation validation
        |
        v
MLflow registration run
        |
        v
Registered model version
        |
        v
champion alias
        |
        v
FastAPI scoring service
        |
        +----> Prometheus metrics
                    |
                    v
                 Grafana
```

PostgreSQL stores MLflow tracking and registry metadata. MLflow artifacts, PostgreSQL data, Prometheus time series and Grafana state use persistent volumes.

## Approved Fraud Policy

The approved policy loader validates that the threshold policy, final evaluation and model artifact describe the same selected model and operating policy.

The current approved policy uses:

- Model: `xgboost`
- Artifact: `tuned_xgboost.joblib`
- Decision threshold: `0.6154609918594361`
- Daily capacity: `100`
- Capacity scope: `unique_customer_cards_per_day`

The loader rejects inconsistent, malformed or unsafe artifact references before deserializing the model.

The deployed model returns:

- `fraud_probability`
- `threshold_decision`

The scoring endpoint applies the approved transaction-level threshold. It does not enforce the daily customer-card capacity across requests. Capacity enforcement requires shared operational state and downstream ranking across all candidate transactions for a given day.

The API therefore reports:

```json
{
  "capacity_enforced_by_endpoint": false
}
```

## MLflow Registration

The registration job validates and logs:

- The approved model artifact.
- The threshold policy.
- The final evaluation report.
- Model and capacity metadata.
- Validation and test metrics.
- The source Git revision.
- An input example and model signature.

The registered model is named `fraud-detection-policy` by default. The approved version receives the `champion` alias.

Consumers load the immutable version resolved from that alias rather than loading arbitrary files from `data/models`.

## Scoring API

The scoring service is implemented with FastAPI and Uvicorn.

Endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health/live` | `GET` | Confirm that the process is alive |
| `/health/ready` | `GET` | Confirm that the approved model is loaded |
| `/v1/score` | `POST` | Score one or more transactions |
| `/metrics` | `GET` | Expose Prometheus metrics |

The service validates request structure, batch size and model feature types before inference.

It loads one resolved model version during application startup. Readiness is reported only after MLflow metadata, approval tags and the model artifact have been validated.

## Scoring Request

Example request:

```bash
curl \
  --fail-with-body \
  --silent \
  --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "transactions": [
      {
        "transaction_id": 1000001,
        "tx_datetime": "2018-04-01T08:00:00Z",
        "customer_id": 1001,
        "terminal_id": 2001,
        "tx_amount": 800.0,
        "transaction_hour": 8,
        "day_of_week": 6,
        "is_weekend": true,
        "is_night": false,
        "customer_previous_transaction_count": 5,
        "customer_previous_mean_amount": 100.0,
        "amount_to_customer_previous_mean": 8.0,
        "terminal_previous_transaction_count": 20
      }
    ]
  }' \
  http://127.0.0.1:8000/v1/score \
  | python -m json.tool
```

The response contains the immutable model version, alias, source revision, approved threshold, capacity metadata and one prediction per transaction.

## Prometheus Metrics

The scoring service publishes:

- Request counts by endpoint, method and status.
- Request-duration histograms.
- Number of scored transactions.
- Number of threshold-positive decisions.
- Fraud-probability histograms.
- Loaded-model identity information.

The Kubernetes deployment uses a headless metrics service and Prometheus DNS discovery. Each scoring replica is scraped independently, preventing counters from being mixed through a load-balanced service address.

## Grafana Dashboard

Grafana provisions the Prometheus datasource and the `Fraud Scoring Operations` dashboard automatically.

The dashboard includes:

- API availability.
- Loaded model identity.
- Scored transactions.
- Threshold-positive decisions.
- Scoring request rate.
- Scoring latency p95.
- Responses by HTTP status.
- Mean fraud probability.

Rate-based panels require recent scoring traffic. They may display no data when no scoring request has occurred within the query window.

## Local Environment

Copy the example configuration:

```bash
cp .env.mlops.example .env.mlops
```

Set local values for:

- `MLFLOW_POSTGRES_DB`
- `MLFLOW_POSTGRES_USER`
- `MLFLOW_POSTGRES_PASSWORD`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`
- `FRAUD_REGISTERED_MODEL_NAME`
- `FRAUD_MODEL_ALIAS`
- `FRAUD_SCORING_MAX_BATCH_SIZE`

The `.env.mlops` file is excluded from Git. Do not commit passwords or generated Kubernetes Secrets.

## Build the MLOps Images

```bash
docker compose \
  --env-file .env.mlops \
  --file compose.mlops.yaml \
  build \
  mlflow \
  scoring-api
```

Both application images run as non-root users.

## Start MLflow with Docker Compose

Start the tracking server and PostgreSQL:

```bash
docker compose \
  --env-file .env.mlops \
  --file compose.mlops.yaml \
  up \
  --detach \
  --wait \
  mlflow
```

MLflow is available at:

```text
http://127.0.0.1:5000
```

## Register the Approved Policy

The Phase 3 model and evaluation artifacts must exist under `data/models`, and the input example must exist under `data/ml`.

```bash
python -m fraud_lakehouse.mlflow_registry_job \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment-name fraud-policy-registration \
  --registered-model-name fraud-detection-policy \
  --alias champion \
  --model-dir data/models \
  --policy-path data/models/threshold_policy.json \
  --evaluation-path data/models/final_evaluation.json \
  --input-example-path data/ml/test.parquet \
  --input-example-rows 5 \
  --source-revision "$(git rev-parse HEAD)"
```

The scoring service will not become ready until the configured model and alias exist.

## Start the Complete Compose Stack

After registration:

```bash
docker compose \
  --env-file .env.mlops \
  --file compose.mlops.yaml \
  up \
  --detach \
  --wait \
  scoring-api \
  prometheus \
  grafana
```

Local endpoints:

| Service | Address |
| --- | --- |
| MLflow | `http://127.0.0.1:5000` |
| Scoring API | `http://127.0.0.1:8000` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

## Stop the Compose Stack

Stop containers while preserving persistent volumes:

```bash
docker compose \
  --env-file .env.mlops \
  --file compose.mlops.yaml \
  down
```

Do not add `--volumes` unless the MLflow database, registered models, artifacts, Prometheus history and Grafana state should be permanently deleted.

## Create the Kubernetes Cluster

The local Kubernetes deployment uses kind.

Required commands:

- `docker`
- `kubectl`
- `kind`

Create the cluster:

```bash
./kubernetes/create-cluster.sh
```

The cluster name is `fraud-lakehouse`, and the kubectl context is `kind-fraud-lakehouse`.

The single control-plane node maps local ports to Kubernetes NodePorts.

## Build Images for Kubernetes

Build the same images used by Docker Compose:

```bash
docker compose \
  --env-file .env.mlops \
  --file compose.mlops.yaml \
  build \
  mlflow \
  scoring-api
```

The deployment scripts load these local images into the kind node.

## Deploy MLflow to Kubernetes

```bash
./kubernetes/deploy-mlflow.sh
```

The script:

1. Loads the MLflow image into kind.
2. Creates the namespace.
3. Generates the PostgreSQL Secret from `.env.mlops`.
4. Applies the MLflow and PostgreSQL manifests.
5. Waits for both workloads to become ready.

Kubernetes MLflow is available at:

```text
http://127.0.0.1:15000
```

## Register the Policy in Kubernetes MLflow

A new kind cluster has an empty MLflow registry. Register the approved policy against the Kubernetes endpoint:

```bash
python -m fraud_lakehouse.mlflow_registry_job \
  --tracking-uri http://127.0.0.1:15000 \
  --experiment-name fraud-policy-registration \
  --registered-model-name fraud-detection-policy \
  --alias champion \
  --model-dir data/models \
  --policy-path data/models/threshold_policy.json \
  --evaluation-path data/models/final_evaluation.json \
  --input-example-path data/ml/test.parquet \
  --input-example-rows 5 \
  --source-revision "$(git rev-parse HEAD)"
```

## Deploy Scoring to Kubernetes

```bash
./kubernetes/deploy-scoring.sh
```

The deployment runs two scoring replicas with:

- Rolling updates.
- Startup, readiness and liveness probes.
- Explicit resource requests and limits.
- Non-root execution.
- Linux capabilities removed.
- Service-account token mounting disabled.

The Kubernetes scoring endpoint is:

```text
http://127.0.0.1:18000
```

## Deploy Observability to Kubernetes

```bash
./kubernetes/deploy-observability.sh
```

The script generates ConfigMaps from the canonical monitoring files, generates the Grafana administrator Secret from `.env.mlops`, applies the manifests and waits for both deployments.

Kubernetes endpoints:

| Service | Address |
| --- | --- |
| Prometheus | `http://127.0.0.1:19090` |
| Grafana | `http://127.0.0.1:13000` |

The Grafana credentials are the values configured in `.env.mlops`.

## Inspect Kubernetes Resources

```bash
kubectl \
  --context kind-fraud-lakehouse \
  --namespace fraud-lakehouse \
  get \
  deployments,statefulsets,pods,services,persistentvolumeclaims
```

Expected workloads:

- One MLflow replica.
- One PostgreSQL replica.
- Two scoring replicas.
- One Prometheus replica.
- One Grafana replica.

## Restart Recovery

Persistent volumes retain MLflow, PostgreSQL, Prometheus and Grafana state across pod restarts.

Example:

```bash
kubectl \
  --context kind-fraud-lakehouse \
  --namespace fraud-lakehouse \
  rollout restart \
  deployment/mlflow \
  deployment/scoring-api \
  deployment/prometheus \
  deployment/grafana
```

Wait for recovery:

```bash
for deployment in \
  mlflow \
  scoring-api \
  prometheus \
  grafana
do
  kubectl \
    --context kind-fraud-lakehouse \
    --namespace fraud-lakehouse \
    rollout status \
    "deployment/${deployment}" \
    --timeout=300s
done
```

## Delete the Kubernetes Cluster

Deleting the kind cluster permanently removes its workloads and local persistent volumes:

```bash
kind delete cluster \
  --name fraud-lakehouse
```

Only run this command when the Kubernetes MLflow registry, artifacts, monitoring history and Grafana state are no longer required.

## Automated Validation

Continuous integration validates:

- Python formatting and linting.
- Unit and integration tests.
- Approved-policy validation.
- MLflow registration behavior.
- Scoring API behavior.
- MLOps Compose configuration.
- MLflow and scoring image builds.
- Non-root container execution.
- Prometheus configuration.
- Kubernetes Kustomize rendering.
- Kubernetes deployment-script syntax.

The complete kind deployment remains a local integration test because trained model artifacts and datasets are intentionally excluded from Git.

## Local-Development Limitations

This deployment is intended for reproducible local development and portfolio demonstration.

Current limitations:

- kind uses one local control-plane node.
- Persistent volumes remain on that local cluster.
- MLflow and the scoring endpoint do not use TLS.
- MLflow, Prometheus and the scoring endpoint do not provide production authentication.
- Grafana uses local administrator credentials.
- The scoring API expects precomputed model features.
- The API does not perform online feature retrieval.
- The API applies the threshold but does not enforce daily alert capacity.
- There is no distributed tracing or external alert manager.
- There is no cloud object storage or managed database.
- Disaster recovery is limited to locally reproducible artifacts and source data.

A production deployment would require external secret management, TLS, authenticated endpoints, durable remote storage, backups, network policies, autoscaling, online feature computation and operational alert-capacity enforcement.