#!/usr/bin/env bash

set -euo pipefail

readonly CLUSTER_NAME="${KIND_CLUSTER_NAME:-fraud-lakehouse}"
readonly KUBE_CONTEXT="kind-${CLUSTER_NAME}"
readonly NAMESPACE="fraud-lakehouse"
readonly MLFLOW_IMAGE="fraud-lakehouse-mlflow:local"
readonly SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
readonly REPOSITORY_ROOT="$(
  cd "${SCRIPT_DIRECTORY}/.."
  pwd
)"
readonly ENVIRONMENT_FILE="${1:-${REPOSITORY_ROOT}/.env.mlops}"
readonly MANIFEST_DIRECTORY="${SCRIPT_DIRECTORY}/mlflow"

if [[ ! -f "${ENVIRONMENT_FILE}" ]]
then
  echo "Missing environment file: ${ENVIRONMENT_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENVIRONMENT_FILE}"
set +a

for variable_name in \
  MLFLOW_POSTGRES_DB \
  MLFLOW_POSTGRES_USER \
  MLFLOW_POSTGRES_PASSWORD
do
  if [[ -z "${!variable_name:-}" ]]
  then
    echo "Missing required variable: ${variable_name}" >&2
    exit 1
  fi
done

kubectl cluster-info \
  --context "${KUBE_CONTEXT}" \
  > /dev/null

kind load docker-image \
  "${MLFLOW_IMAGE}" \
  --name "${CLUSTER_NAME}"

kubectl \
  --context "${KUBE_CONTEXT}" \
  apply \
  --filename "${MANIFEST_DIRECTORY}/namespace.yaml"

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  create secret generic mlflow-postgres \
  --from-literal="database=${MLFLOW_POSTGRES_DB}" \
  --from-literal="username=${MLFLOW_POSTGRES_USER}" \
  --from-literal="password=${MLFLOW_POSTGRES_PASSWORD}" \
  --dry-run=client \
  --output yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  apply \
  --kustomize "${MANIFEST_DIRECTORY}"

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  rollout status \
  statefulset/mlflow-postgres \
  --timeout=180s

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  rollout status \
  deployment/mlflow \
  --timeout=300s

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  get \
  pods,services,persistentvolumeclaims