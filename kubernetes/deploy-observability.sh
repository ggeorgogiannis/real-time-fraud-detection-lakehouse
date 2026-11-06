#!/usr/bin/env bash

set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-fraud-lakehouse}"
KUBE_CONTEXT="${KUBE_CONTEXT:-kind-${KIND_CLUSTER_NAME}}"
KUBERNETES_NAMESPACE="${KUBERNETES_NAMESPACE:-fraud-lakehouse}"

SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
REPOSITORY_ROOT="$(
  cd "${SCRIPT_DIRECTORY}/.."
  pwd
)"
ENVIRONMENT_FILE="${REPOSITORY_ROOT}/.env.mlops"

for command in kind kubectl; do
  if ! command -v "${command}" > /dev/null 2>&1; then
    echo "Required command is unavailable: ${command}" >&2
    exit 1
  fi
done

if ! kind get clusters | grep -Fxq "${KIND_CLUSTER_NAME}"; then
  echo "kind cluster does not exist: ${KIND_CLUSTER_NAME}" >&2
  exit 1
fi

if [[ ! -f "${ENVIRONMENT_FILE}" ]]; then
  echo "Missing local environment file: ${ENVIRONMENT_FILE}" >&2
  exit 1
fi

read_environment_value() {
  local variable_name="$1"
  local matching_line

  matching_line="$(
    grep -E "^${variable_name}=" "${ENVIRONMENT_FILE}" \
      | tail -n 1 \
      || true
  )"

  if [[ -z "${matching_line}" ]]; then
    echo "Missing ${variable_name} in ${ENVIRONMENT_FILE}" >&2
    exit 1
  fi

  printf '%s' "${matching_line#*=}"
}

grafana_admin_user="$(
  read_environment_value GRAFANA_ADMIN_USER
)"
grafana_admin_password="$(
  read_environment_value GRAFANA_ADMIN_PASSWORD
)"

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout status \
  deployment/scoring-api \
  --timeout=300s

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  create secret generic grafana-admin \
  --from-literal=admin-user="${grafana_admin_user}" \
  --from-literal=admin-password="${grafana_admin_password}" \
  --dry-run=client \
  --output=yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  create configmap prometheus-config \
  --from-file=prometheus.yml="${REPOSITORY_ROOT}/monitoring/prometheus/prometheus.yml" \
  --dry-run=client \
  --output=yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  create configmap grafana-datasource \
  --from-file=prometheus.yml="${REPOSITORY_ROOT}/monitoring/grafana/provisioning/datasources/prometheus.yml" \
  --dry-run=client \
  --output=yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  create configmap grafana-dashboard-provider \
  --from-file=dashboards.yml="${REPOSITORY_ROOT}/monitoring/grafana/provisioning/dashboards/dashboards.yml" \
  --dry-run=client \
  --output=yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  create configmap grafana-dashboard \
  --from-file=fraud-scoring.json="${REPOSITORY_ROOT}/monitoring/grafana/dashboards/fraud-scoring.json" \
  --dry-run=client \
  --output=yaml \
  | kubectl \
      --context "${KUBE_CONTEXT}" \
      apply \
      --filename -

kubectl \
  --context "${KUBE_CONTEXT}" \
  apply \
  --kustomize "${REPOSITORY_ROOT}/kubernetes/observability"

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout restart \
  deployment/prometheus \
  deployment/grafana

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout status \
  deployment/prometheus \
  --timeout=300s

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout status \
  deployment/grafana \
  --timeout=300s