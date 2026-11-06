#!/usr/bin/env bash

set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-fraud-lakehouse}"
KUBE_CONTEXT="${KUBE_CONTEXT:-kind-${KIND_CLUSTER_NAME}}"
KUBERNETES_NAMESPACE="${KUBERNETES_NAMESPACE:-fraud-lakehouse}"
SCORING_IMAGE="${SCORING_IMAGE:-fraud-lakehouse-scoring:local}"

SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
REPOSITORY_ROOT="$(
  cd "${SCRIPT_DIRECTORY}/.."
  pwd
)"

for command in docker kind kubectl; do
  if ! command -v "${command}" > /dev/null 2>&1; then
    echo "Required command is unavailable: ${command}" >&2
    exit 1
  fi
done

if ! kind get clusters | grep -Fxq "${KIND_CLUSTER_NAME}"; then
  echo "kind cluster does not exist: ${KIND_CLUSTER_NAME}" >&2
  exit 1
fi

if ! docker image inspect "${SCORING_IMAGE}" > /dev/null 2>&1; then
  echo "Scoring image does not exist locally: ${SCORING_IMAGE}" >&2
  echo "Build it with Docker Compose before deploying." >&2
  exit 1
fi

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout status \
  deployment/mlflow \
  --timeout=300s

kind load docker-image \
  "${SCORING_IMAGE}" \
  --name "${KIND_CLUSTER_NAME}"

kubectl \
  --context "${KUBE_CONTEXT}" \
  apply \
  --kustomize "${REPOSITORY_ROOT}/kubernetes/scoring"

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  rollout status \
  deployment/scoring-api \
  --timeout=300s

kubectl \
  --context "${KUBE_CONTEXT}" \
  --namespace "${KUBERNETES_NAMESPACE}" \
  get \
  pods,service \
  --selector app.kubernetes.io/name=scoring-api