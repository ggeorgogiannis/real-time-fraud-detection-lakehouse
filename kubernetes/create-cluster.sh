#!/usr/bin/env bash

set -euo pipefail

readonly CLUSTER_NAME="fraud-lakehouse"
readonly NODE_IMAGE="kindest/node:v1.36.4@sha256:099e049362a1526b2db71494e1947aae99bd16290d7c895f2b7ea312e3cbfaed"
readonly SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
readonly CLUSTER_CONFIG="${SCRIPT_DIRECTORY}/kind-config.yaml"

if kind get clusters 2>/dev/null \
  | grep --fixed-strings --line-regexp "${CLUSTER_NAME}" > /dev/null
then
  echo "kind cluster already exists: ${CLUSTER_NAME}"
  exit 0
fi

kind create cluster \
  --config "${CLUSTER_CONFIG}" \
  --image "${NODE_IMAGE}" \
  --wait 180s

kubectl cluster-info \
  --context "kind-${CLUSTER_NAME}"