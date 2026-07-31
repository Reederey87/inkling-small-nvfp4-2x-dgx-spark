#!/usr/bin/env bash
# Inkling smoke/correctness gate (G1+G2). Runs from the Mac; executes smoke.py on spark1
# against the loopback API. Usage: bash bench/smoke.sh
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "python3 - --url http://127.0.0.1:$API_PORT --model $SERVED_MODEL_NAME" \
  < "$KIT/bench/smoke.py"
