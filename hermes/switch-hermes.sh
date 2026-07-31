#!/usr/bin/env bash
# switch-hermes.sh — install the Inkling Hermes profile on the head node.
#
# Usage (from your orchestrator machine):
#   bash hermes/switch-hermes.sh inkling            # install the bundled Inkling profile
#   bash hermes/switch-hermes.sh /path/to/other.yaml # install any other profile
#
# What it does: preflights that the Inkling server (:8001) and the de-streaming
# proxy (:8002) are up, backs up the live ~/.hermes/config.yaml (timestamped),
# installs the profile, restarts hermes-gateway, and verifies with a one-shot.
#
# Env overrides: HEAD (default spark1.local), CLUSTER_USER (default nvidia),
#                HERMES_SSH (default = HEAD — the host Hermes runs on).
set -euo pipefail

ARG="${1:?usage: switch-hermes.sh <inkling|/path/to/config.yaml>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HEAD="${HERMES_SSH:-${HEAD:-spark1.local}}"
CLUSTER_USER="${CLUSTER_USER:-nvidia}"
PROXY_PORT=8002
MODEL_PORT=8001

case "$ARG" in
  inkling) SRC="$HERE/config.yaml" ;;
  *)       SRC="$ARG" ;;
esac
[ -f "$SRC" ] || { echo "FAIL: profile not found: $SRC" >&2; exit 1; }

echo "== preflight: Inkling serving on :$MODEL_PORT"
ssh "$HEAD" "curl -fsS --max-time 5 http://127.0.0.1:$MODEL_PORT/health >/dev/null" \
  || { echo "FAIL: nothing healthy on 127.0.0.1:$MODEL_PORT — start the cluster first" >&2; exit 1; }

echo "== ensuring inkling-destream-proxy is active on $HEAD"
ssh "$CLUSTER_USER@$HEAD" 'systemctl --user reset-failed inkling-destream-proxy 2>/dev/null; systemctl --user start inkling-destream-proxy; systemctl --user is-active inkling-destream-proxy' \
  || { echo "FAIL: de-streaming proxy not active — install it first (see hermes/README.md)" >&2; exit 1; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
echo "== backing up live config on $HEAD -> ~/.hermes/config.yaml.bak-switch-$TS"
ssh "$HEAD" "cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-switch-$TS 2>/dev/null || true"

echo "== installing profile ($SRC)"
scp -q "$SRC" "$HEAD:~/.hermes/config.yaml"
ssh "$HEAD" 'chmod 600 ~/.hermes/config.yaml'

echo "== restarting hermes-gateway"
ssh "$HEAD" 'systemctl --user reset-failed hermes-gateway 2>/dev/null; systemctl --user restart hermes-gateway; sleep 6; systemctl --user is-active hermes-gateway'

echo "== verify (one-shot)"
ssh "$HEAD" '~/.local/bin/hermes -z "Reply with exactly: pong"' | tail -2

echo "ok: hermes switched using $SRC (backup of previous: ~/.hermes/config.yaml.bak-switch-$TS)"
