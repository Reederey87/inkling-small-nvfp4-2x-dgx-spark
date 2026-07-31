#!/usr/bin/env bash
# Install the Inkling kit and systemd user units on both nodes.
# Units are installed but NOT enabled and NOT started — this is a window-only service.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

install_node() {
  local host="$1" service="$2"
  rsync -a --exclude='.env.inkling' --exclude='.git' --exclude='bench/results' "$KIT/" "$CLUSTER_USER@$host:$KIT_DIR/" \
    || fail "rsync to $host failed" "check Mac SSH access and $KIT_DIR permissions"
  # Top-level kit scripts must stay executable on-node (preflight/render-env are invoked via
  # the systemd units' flat $KIT_DIR/<script> paths); bench/ too (manual gates).
  ssh "$CLUSTER_USER@$host" "chmod +x '$KIT_DIR'/*.sh '$KIT_DIR'/bench/*.sh 2>/dev/null || chmod +x '$KIT_DIR'/*.sh"
  echo "ok: kit synced to $host"

  ssh "$CLUSTER_USER@$host" "mkdir -p ~/.config/systemd/user && cp '$KIT_DIR/systemd/$service' ~/.config/systemd/user/ && systemctl --user daemon-reload" \
    || fail "service install failed on $host" "check systemd user manager for nvidia"
  echo "ok: $service installed on $host"

  linger="$(ssh "$CLUSTER_USER@$host" 'loginctl show-user "$CLUSTER_USER" --property=Linger')" \
    || fail "could not inspect linger on $host" "check the node"
  [ "$linger" = "Linger=yes" ] || fail "linger is not enabled on $host" "fix before boot"
  echo "ok: linger enabled on $host"
}

install_node "$HEAD_HOST" "vllm-inkling-head.service"
install_node "$WORKER_HOST" "vllm-inkling-worker.service"
echo "ok: services installed; units were not enabled or started (window-only — see docs/BRINGUP-RUNBOOK.md)"
