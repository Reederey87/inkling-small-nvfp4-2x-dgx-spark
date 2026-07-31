#!/usr/bin/env bash
# Role-aware pre-start guards for the vllm-inkling units. Runs as 'nvidia' on the
# node itself (ExecStartPre). Usage: preflight.sh <head|worker>
# Every wait is bounded; systemd TimeoutStartSec must exceed the worst-case sum.
#
# Adapted from 2xSPARK-CLUSTER/preflight.sh. Key differences:
#   - checks the INKLING image/weights/container/unit names;
#   - hard-fails if PROD (vllm-dsv4) is up — the two models cannot share the memory pool;
#   - never touches the prod container (the rm below targets vllm-inkling only).
set -euo pipefail

ROLE="${1:?usage: preflight.sh <head|worker>}"
# Boot-critical: don't rely on the systemd user manager's PATH.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

wait_for() { # wait_for <secs> <label> <cmd...>
  local deadline=$(( $(date +%s) + $1 )) label="$2"; shift 2
  until "$@" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "preflight FAIL: timed out waiting for: $label" >&2; exit 1
    fi
    sleep 5
  done
  echo "preflight ok: $label"
}

# Sysctl-parity guard (ported from prod; same hardware, same failure class): on this
# unified-memory platform the kernel low-watermark (~1.25x vm.min_free_kbytes) is reserved
# from CUDA-visible memory with no owning PID. Cross-node asymmetry crash-loops the worker.
# No-arg call: runtime vs persisted — WARN only. `peer` call (head-only): local vs peer —
# FAIL (this caused prod's 2026-07-18 outage).
MIN_FREE_KBYTES_LOCAL=""
check_min_free_parity() { # check_min_free_parity [peer]
  local conf="/etc/sysctl.d/90-dspark-oom.conf" runtime persisted peer_val
  runtime=$(sysctl -n vm.min_free_kbytes 2>/dev/null || true)
  if [ -z "$MIN_FREE_KBYTES_LOCAL" ] && printf '%s' "$runtime" | grep -qE '^[0-9]+$'; then
    MIN_FREE_KBYTES_LOCAL="$runtime"
  fi

  if [ "${1:-}" = "peer" ]; then
    # Absolute path: the remote non-interactive session doesn't inherit our PATH export.
    # grep-extract the digit line: MOTD/banner stdout pollution must not reach the
    # comparison — a polluted read lands on the WARN path, never a false FAIL.
    peer_val=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLUSTER_USER@$PEER_R1" /usr/sbin/sysctl -n vm.min_free_kbytes 2>/dev/null \
      | grep -E '^[0-9]+$' | tail -n1 || true)
    if ! printf '%s' "$MIN_FREE_KBYTES_LOCAL" | grep -qE '^[0-9]+$' || ! printf '%s' "$peer_val" | grep -qE '^[0-9]+$'; then
      echo "preflight WARN: could not read peer vm.min_free_kbytes — parity unverified" >&2
      return 0
    fi
    if [ "$MIN_FREE_KBYTES_LOCAL" != "$peer_val" ]; then
      echo "preflight FAIL: vm.min_free_kbytes asymmetric (local $MIN_FREE_KBYTES_LOCAL vs peer $peer_val) — this caused the 2026-07-18 worker crash-loop outage (kernel low-watermark is reserved from CUDA-visible memory). Standardize BOTH nodes on vm.min_free_kbytes=1048576 (1 GiB): sysctl -w vm.min_free_kbytes=1048576 + update /etc/sysctl.d/90-dspark-oom.conf on the deviating node, then restart" >&2
      exit 1
    fi
    echo "preflight ok: vm.min_free_kbytes symmetric across nodes ($MIN_FREE_KBYTES_LOCAL)"
    return 0
  fi

  persisted=$(grep -E '^[[:space:]]*vm\.min_free_kbytes[[:space:]]*=' "$conf" 2>/dev/null \
    | tail -n1 | sed -E 's/^[[:space:]]*vm\.min_free_kbytes[[:space:]]*=[[:space:]]*([0-9]+)[[:space:]]*$/\1/' || true)
  if ! printf '%s' "$runtime" | grep -qE '^[0-9]+$' || ! printf '%s' "$persisted" | grep -qE '^[0-9]+$'; then
    echo "preflight WARN: vm.min_free_kbytes runtime (${runtime:-<unreadable>}) != persisted 90-dspark-oom.conf (${persisted:-<missing>}) — could not verify" >&2
    return 0
  fi
  if [ "$runtime" != "$persisted" ]; then
    echo "preflight WARN: vm.min_free_kbytes runtime ($runtime) != persisted 90-dspark-oom.conf ($persisted) — next reboot changes memory behavior" >&2
  else
    echo "preflight ok: vm.min_free_kbytes runtime matches persisted ($runtime)"
  fi
  return 0
}

# Memory-collision guard: Inkling needs ~103 GiB/node at util 0.85 out of 121.7 GiB —
# it CANNOT share the node with another big model. Fail fast with a clear message
# instead of letting two engines fight over the pool (slow, confusing OOM deep in
# NCCL rendezvous). Names come from cluster.env (CONFLICT_CONTAINERS/CONFLICT_UNITS —
# set to whatever else runs on your nodes; the reference cluster runs a DeepSeek
# stack as 'vllm-dsv4'). FORCE=1 overrides (you own the consequences).
check_conflicts_absent() {
  local bad=0 name unit
  for name in ${CONFLICT_CONTAINERS:-vllm-dsv4}; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$name"; then
      echo "preflight FAIL: container $name is RUNNING on this node — Inkling cannot coexist with it." >&2
      bad=1
    fi
  done
  for unit in ${CONFLICT_UNITS:-vllm-dsv4-head.service vllm-dsv4-worker.service}; do
    if [ "$(systemctl --user is-active "$unit" 2>/dev/null || true)" = "active" ]; then
      echo "preflight FAIL: conflicting unit $unit is active on this node." >&2
      bad=1
    fi
  done
  if [ "$bad" = "1" ]; then
    echo "Stop the conflicting stack first, or FORCE=1 to override." >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
    echo "preflight WARN: FORCE=1 — proceeding despite a conflict; expect memory failure" >&2
  else
    echo "preflight ok: no conflicting workloads on this node"
  fi
}

case "$ROLE" in
  head)   MY_R1="$HEAD_R1";   PEER_R1="$WORKER_R1" ;;
  worker) MY_R1="$WORKER_R1"; PEER_R1="$HEAD_R1" ;;
  *) echo "preflight FAIL: unknown role $ROLE" >&2; exit 1 ;;
esac

# Boot dependencies, in order (linger starts us early in boot).
wait_for "$BOOT_DEP_WAIT_SECS" "QSFP rail-1 IP $MY_R1 assigned" \
  sh -c "ip -4 -o addr show dev $QSFP_IF | grep -Fq '$MY_R1/'"
wait_for "$BOOT_DEP_WAIT_SECS" "docker daemon answering" docker info
[ -e /dev/infiniband ] || { echo "preflight FAIL: /dev/infiniband missing" >&2; exit 1; }

docker image inspect "$INKLING_VLLM_IMAGE" >/dev/null 2>&1 \
  || { echo "preflight FAIL: image $INKLING_VLLM_IMAGE not present (runbook §2)" >&2; exit 1; }

# Weights present in the dedicated cache (served by HF id, offline mode).
MODEL_HUB_DIR="$HF_CACHE/hub/models--${INKLING_MODEL//\//--}"
[ -d "$MODEL_HUB_DIR" ] && find "$MODEL_HUB_DIR" -name config.json -print -quit | grep -q . \
  || { echo "preflight FAIL: weights not found under $MODEL_HUB_DIR (runbook §3-4)" >&2; exit 1; }

# Non-fatal: runtime vs persisted sysctl drift.
check_min_free_parity

# Fatal (unless FORCE=1): prod must be fully down on this node.
check_conflicts_absent

# A stale container from a previous run must not block compose. NEVER vllm-dsv4 here.
docker rm -f vllm-inkling >/dev/null 2>&1 || true

if [ "$ROLE" = "head" ]; then
  # accept-new is safe on this isolated point-to-point fabric and covers a
  # first boot where known_hosts wasn't seeded yet.
  wait_for "$BOOT_DEP_WAIT_SECS" "peer $PEER_R1 reachable over QSFP" \
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$CLUSTER_USER@$PEER_R1" true

  # Fatal: cross-node sysctl symmetry (see check_min_free_parity).
  check_min_free_parity peer

  # Wait for the worker unit — and revive it if it is start-limited/failed
  # (drill-proven on prod: a flapping window can exhaust the worker's StartLimit,
  # leaving head-waits-forever unless someone resets it).
  deadline=$(( $(date +%s) + WORKER_WAIT_SECS ))
  while :; do
    wstate=$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
      "$CLUSTER_USER@$PEER_R1" systemctl --user is-active vllm-inkling-worker.service 2>/dev/null || true)
    [ "$wstate" = "active" ] && { echo "preflight ok: worker unit active on $PEER_R1"; break; }
    if [ "$wstate" = "failed" ] || [ "$wstate" = "inactive" ]; then
      echo "preflight: worker unit is '$wstate' — reset+start over QSFP"
      ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLUSTER_USER@$PEER_R1" \
        'systemctl --user reset-failed vllm-inkling-worker.service 2>/dev/null; systemctl --user start vllm-inkling-worker.service' || true
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "preflight FAIL: timed out waiting for worker unit on $PEER_R1 (last state: $wstate)" >&2
      exit 1
    fi
    sleep 10
  done
fi

echo "preflight passed ($ROLE)"
