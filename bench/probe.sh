#!/usr/bin/env bash
# Inkling throughput probe (G3) + soak (G4). Runs from the Mac; executes cbench.py on spark1
# against the loopback API, records JSONL under bench/results/ on the Mac.
#
#   bash bench/probe.sh          # baseline: C1 x3 reps + C8 x3 reps (2048 max tokens)
#   bash bench/probe.sh soak     # 30 min continuous C8 — watch for the #41725 hang class
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

MODE="${1:-base}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$KIT/bench/results"

run_cbench() { # run_cbench <concurrency> <reps> <label> <outfile>
  local conc="$1" reps="$2" label="$3" out="$4"
  ssh "$CLUSTER_USER@$HEAD_HOST" \
    "python3 - --url http://127.0.0.1:$API_PORT --model $SERVED_MODEL_NAME \
       --concurrency $conc --reps $reps --max-tokens 2048 --label $label" \
    < "$KIT/bench/cbench.py" | tee -a "$out"
}

case "$MODE" in
  base)
    OUT="$KIT/bench/results/probe-base-$TS.jsonl"
    echo "== C1 x3 -> $OUT"
    run_cbench 1 3 c1 "$OUT"
    echo "== C8 x3 -> $OUT"
    run_cbench 8 3 c8 "$OUT"
    ;;
  soak)
    OUT="$KIT/bench/results/probe-soak-$TS.jsonl"
    echo "== soak: C8 x1 reps in a 30-min loop -> $OUT"
    deadline=$(( $(date +%s) + 1800 ))
    i=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
      i=$((i + 1))
      echo "== soak rep block $i ($(date -u +%H:%M:%SZ))"
      run_cbench 8 1 "soak$i" "$OUT" || echo "WARN: soak block $i had request errors" >&2
    done
    ;;
  *)
    echo "usage: bash bench/probe.sh [base|soak]" >&2
    exit 1
    ;;
esac
echo "== done. Results: $OUT"
