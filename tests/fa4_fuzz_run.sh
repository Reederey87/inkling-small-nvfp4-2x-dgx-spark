#!/usr/bin/env bash
# fa4_fuzz_run.sh — crash-resilient driver for tests/fa4_fuzz.py.
# The fuzzer dies on any CUDA fault (context poisoned); this loop records the
# last LAUNCHED case as the repro and resumes at the next case (per-case seeds
# make resume exact). Run ON a node with the runtime image + GPU:
#   IMAGE=vllm-inkling-runtime:v026-scipy-fa4sm120 bash tests/fa4_fuzz_run.sh
# Env: N (end index, default 200), LOG (default /tmp/fa4_fuzz.log),
#      KIT (repo root on the node, default /home/nvidia/inkling-small-nvfp4),
#      FA4_BUNDLE_DIR (optional hardened bundle mount source).
set -uo pipefail

IMG="${IMAGE:-vllm-inkling-runtime:v026-scipy-fa4sm120}"
N="${N:-200}"
LOG="${LOG:-/tmp/fa4_fuzz.log}"
KIT="${KIT:-/home/nvidia/inkling-small-nvfp4}"
HF_CACHE="${HF_CACHE:-/home/nvidia/hf-cache-inkling}"
EXTRA_MOUNTS=()
EXTRA_ENV=()
if [ -n "${FA4_BUNDLE_DIR:-}" ]; then
  EXTRA_MOUNTS+=(-v "$FA4_BUNDLE_DIR:/patched:ro")
  EXTRA_ENV+=(-e "FA4_SM120_BUNDLE_DIR=/patched")
fi

: > "$LOG"
i=0
restarts=0
while [ "$i" -lt "$N" ] && [ "$restarts" -lt 25 ]; do
  docker run --rm --gpus all \
    -v "$HF_CACHE:/cache/huggingface" \
    -v "$KIT/tests:/t:ro" \
    "${EXTRA_MOUNTS[@]}" \
    -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 "${EXTRA_ENV[@]}" \
    --entrypoint python3 "$IMG" /t/fa4_fuzz.py --start "$i" --n "$N" >> "$LOG" 2>&1
  ec=$?
  echo "batch exit=$ec start=$i $(date -u +%H:%M:%S)" >> "$LOG"
  [ "$ec" -eq 0 ] && break
  last=$(grep -oE "LAUNCHED case=[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
  [ -z "$last" ] && break
  echo "CRASH at case=$last (exit=$ec)" >> "$LOG"
  i=$((last + 1))
  restarts=$((restarts + 1))
done
echo "FUZZ DONE restarts=$restarts" >> "$LOG"
tail -3 "$LOG"
