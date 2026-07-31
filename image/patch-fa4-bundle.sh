#!/usr/bin/env bash
# patch-fa4-bundle.sh — materialize a writable, hardened copy of the
# SecondNatureComputing/flash-attn-4-sm120 FA4 bundle for the DGX Spark (SM121a).
#
# OPTIONAL HARDENING. The load-bearing fix for the varlen cudaErrorIllegalAddress
# lives in the shim (fa4_sm120_shim.py's clamped score_mod). This script adds
# defense-in-depth patches to a copy of the bundle itself:
#   1. paged_kv.py::PagedKVManager.load_page_table — clamp the page-table load index
#      (select-after-load: the mPageTable[page_idx] read executes even when is_valid
#      is false; num_threads=384 > tile_n=128 sweeps 3 tiles of rows → wild read).
#   2. flash_fwd.py::_paged_load_page_table_and_ptrs — same clamp (executed path).
#   3. interface.py — num_stages=1 for the SM120 paged path (conservative; the
#      stage-2 double-buffered paged pipeline is the least-tested code in the bundle).
#
# Usage:  bash patch-fa4-bundle.sh [TARGET_DIR]   (default: /home/nvidia/fa4-bundle-patched)
# Requires: HF cache with the bundle snapshot already downloaded (run any shimmed
#           container once, or huggingface-cli download SecondNatureComputing/flash-attn-4-sm120).
set -euo pipefail

REVISION="60117041e10fcc6f19882afd274318c755a5ef6e"
TARGET="${1:-/home/nvidia/fa4-bundle-patched}"
HF_CACHE="${HF_CACHE:-/home/nvidia/hf-cache-inkling}"
SNAP="$HF_CACHE/hub/models--SecondNatureComputing--flash-attn-4-sm120/snapshots/$REVISION"
SRC="$SNAP/build/torch-cuda"

[ -d "$SRC" ] || { echo "ERROR: bundle snapshot not found at $SRC" >&2; exit 1; }

echo "== materializing writable copy at $TARGET"
rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -rL "$SRC/." "$TARGET/"
chmod -R u+w "$TARGET"
find "$TARGET" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

python3 - "$TARGET" <<'PYEOF'
import re, sys
target = sys.argv[1]

def patch(path, old, new, name):
    p = f"{target}/{path}"
    src = open(p).read()
    if "DSPARK-FIX" in src or "DSPARK-PATCH" in src:
        print(f"  {name}: already patched")
        return
    if old not in src:
        print(f"  ERROR: {name}: anchor not found in {path}" , file=sys.stderr)
        sys.exit(1)
    open(p, "w").write(src.replace(old, new, 1))
    print(f"  {name}: applied")

CLAMP = (
    "# DSPARK-FIX: select-after-load hazard — the page-table load executes\n"
    "            # even when is_valid is false; clamp the index (value still\n"
    "            # discarded by is_valid downstream).\n"
)

# 1. paged_kv.py manager clamp
patch(
    "paged_kv.py",
    "            page = self.mPageTable[page_idx] if is_valid else 0",
    CLAMP +
    "            safe_page_idx = cutlass.min(page_idx, self.mPageTable.shape[0] - 1)\n"
    "            page = self.mPageTable[safe_page_idx] if is_valid else 0",
    "paged_kv.load_page_table clamp",
)

# 2. flash_fwd.py inline clamp (the path the SM120 kernel actually executes)
patch(
    "flash_fwd.py",
    "            page = mPageTable[page_idx] if is_valid else 0",
    CLAMP +
    "            safe_page_idx = cutlass.min(page_idx, mPageTable.shape[0] - 1)\n"
    "            page = mPageTable[safe_page_idx] if is_valid else 0",
    "flash_fwd._paged_load_page_table_and_ptrs clamp",
)

# 3. num_stages=1 for the SM120 paged path
patch(
    "interface.py",
    "                    num_stages_sm120 = 2",
    "                    num_stages_sm120 = 1  # DSPARK-PATCH: conservative; stage-2\n"
    "                                          # paged pipeline is the least-tested path",
    "interface num_stages=1",
)
PYEOF

echo "== done. Set FA4_BUNDLE_DIR=$TARGET in cluster.env (or leave"
echo "   FA4_BUNDLE_DIR empty to use the stock bundle — the shim's clamped score_mod"
echo "   is the load-bearing fix and works with the stock bundle)."
