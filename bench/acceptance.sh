#!/usr/bin/env bash
# acceptance.sh — sample the vLLM spec-decode Prometheus counters on the head
# node and compute acceptance rate / mean acceptance length over a window.
# Run from the Mac while traffic is flowing (e.g. during bench/probe.sh):
#   bash bench/acceptance.sh [SECONDS]     (default 60)
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"
SECS="${1:-60}"

ssh "nvidia@$HEAD_HOST" "python3 - '$SECS' '$API_PORT'" <<'PY'
import re, sys, time, urllib.request
secs, port = float(sys.argv[1]), sys.argv[2]

def sample():
    txt = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    def g(name):
        for cand in (name + "_total", name):
            m = re.search(rf"^{re.escape(cand)}(?:\{{[^}}]*\}})? (\d+(?:\.\d+)?)$", txt, re.M)
            if m:
                return float(m.group(1))
        return 0.0
    return (g("vllm:spec_decode_num_drafts"),
            g("vllm:spec_decode_num_draft_tokens"),
            g("vllm:spec_decode_num_accepted_tokens"))

d0, t0, a0 = sample()
time.sleep(secs)
d1, t1, a1 = sample()
dd, dt, da = d1 - d0, t1 - t0, a1 - a0
print(f"window={secs:.0f}s drafts={dd:.0f} draft_tokens={dt:.0f} accepted={da:.0f}")
if dt > 0:
    print(f"acceptance_rate={da/dt:.3f} mean_accept_length={1 + da/max(dd, 1.0):.2f}")
else:
    print("no spec-decode traffic in window (MTP off or engine idle)")
PY
