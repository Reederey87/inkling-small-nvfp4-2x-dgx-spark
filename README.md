# Inkling-Small-NVFP4 on 2× NVIDIA DGX Spark — full 1M context, production kit

Serve **[thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)**
(276B-parameter MoE, 12B active, NVFP4) at its **native 1,048,576-token context**
on **two desktop NVIDIA DGX Sparks** (GB10 Grace-Blackwell, SM121a, 121 GiB
unified memory each), tensor-parallel over a single QSFP cable, on **vLLM 0.26.0**.

This is the complete, reproducible deployment kit — configs, systemd units,
kernel shim, fuzz/bench harnesses, Hermes Agent profile, and docs that explain
every parameter choice — proven on a real cluster, operated from a Mac over SSH.

**Headline results (measured on the reference cluster, 2026-08-01):**

| | Value |
|---|---|
| Context window | **1,048,576 tokens** (model-native ceiling) |
| KV pool | **2,044,677 tokens — 1.95× the window** (bf16 KV, no fp4 needed) |
| Needle retrieval | **HIT at 9 depths, out to 1,020,076 tokens** |
| Decode throughput | **21.4 tok/s C1 · 93.9 tok/s C8** (4.4× scaling) |
| Long-ctx prefill | ~1.8–2.4K tok/s (full-1M cold prefill ≈ 9–13 min TTFT) |
| Stability | smoke 7/7 (incl. vision) · 200/200 kernel-fuzz cases · C8 soak clean |
| Boot to serving | ~5 min (weights ~170 s + engine init ~45 s) |
| Fabric | ~23 GB/s NCCL all-reduce over dual-rail QSFP RoCE |

## Why this repo

- **Full 1M context on vLLM — without quantizing the KV cache.** The documented
  recipe elsewhere (SGLang) needed a custom fp4-KV patch. On vLLM we measured
  that Inkling's hybrid KV allocator caps the 35 sliding-window + conv layers
  per request — only 7/42 global-attention layers scale with context —
  so the whole 1M window costs ~16 GiB of plain bf16 per node. What actually
  blocked 1M was accounting (the profiler's workspace reservation and the
  `request_memory()` boot gate), and the fix is two lines: a KV **byte-pin** +
  util as gate-only. Full analysis: [docs/CONTEXT-1M.md](docs/CONTEXT-1M.md).
- **It serves at all only because of the work encoded here.** Stock vLLM 0.26
  cannot run this model on SM12x: the FA4 paged-KV path is gated off, the
  fallback kernel crashes on real serving batches, and the stock NCCL wedges
  cross-node collectives. Each blocker is root-caused, fixed, and covered by a
  test harness — see below.
- **Everything is measured, nothing is vibes.** Every number above comes from
  the harnesses in `bench/` and `tests/`, and every knob in `cluster.env.example`
  carries the rationale (and usually the failure) behind its value.
- **Agent-ready.** Ships a Hermes Agent profile (1M window, local vision aux,
  retuned compaction), a switch script, and a de-streaming proxy that fixes
  vLLM 0.26's streaming tool-call parser for this model.

## Measured performance

2-node TP=2, vLLM 0.26.0, `reasoning_effort=high` (thinking tokens included),
2048-token generations, ~576-token prompts, prefix caching defeated:

| Metric | Stock profile (131K) | 1M lane |
|---|---|---|
| Context | 131,072 | **1,048,576** |
| KV pool | 213,780 tokens (1.63×) | **2,044,677 tokens (1.95×)** |
| Decode C1 | **21.4 tok/s** (3 reps: 21.39/21.46/21.46) | pool-size-independent |
| Decode C8 | **93.9 tok/s** aggregate (3 reps: 92.4/94.3/95.1) | pool-size-independent |
| Needle retrieval | — | **HIT @ 44K–1,020K** (`bench/needle.py`) |
| Weights footprint | 78.3 GiB / node | same |
| Config | `cluster.env.example` defaults | one commented block |

Context: on the same hardware, the production DeepSeek-V4-Flash stack
(284B/13B-active, vLLM 0.26 + spec decode) measures ~21→88 tok/s C1→C8 in the
same harness class — Inkling-Small lands in the same serving envelope *without*
speculative decoding enabled yet.

## The 1M-context lane (60 seconds)

```bash
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=768
GPU_MEMORY_UTILIZATION=0.80          # startup gate only — always passes
KV_CACHE_MEMORY_BYTES=31675322368    # 29.5 GiB → measured pool 2,044,677 tokens
```

`GPU_MEMORY_UTILIZATION` is only vLLM's unconditional startup check
(`free ≥ util×total`), which a desktop-equipped node can never satisfy above
0.87; `KV_CACHE_MEMORY_BYTES` sizes the pool exactly and *"does not respect
gpu_memory_utilization"* (`gpu_worker.py`), bypassing both the gate math and
the profiler's 11–22 GiB workspace reservation. Pin-size guide (17 GiB ≈ 1.1M
pool for more system margin), bring-up traps, gates, and the fp4-KV analysis:
[docs/CONTEXT-1M.md](docs/CONTEXT-1M.md).

## What it took to serve this model on DGX Spark (work done)

Each item was a hard blocker found and fixed on the reference cluster:

1. **FA4 paged-KV is gated off on SM12x** — Inkling's relative attention
   requires FA4's `score_mod` + `aux_tensors`. The kit ships
   [`image/fa4_sm120_shim.py`](image/fa4_sm120_shim.py), redirecting vLLM's FA4
   calls to the SecondNatureComputing SM120 bundle (upstream PRs #2348/#2336
   et al., revision-pinned).
2. **The bundle's varlen kernel crashes on real serving** — two distinct
   defects, root-caused with a 200-case fuzzer + batch-vs-per-seq numeric
   proofs, fixed in the shim: a **bounds-clamped score_mod** (aux gather OOB on
   padded varlen rows) and **batch-splitting** of prefill-containing batches
   (residual b≥3 global-window faults). Evidence + upstream-ready write-ups:
   [docs/KERNEL-FIX.md](docs/KERNEL-FIX.md).
3. **NCCL 2.28.9 wedges cross-node collectives under load** — py-spy caught a
   rank host-blocked inside `ncclAllGather` with the GPU spinning at 96%. The
   overlay pins `nvidia-nccl-cu13==2.30.7`; soak clean.
4. **MoE backend selection** — Inkling has NVFP4 routed *and* BF16 shared
   experts; a global `--moe-backend` must satisfy both (auto resolves to
   FLASHINFER_CUTLASS on SM121; `cutlass` hard-fails).
5. **`--enforce-eager` is mandatory** — piecewise cudagraphs crash cross-node
   (vLLM #46253 class).
6. **The 1M memory-gate campaign** — unified-memory `cudaFreeMem` churns
   105.8–106.9 GiB with a desktop resident, so util-based boots at ≥0.875
   flap forever; solved deterministically with the byte-pin lane above.
7. **Memory sizing for unified memory** + **scipy/einops/tvm-ffi overlay** on
   the official arm64 image — [`image/Dockerfile`](image/Dockerfile).

## Layout

```
cluster.env.example     # all knobs + rationale — copy to cluster.env, edit [SITE]
docker-compose.inkling.yml
render-env.sh           # cluster.env -> node-local .env.inkling
install-services.sh     # sync repo to both nodes + install systemd user units
preflight.sh            # go/no-go checks (fabric, images, weights, conflicts)
start-cluster.sh        # worker first, then head; polls /health
stop-cluster.sh
image/                  # Dockerfile, FA4 SM120 shim (+ fixes), bundle hardening script
systemd/                # head/worker user units + de-streaming proxy unit
proxy/                  # de-streaming proxy (Hermes tool-call workaround)
hermes/                 # Hermes Agent profile (1M-ready), switch script, setup guide
tests/                  # kernel self-tests: shim check, serving shapes, fuzzer, numerics, NCCL
bench/                  # smoke.sh, probe.sh (C1/C8), cbench.py, needle.py (long-ctx probe)
docs/                   # DEPLOYMENT, CONFIGURATION, CONTEXT-1M (1M-context lane),
                        # KERNEL-FIX, TROUBLESHOOTING,
                        # NETWORKING, UPSTREAM-ISSUES (ready-to-file bug reports)
```

## Quickstart

```
cp cluster.env.example cluster.env && $EDITOR cluster.env   # [SITE] blocks
bash install-services.sh          # -> both nodes
# build image + download ~160 GB weights per node (docs/DEPLOYMENT.md §4-5)
bash preflight.sh
bash start-cluster.sh
bash bench/smoke.sh               # health, models, QA, concurrency
bash bench/probe.sh               # C1 + C8 throughput
# 1M lane: uncomment the block in cluster.env, re-sync, restart, then:
ssh nvidia@<head> 'python3 ~/inkling-small-nvfp4/bench/needle.py --depths 700000 950000'
```

Full guide: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Anything that can go
wrong (we hit it first): [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
Running an agent client (Hermes Agent) against the deployment, incl. the
required de-streaming proxy: [hermes/README.md](hermes/README.md).

## Requirements

- 2× NVIDIA DGX Spark with a direct QSFP DAC (static IPs on both rails, MTU 9000)
- A user in `docker` group with linger enabled, passwordless SSH from the Mac
- ~180 GB free NVMe per node; no HF token needed (model is public, Apache-2.0)
- vLLM 0.26.0 arm64 base image (pulled by the Dockerfile)

## Model notes

- Thinking is on by default (`reasoning_effort: high`); chain-of-thought is
  returned in `message.reasoning`, not `content`. Small `max_tokens` can be
  fully consumed by thinking (empty content, `finish_reason=length`).
- The NVFP4 checkpoint runs W4A16-style on SM121 (the W4A4 path targets
  SM100+); the kernel dequantizes routed experts on the fly. Works fine —
  the throughput numbers above are this path.
- Multimodal: the smoke gate verifies a basic image input (`vision-red` check)
  end-to-end through this deployment. Benchmarks are text-only; audio is
  untested (the recipe's `vllm[audio]` extra is not in the overlay).
- No fp4/NVFP4 **KV cache** for this model on vLLM 0.26 (SM100-gated upstream;
  Inkling's KV path is custom bf16-only kernels) — the 1M lane uses bf16 KV,
  which suffices; see [docs/CONTEXT-1M.md](docs/CONTEXT-1M.md).

## License & credits

Apache-2.0 (see LICENSE). Model: Thinking Machines Lab (Apache-2.0). vLLM:
vLLM project. SM120 FA4 bundle: Second Nature Computing / Dao-AILab
flash-attention (BSD-3). This repo is an independent deployment recipe, not
affiliated with TML, vLLM, or NVIDIA.
