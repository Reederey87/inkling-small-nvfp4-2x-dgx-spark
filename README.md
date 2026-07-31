# Inkling-Small-NVFP4 on 2× NVIDIA DGX Spark

Production-grade deployment kit for **[thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)**
(276B-parameter MoE, 12B active, 1M-context, NVFP4) on a **2-node NVIDIA DGX
Spark cluster** (GB10 Grace-Blackwell, SM121a, 121 GiB unified memory per node),
tensor-parallel across the two nodes over a direct QSFP link, served by
**vLLM 0.26.0**.

Operated **from a Mac over SSH** — the Mac only orchestrates; everything runs on
the Sparks. The whole flow is scripted: configure one file, run
`install-services.sh`, `preflight.sh`, `start-cluster.sh`.

**Tested on 2× DGX Spark** (DGX OS / Ubuntu 24.04, driver 580.173.02,
CUDA 13.0, Docker 29.x). This repo is the generic, sanitized recipe of a real
deployment — including two upstream-reportable kernel fixes required to serve
the model at all on SM12x (see [docs/KERNEL-FIX.md](docs/KERNEL-FIX.md)).

## Measured performance (reference cluster, stock profile)

2-node TP=2, vLLM 0.26.0, `reasoning_effort=high` (thinking tokens included),
2048-token generations, ~576-token prompts, prefix caching defeated:

| Metric | Value | Notes |
|---|---|---|
| Decode throughput, C1 | **21.4 tok/s** | 21.39 / 21.46 / 21.46 across 3 reps |
| Decode throughput, C8 | **93.9 tok/s** aggregate | 92.4 / 94.3 / 95.1 across 3 reps (16,384 tokens/rep) |
| C8 scaling | **4.4×** vs C1 | 8 concurrent requests |
| KV pool | **213,780 tokens** (1.63× @131K ctx) | at `GPU_MEMORY_UTILIZATION=0.85`, `MAX_NUM_BATCHED_TOKENS=8192`; see the trade-off knob |
| Weights footprint | 78.3 GiB / node | TP=2 shards of the ~160 GB checkpoint |
| Boot to serving | ~5 min | weights ~170 s + engine init ~21 s |
| Fabric | ~23 GB/s | NCCL all-reduce over dual-rail QSFP RoCE |
| Stability gate | 200/200 kernel-fuzz cases, C1×3 + C8×3 bench reps, C8 soak | all clean |

Context: on the same 2× Spark hardware, the production DeepSeek-V4-Flash stack
(284B/13B-active, vLLM 0.26 + MTP spec-decode) measures ~21→88 tok/s C1→C8 in
the same harness class — Inkling-Small lands in the same serving envelope
(C8 ~94 tok/s) *without* speculative decoding enabled yet.

Config knobs that move these numbers (context length ↔ KV pool ↔ prefill
speed): [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## What it takes to serve this model on DGX Spark (work done)

Stock vLLM 0.26 cannot serve Inkling on SM12x. This repo encodes the full
bring-up — each item below was a hard blocker found and fixed on the reference
cluster:

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
3. **MoE backend selection** — Inkling has NVFP4 routed *and* BF16 shared
   experts; a global `--moe-backend` must satisfy both (auto resolves to
   FLASHINFER_CUTLASS on SM121; `cutlass` hard-fails).
4. **`--enforce-eager` is mandatory** — piecewise cudagraphs crash cross-node
   (vLLM #46253 class).
5. **Memory sizing for unified memory** — vLLM reads CUDA device-free, not
   MemAvailable; `GPU_MEMORY_UTILIZATION=0.85` clears the boot-time dip band.
6. **scipy/einops/tvm-ffi overlay** on the official arm64 image (the model
   imports them at construction) — [`image/Dockerfile`](image/Dockerfile).

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
systemd/                # head/worker user units
tests/                  # kernel self-tests: shim check, serving shapes, fuzzer, numerics, NCCL
bench/                  # smoke.sh, probe.sh (C1/C8), cbench.py
docs/                   # DEPLOYMENT, CONFIGURATION, KERNEL-FIX, TROUBLESHOOTING,
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
```

Full guide: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Anything that can go
wrong (we hit it first): [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

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
- NCCL: the overlay pins `nvidia-nccl-cu13==2.30.7` — the stock image's 2.28.9
  wedges cross-node collectives under serving load on this fabric (see
  docs/TROUBLESHOOTING.md).

## License & credits

Apache-2.0 (see LICENSE). Model: Thinking Machines Lab (Apache-2.0). vLLM:
vLLM project. SM120 FA4 bundle: Second Nature Computing / Dao-AILab
flash-attention (BSD-3). This repo is an independent deployment recipe, not
affiliated with TML, vLLM, or NVIDIA.
