# Configuration guide — every knob, what it does, why it's set that way

The serve argv lives in `docker-compose.inkling.yml`; the values live in
`cluster.env`. This doc explains the choices (the recipe defaults are sized for
H200×2, not 121 GiB UMA per node).

## Serve profile

| Knob | Value | Why |
|---|---|---|
| `MAX_MODEL_LEN` | 131072 | Validated baseline. Native ceiling is 1,048,576 — served by the **1M lane** (config, sizing math, gates): [docs/CONTEXT-1M.md](CONTEXT-1M.md). Only 7/42 layers are global attention and vLLM caps SWA/conv state per request, so pool cost is ~15.5 KiB/token/node (measured: 29.5 GiB pin → 2,044,677-token pool) and does not grow with this setting — the pool only has to fit one full-length request. |
| `MAX_NUM_SEQS` | 8 | UMA-sized. The recipe's 256 assumes datacenter VRAM. 8 keeps worst-case concurrent KV within the pool. (1M lane: 2.) |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | Recipe default chunk size. **Trade-off:** the profiler's activation/workspace reservation scales with this — 8192 reserves ~22 GiB, 1024 ~11 GiB — i.e. KV pool drops from ~925K to ~213K tokens. Lower it (e.g. 2048) if you need long-context capacity over prefill speed. (1M lane: 768.) |
| `GPU_MEMORY_UTILIZATION` | 0.85 | vLLM reads CUDA device-free memory, which dips ~3 GiB during boot churn on unified memory. 0.85 clears the worst observed dip by ~3.3 GiB; 0.88/0.90 failed intermittently. **Change both nodes or neither.** (1M lane: 0.80 — startup gate only; the pool is byte-pinned, see CONTEXT-1M.md.) |
| `KV_CACHE_MEMORY_BYTES` | empty | Empty = profiler sizes the pool. **1M lane: 31675322368 (29.5 GiB)** — the pin sizes the pool exactly and skips the profiler reservation ("does not respect gpu_memory_utilization", gpu_worker.py); measured pool 2,044,677 tokens. |
| `ENFORCE_EAGER` | 1 | **Required.** Piecewise cudagraphs crash cross-node on this fabric (#46253 class). Re-test after a vLLM upgrade. |
| `DISABLE_CUSTOM_ALL_REDUCE` | 1 | PYNCCL-only (#41725 class safety). No measured cost at TP=2 over 23 GB/s. |
| `MOE_BACKEND` | empty (auto) | Inkling has NVFP4 routed + BF16 shared experts; a global backend must satisfy both — `cutlass` hard-fails the unquantized family. Auto resolves per family (FLASHINFER_CUTLASS for both on SM121). |
| `MTP_NUM_TOKENS` | empty (off) | MTP speculative decoding: SM121's 99 KB SMEM + K=128 tile history is a qualification project, not a baseline. Try `2` after the stack is green. |
| `REASONING_EFFORT` | high | Inkling chat-template dial. Thinking shares the output token budget — size `max_tokens` accordingly. |
| `--tokenizer-mode inkling` / `--reasoning-parser inkling` / `--tool-call-parser inkling` / `--enable-auto-tool-choice` | on | The model's required serving configuration (from the official recipe). |
| `--kernel-config.enable_flashinfer_autotune=False` | off | Recipe flag; avoids first-step autotune stalls. |
| `--kernel-config.enable_cutedsl_warmup=False` | off | Boot-time cute-dsl warmup spun 30+ min on SM121; lazy per-shape compile is faster end-to-end. |
| `VLLM_USE_V2_MODEL_RUNNER=1` | env | Required by the Inkling serving path. |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` | env | Recipe env; enables the cute-dsl cache layer. |

## Memory map (per node, 121 GiB unified)

- Weights (TP=2 shard): ~78.3 GiB
- Profiler workspace reservation: ~22 GiB at `MAX_NUM_BATCHED_TOKENS=8192`, ~11 GiB at 1024 (bypassed entirely by the byte-pin)
- KV pool at util 0.85 (profiler-sized): **~213K tokens** (MNBT 8192, 1.63 × 131K); MNBT 1024 → **~925K tokens**. Pool cost ~15.5 KiB/token/node (measured).
- 1M lane (util 0.80 + `KV_CACHE_MEMORY_BYTES=29.5 GiB`, MNBT 768): pool **2,044,677 tokens measured** (1.95 × 1M) — usage ≈ 78.3 + 29.5 + ~1.5 = 109 GiB
- Engine init: ~45 s (pin path); weights load: ~170 s

## 1M-context lane

Full 1,048,576-token context on bf16 KV (no fp4 KV exists for this model on
vLLM — why, plus sizing math and gates): [docs/CONTEXT-1M.md](CONTEXT-1M.md).
Config block in `cluster.env.example`; verification probe: `bench/needle.py`.

## FA4 kernel fix knobs (docs/KERNEL-FIX.md)

| Knob | Value | Why |
|---|---|---|
| `FA4_SM120_SHIM` | 1 (default) | Set 0 to disable the shim (stock vLLM — won't boot this model on SM12x). |
| `FA4_SM120_BATCH_SPLIT` | 1 (default) | Splits prefill-containing batches into per-seq kernel calls. Set 0 to reproduce the varlen kernel crash. |
| `FA4_BUNDLE_DIR` | empty | Optional hardened bundle copy (image/patch-fa4-bundle.sh). Stock bundle works — the shim fixes are the load-bearing ones. |
| `SHIM_PATH` | `$KIT_DIR/image/fa4_sm120_shim.py` | Bind-mounted into the container so shim updates don't need image rebuilds. |

## Fabric / NCCL

Dual PCIe-twin rails merged by NCCL (`NCCL_IB_MERGE_NICS=1`,
`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`) → ~23 GB/s measured all-reduce. MTU 9000.
`MASTER_ADDR` must be the rail-1 head IP. See docs/NETWORKING.md.
