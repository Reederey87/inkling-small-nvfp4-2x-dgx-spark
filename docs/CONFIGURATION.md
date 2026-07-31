# Configuration guide — every knob, what it does, why it's set that way

The serve argv lives in `docker-compose.inkling.yml`; the values live in
`cluster.env`. This doc explains the choices (the recipe defaults are sized for
H200×2, not 121 GiB UMA per node).

## Serve profile

| Knob | Value | Why |
|---|---|---|
| `MAX_MODEL_LEN` | 131072 | Validated baseline. Native ceiling is 1,048,576 — raise stepwise after green; KV cost is ~28 KiB/token cluster-wide (only 7/42 layers are global attention). |
| `MAX_NUM_SEQS` | 8 | UMA-sized. The recipe's 256 assumes datacenter VRAM. 8 keeps worst-case concurrent KV within the pool. |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | Recipe default chunk size. **Trade-off:** the activation/workspace reservation scales with this — 8192 costs ~19.5 GiB vs 1024, i.e. KV pool drops from ~925K to ~213K tokens. Lower it (e.g. 2048) if you need long-context capacity over prefill speed. |
| `GPU_MEMORY_UTILIZATION` | 0.85 | vLLM reads CUDA device-free memory, which dips ~3 GiB during boot churn on unified memory. 0.85 clears the worst observed dip by ~3.3 GiB; 0.88/0.90 failed intermittently. **Change both nodes or neither.** |
| `KV_CACHE_MEMORY_BYTES` | empty | Profiler sizes the pool. Pin it only if you see boot-to-boot KV variance. |
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
- Activation/workspace at `MAX_NUM_BATCHED_TOKENS=8192`: ~19.5 GiB
- KV pool at util 0.85: ~23.5 GiB → **~213K tokens** (1.63 × 131K)
- Engine init: ~21 s; weights load: ~170 s

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
