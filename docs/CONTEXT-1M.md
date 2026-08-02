# 1M-context lane — full 1,048,576-token window on 2× DGX Spark (vLLM, bf16 KV)

How to serve Inkling-Small-NVFP4 at its native 1,048,576-token context on this
kit, why it works on vLLM **without** a quantized KV cache, and why the fp4-KV
route that SGLang uses is not available here (yet).

**Verified on the reference cluster (2026-08-01): KV pool 2,044,677 tokens =
1.95× the 1M window, smoke 7/7, serving green.** The mechanism below is the
result of a boot-debugging campaign — read it before touching the values.

## TL;DR — pin the KV pool, underutilize the gate

```bash
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=768
GPU_MEMORY_UTILIZATION=0.80          # startup gate only — always passes
KV_CACHE_MEMORY_BYTES=31675322368    # 29.5 GiB → MEASURED pool 2,044,677 tokens
```

Boot, then confirm the head journal: `GPU KV cache size: 2,044,677 tokens` and
`Maximum concurrency for 1,048,576 tokens per request: 1.95x`.

The two knobs do different jobs, and confusing them is what makes this lane
hard:

- **`GPU_MEMORY_UTILIZATION` is only a startup GATE.** vLLM 0.26's
  `request_memory()` (`v1/worker/utils.py`) unconditionally requires
  `free ≥ util × total` at boot. On these nodes `cudaFreeMem` reads
  ~105.8–106.9 GiB with the desktop/Hermes resident — so 0.88/0.89 can *never*
  boot and 0.875 passes only on lucky churn readings. At 0.80 (97.35 GiB) the
  gate always passes.
- **`KV_CACHE_MEMORY_BYTES` sizes the pool — and skips both the gate-math and
  the profiler.** Per `v1/worker/gpu_worker.py` (~line 462), when the pin is
  set the pool is exactly the pin and "does not respect the
  gpu_memory_utilization config". This also bypasses the profiler's
  conservative workspace reservation (~11 GiB at MNBT 1024, ~22 GiB at 8192 —
  that reservation, not the KV itself, is what made 1M unreachable through the
  util lever: at util 0.85 the profiler left only a 925,452-token pool).

Envelope: 78.3 weights + 29.5 KV + ~1.5 workspace ≈ **109 GiB** of 121.69 —
the kernel reclaims page cache to cover the gap vs the conservative
`cudaFreeMem` reading. It boots clean; OS/desktop keep ~12 GiB. If you want
more system margin and only need one full-length request, pin ~17 GiB
(≈1.1M pool) instead — see the sizing table.

## Why 1M is (nearly) free on vLLM — the hybrid-allocator effect

Inkling is a hybrid: 42 layers = **7 global (full) attention + 35 sliding-window
(SWA-512)**, plus short-conv state. Two different KV-allocator behaviors matter:

- **SGLang (bf16 KV):** the SWA/mamba reserves *scale with the declared
  context*, so raising 64K → 1M costs most of the pool — bf16 caps near ~354K
  pool tokens. That is why the SGLang port needed a **patched fp4 KV cache**
  (`--kv-cache-dtype fp4_mx_block16`, a custom 6-file patch) to make 1M
  essentially free there (pool 1,104,683 @64K → 1,082,627 @1M, ~2%).
- **vLLM (this kit, bf16 KV):** the hybrid KV-cache manager caps SWA layers at
  their 512-token window and conv state at its kernel size — **constant,
  independent of `max_model_len`**. Only the 7 global layers scale with
  context. The pool therefore barely cares about the context setting either;
  it just has to fit **one** full-length request
  (`check_enough_kv_cache_memory` at startup).

Measured pool sizes on the reference cluster (all bf16 KV):

| config | KV pool (tokens) | per-token cost |
|---|---|---|
| util 0.85, MNBT 8192 (profiler) | 213,780 | — |
| util 0.85, MNBT 1024 (profiler) | 925,452 | — |
| **pin 29.5 GiB, MNBT 768** | **2,044,677** | **~15.5 KiB/token/node** |

The per-token rate matches the structure: vLLM groups the 42 layers 7-per-group
into 7 shared buffers; TP=2 shards the 8 KV heads to 4/node; 7 × 4 × 128 ×
2 (K+V) × 2 B = 14 KiB + fragmentation ≈ 15.5 KiB/token/node (≈28 KiB
cluster-wide). **1M tokens ≈ 15.9 GiB/node of KV** — cheap. What was expensive
is the profiler's workspace reservation, which the pin bypasses.

Pin-size guide (at ~15.5 KiB/token/node):

| `KV_CACHE_MEMORY_BYTES` | pool (tokens) | use |
|---|---|---|
| ~17 GiB (`18253611008`) | ~1.1M | minimal 1M lane, max system margin |
| ~24 GiB (`25769803776`) | ~1.55M | headroom choice |
| 28.5 GiB (`30601641984`) | **1,950,538 (measured)** | **1M lane + MTP drafter on** |
| **29.5 GiB (`31675322368`)** | **~2.04M (measured)** | 2× full-length concurrency |

**MTP drafter headroom (2026-08-02):** with `MTP_NUM_TOKENS=1` the engine loads
the checkpoint's NextN depth-0 block (BF16, ~0.3 GiB/node TP-sharded) plus a
1-layer draft KV and the rejection sampler — but the pin bypasses the profiler
that would otherwise reserve for them, so shave the pin by ~1 GiB (29.5 → 28.5,
measured pool above). If boot or first-token OOMs ever appear on other rigs,
drop another 0.5–1 GiB; the byte-pin path has no safety check by design
(`gpu_worker.py` skips memory profiling when `--kv-cache-memory-bytes` is set).

## Why not fp4/NVFP4 KV cache on vLLM? (the SGLang trick, examined)

On vLLM 0.26 a quantized KV cache does not exist for this model:

1. **Upstream nvfp4 KV is gated to SM100 (datacenter Blackwell).**
   `--kv-cache-dtype nvfp4` requires the FlashInfer trtllm-gen path and
   capability family 100; GB10 is SM121 → family 12 → rejected at backend
   selection (`kv_cache_dtype not supported`). Same in current master.
2. **SM120/121 nvfp4 KV exists only in unmerged PRs** — vllm-project/vllm
   **#46329** (FlashInfer FA2 path, Gemma 3/4, validated on GB10 sm_121,
   ~3.5× pool vs bf16), **#50288**, issue **#49011** — and those cover
   **standard attention** models. Inkling does not use the stock attention
   path at all.
3. **Inkling's KV path is custom and bf16-only on vLLM:** FA4
   relative-attention kernels (shim: `image/fa4_sm120_shim.py`) +
   `fused_qkvr_prep` write KV in model dtype; the conv state asserts bf16.
   There is no scale/descale plumbing — passing any quantized
   `--kv-cache-dtype` would feed garbage. (fp8 KV is nominally registered in
   the model's attention wrapper but unverified with the relative-attention
   kernel — documented last resort, not a lane.)
4. The checkpoint itself declares `kv_cache_quant_algo: "none"`.

Porting fp4 KV to Inkling-on-vLLM means new packed-fp4 write/read paths in the
custom kernels (the same class of work the SGLang patch was) plus the SM12x
gate-lifts from #46329 — a kernel project, not a config flip. Track the
upstream PRs; meanwhile the bf16 pin above already delivers ~2× the 1M window.
Quantized KV would buy headroom (a ~3.5× larger pool), not the window itself.

## Bring-up notes (learned the hard way)

- **Both nodes get the same `cluster.env`** (parity rule — preflight fails on
  asymmetry); `render-env.sh` runs per node at unit start.
- **Boot-order trap:** the worker (rank 1) blocks at the distributed-init
  barrier until the head (rank 0, the rendezvous) exists. Start worker, then
  head promptly (`start-cluster.sh` does) — don't wait for the worker's
  weight-load to finish before starting the head.
- If a unit hits systemd's start-rate limit after repeated failures
  ("Start request repeated too quickly"): `systemctl --user reset-failed
  vllm-inkling-{head,worker}.service`, then start again.
- **Gates after boot:**
  1. **Pool gate** — head journal shows `GPU KV cache size` ≥ 1,048,576 and
     `Maximum concurrency for 1,048,576 tokens per request` ≥ 1.0x.
  2. **Acceptance gate** — a >131,072-token request must not 400.
  3. **Retrieval gate** — `bench/needle.py --depths 65536 262144 524288`
     (run on the head; API is loopback-only). Long prefills are SLOW at
     MNBT 768 — minutes for ≥256K prompts; that is the lane cost, not a fault.
  4. `bench/smoke.sh` + `bench/probe.sh` — decode throughput is
     pool-size-independent and should match the stock profile.

## Client side (Hermes)

`hermes/config.yaml` ships matched to the 131K default. For the 1M lane set
`model.context_length: 1048576` and the compression block to
`threshold: 0.25`, `threshold_tokens: 260000`,
`proactive_prune_tokens: 48000`, `proactive_prune_min_reclaim_tokens: 20000`
(keep `idle_compact_after_seconds: 0` — compaction runs on the local model).
Commented values are inline in `hermes/config.yaml`.

## Trade-offs / when NOT to use the lane

- **Prefill speed:** MNBT 768 vs 8192 costs real TTFT on long prompts (the
  workspace reservation is what made prefill fast). Throughput traffic (many
  short requests) belongs on the default profile.
- **System margin:** at the 29.5 GiB pin the box runs with ~12 GiB free.
  Co-tenant spikes (VS Code Remote, browser) eat into it; the 17 GiB pin is
  the conservative choice.
- **Memory-check history:** before the pin mechanism, this lane required
  `GPU_MEMORY_UTILIZATION≈0.875`, which passes the `request_memory()` gate
  only intermittently on nodes with a desktop session (cudaFreeMem churns
  105.8–106.9 GiB). The pin makes boot deterministic at any co-tenant state
  that leaves ~110 GiB physically available.
