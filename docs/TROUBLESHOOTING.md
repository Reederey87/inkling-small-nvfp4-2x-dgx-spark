# Troubleshooting — every failure this deployment hit, and its fix

Ordered by bring-up phase. If you hit something new, the diagnostic techniques
at the bottom are how each of these was isolated.

## Boot failures

### `ModuleNotFoundError: No module named 'scipy'`
Inkling's model class imports `linear_sum_assignment` at construction; the stock
`vllm/vllm-openai:v0.26.0` image doesn't ship scipy. Fixed in the overlay
Dockerfile (`pip install scipy einops apache-tvm-ffi`).

### `ValueError: Free memory on device ... less than desired`
vLLM reads **CUDA device-free memory**, not `MemAvailable`. On the Spark's
unified memory, the boot-time dip band is ~3 GiB; `GPU_MEMORY_UTILIZATION`
0.88–0.90 failed intermittently. **0.85** clears the worst observed dip by
~3.3 GiB. Rule: change on BOTH nodes or neither (asymmetric TP fails hard).

### `--moe-backend cutlass` rejected
Inkling has TWO MoE families: NVFP4 routed experts AND unquantized BF16 shared
experts. A single global backend must satisfy both; `cutlass` is invalid for the
unquantized family. Leave `MOE_BACKEND` empty (auto): the families resolve
independently → `FLASHINFER_CUTLASS` (NVFP4) + `FlashInfer CUTLASS` (BF16) on
SM121.

### `AssertionError: Paged KV not supported on SM 12.0 in this PR`
vLLM's bundled FA4 gates SM12x out of paged KV. Inkling's relative attention
hard-requires the FA4 path (score_mod + aux_tensors). This is what the
`fa4_sm120_shim.py` + the SecondNatureComputing SM120 bundle solve — see
docs/KERNEL-FIX.md.

### Boot "hangs" 30+ min at 96% GPU during warmup
`--kernel-config.enable_cutedsl_warmup=False` is in the serve argv: the
boot-time cute-dsl warmup JIT-compiles every shape bucket serially. Lazy
compile at first request is faster end-to-end. First request(s) after boot pay
seconds of JIT — expected, one-time per shape bucket per boot.

### `Input tensor addresses changed between capture and replay`
Piecewise/breakable cudagraphs + cross-node host-staged NCCL collectives don't
mix on this fabric (vLLM #46253 class). `--enforce-eager` is mandatory here.

## Serving failures

### First real request stalls forever; engine dies after ~10 min (RPC timeout)
This was the window-1 blocker. Root cause: the SM120 FA4 kernel faults on
varlen batches (docs/KERNEL-FIX.md). Async launch → hung stream; blocking
launch → `cudaErrorIllegalAddress`. Fixed by the shim's clamped score_mod +
batch-split. If you disabled the shim (`FA4_SM120_SHIM=0`) or the split
(`FA4_SM120_BATCH_SPLIT=0`), you get the crash back.

### `cudaErrorIllegalAddress` under concurrent load
Same root cause, above. Verify the running container actually has the fixes:
`docker exec vllm-inkling python3 -c "import vllm.models.inkling.nvidia.ops.fa4_rel_attention as m; print(m._get_score_mod.__name__)"`
must print `_get_score_mod_clamped`. If it doesn't, the bind-mounted shim
(`SHIM_PATH` in cluster.env) isn't in place — re-run `install-services.sh` and
restart the units.

### Empty `content`, `finish_reason: length`, coherent `reasoning`
Not a bug: with `reasoning_effort: high` the model thinks first; thinking
shares the `max_tokens` budget. Raise `max_tokens` (or lower effort via the
chat template kwarg).

### Cross-node collectives hang but single-node NCCL tests pass
Two known causes here. First, the FA4 kernel faults (docs/KERNEL-FIX.md) — fixed
by the shim. Second, **NCCL 2.28.9 wedging under serving load on RoCE**: workers
host-block inside `ncclAllGather` (py-spy shows `_sconv_add_norm`, model.py:111),
GPU spins at 96%, streams never drain, the head RPC-times-out. The stock
`vllm/vllm-openai:v0.26.0` image ships `nvidia-nccl-cu13==2.28.9`; the reference
production stack on this fabric runs **2.30.7** daily without a hang — the
overlay Dockerfile pins it. Verify what the worker actually loaded:
`docker exec vllm-inkling grep -o "/[^ ]*libnccl[^ ]*" /proc/<worker-pid>/maps | sort -u`
(the pip lib, not the system one, is what pynccl uses). Check the fabric env
matches `cluster.env` exactly (dual-HCA merge is load-bearing), then run
`tests/nccl_collectives.py` on both nodes — it reproduces the deployment's
exact NCCL env. If plain collectives pass, the issue is above NCCL.

### systemd unit `failed` after repeated crashes
`systemctl --user reset-failed vllm-inkling-head` (and worker) — systemd
rate-limits restart loops. The units intentionally do NOT auto-restart on
failure during bring-up.

## Diagnostic techniques (how each root cause was isolated)

- **py-spy from a sidecar** (container has no SYS_PTRACE):
  `docker run --rm --privileged --pid=host <image> py-spy dump --pid <host-pid>`
  (`docker top vllm-inkling -eo pid,comm` to find the worker PID).
- **NCCL collective trail**: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL`, then
  align per-communicator `(op, count)` sequences between rank journals — a
  desync shows as one rank launching collectives the other never joins.
- **`CUDA_LAUNCH_BLOCKING=1`**: converts "which kernel hung" into "which launch
  never returns" — py-spy then names the exact op. (Also masks races: our stall
  vanished under it — that was the clue the fault was timing-sensitive.)
- **Op-level fuzzing**: `tests/fa4_fuzz.py` reproduces serving shapes without
  the engine (200 randomized cases, watchdog per case).
- **Batched-vs-per-seq numerics**: `tests/fa4_numeric_check.py` proves whether
  a workaround changes outputs (ours are bit-exact).
