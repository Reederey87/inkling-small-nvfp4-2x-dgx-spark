# The FA4 SM120 kernel situation on DGX Spark (and the two fixes this repo ships)

Audience: anyone deploying Inkling-Small-NVFP4 on SM12x hardware (DGX Spark GB10
SM121a, RTX 50 / RTX PRO Blackwell). This is the longest and most important doc
in the repo — it explains *why* the shim in `image/fa4_sm120_shim.py` exists and
exactly what it fixes, with the evidence.

## Background: Inkling needs FA4, SM12x can't run stock FA4 paged

Inkling's attention is "relative attention": a learned per-(query,head,distance)
bias added to the scores, implemented as a FlashAttention-4 (CuTe DSL)
`score_mod` + `aux_tensors` gather over a paged KV cache. vLLM 0.26's vendored
FA4 gates SM12x with:

```
assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
```

Upstream paged-KV support for SM120 exists only as unmerged PRs
(Dao-AILab/flash-attention #2348, #2336, #2349, #2389, #2439, #2484). The
`SecondNatureComputing/flash-attn-4-sm120` HF repo bundles those PRs, validated
on SM121a. The shim redirects vLLM's `vllm.vllm_flash_attn.cute.interface._flash_attn_fwd`
to that bundle on SM12x devices (revision-pinned; set `FA4_SM120_SHIM=0` to disable).

That gets the model *booting*. Serving is another story.

## Bug 1 — `rel_logits` aux gather reads out of bounds on padded varlen rows

**Symptom:** intermittent `cudaErrorIllegalAddress` (700) in
`_flash_attn_fwd` under real serving; ~12% of randomized serving-realistic
varlen batches with `b >= 2`. Same fault, different manifestation, under async
launch: a hung rank stream (the "first request stalls cross-node" symptom).

**Root cause:** vLLM's `score_mod_rel_bias`
(`vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`) computes
`global_q_idx = seqlen_info.offset_q + q_idx` and gathers
`rel_logits[global_q_idx, h_idx, rel_idx]`. For the padded rows of the last
query tile, `global_q_idx` exceeds `total_q - 1`. The read is out of bounds;
whether it *faults* depends on allocation layout — hence the
nondeterminism (~37% on a `(6,512)` two-seq batch).

**Proof chain (all harnesses in `tests/`):**
- `fa4_fuzz.py`: 200 randomized serving shapes; crash envelope is `b>=2`,
  `q_len>1` batches; single-seq never crashes.
- Padding `rel_logits` by 8192 rows → 8/8 crash-free (pad-out and pad-block-table
  controls don't stabilize it).
- Clamping `global_q_idx` to `rel_logits.shape[0]-1` → the (6,512) class goes
  8/8 clean.
- `fa4_numeric_check.py`: batched vs per-sequence outputs are **bit-exact**
  (rel err 0.0000) for every valid row → padding-row values are provably
  discarded, so clamping cannot change real outputs.

**Fix:** the shim replaces `_get_score_mod` with a clamped variant
(`_patch_inkling_fa4_op`). One `min()` — mirrors the existing `rel_idx` clamp.

## Bug 2 — the SM120 varlen paged kernel still faults for b>=3 global-window batches

**Symptom:** with bug 1 fixed, `cudaErrorIllegalAddress` persists for b>=3
**global-window** batches containing prefill-length queries (max_q 128–1024).
SWA-window batches, all-decode batches (all `q_len==1`), and `b=1` calls never
fault (200-case fuzz: 24 crashes, 100% in that envelope, 0 outside it).

**Controlled experiments that did NOT fix it:** `num_stages` 2→1 (the #2348
double-buffered paged pipeline); select-after-load clamps in
`paged_kv.py::load_page_table` and `flash_fwd.py::_paged_load_page_table_and_ptrs`
(the `mPageTable[page_idx] if is_valid else 0` load executes even when
`is_valid` is false — a real upstream hazard worth fixing anyway; sentinel-verified
live, crash rate unchanged). The residual fault is deeper in the kernel's
global-window varlen scheduling.

**Fix (shipped): batch-split.** The shim's dispatch splits any batch with
`max_seqlen_q > 1` and `b > 1` into per-sequence (`b=1`) kernel calls — the
proven-clean envelope — and writes each result into the caller's `out` slice.
All-decode batches (the steady-state serving shape) stay batched at full speed.
200/200 fuzz cases clean; numerics remain bit-exact. Cost: prefill-heavy steps
issue `b` kernel launches per layer instead of 1 (measured: single-digit %
end-to-end; decode throughput unaffected). Kill switch: `FA4_SM120_BATCH_SPLIT=0`.

## What this costs / what to watch

- cute-dsl JIT compiles per kernel shape bucket; the bundle's compile cache is
  in-memory → first request(s) after each boot pay a one-time JIT (~seconds).
- `enable_cutedsl_warmup=False` in the serve argv: boot-time warmup over all
  shape buckets spun 30+ min on SM121; lazy compile is faster end-to-end.
- `--enforce-eager` is REQUIRED: piecewise cudagraphs crash on cross-node
  collectives (vLLM #46253 class) for this model.
- Lamport fused sconv collective is disabled by design on this fabric (it
  requires MNNVL; DGX Spark is RoCE) — both ranks deterministically take the
  NCCL fallback. That path is healthy (3000+ collectives in boot logs).

## Reporting upstream

The evidence package for both issues is in `docs/UPSTREAM-ISSUES.md`
(vLLM score_mod clamp; bundle varlen kernel). If upstream lands fixes, delete
the corresponding shim patch and re-run `tests/fa4_fuzz.py` as the gate.
