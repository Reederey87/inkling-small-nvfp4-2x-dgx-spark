# Upstream bug reports — FA4 SM120 varlen paged faults (2026-07-31)

Evidence package for two issues to file. Repro harnesses live in `tests/`
(fa4_fuzz.py, fa4_min_one.py, fa4_numeric_check.py, fa4_serving_shapes.py).

## Issue 1 — vLLM: Inkling FA4 score_mod reads aux tensor OOB on padded varlen rows

**Repo:** vllm-project/vllm (models/inkling/nvidia/ops/fa4_rel_attention.py)
**Summary:** `score_mod_rel_bias` gathers `rel_logits[global_q_idx]` where
`global_q_idx = seqlen_info.offset_q + q_idx`. On b>=2 varlen batches the index
addresses padded tile rows beyond `total_q` → intermittent
`cudaErrorIllegalAddress` (700) on SM12x (and potentially latent elsewhere).
**Evidence:** randomized fuzzer (200 serving-realistic cases): ~12% crash rate
for b>=2 global-window batches, b=1 never; oversizing `rel_logits` by 8192 rows
makes crashes vanish (8/8); clamping `global_q_idx` to `rel_logits.shape[0]-1`
in the score_mod makes the (6,512) crash class 8/8 clean; batched vs per-seq
outputs are **bit-exact** for all valid rows, so padding-row values are provably
discarded.
**Fix:** clamp `global_q_idx` exactly like the existing `rel_idx` clamp
(implemented as `DSPARK-FIX` in our shim; see image/fa4_sm120_shim.py
`_patch_inkling_fa4_op`).

## Issue 2 — SecondNatureComputing/flash-attn-4-sm120: SM120 varlen paged kernel
## faults for b>=3 global-window batches with prefill queries

**Repo:** huggingface.co/SecondNatureComputing/flash-attn-4-sm120 (rev 60117041)
**Summary:** with the score_mod gather clamped (issue 1 fixed), the SM120
non-TMA paged kernel (`FlashAttentionForwardSm120`, num_stages=1|2 alike) still
faults intermittently (cudaErrorIllegalAddress) for b>=3 **global-window**
varlen batches that contain prefill-length queries. SWA-window batches and
all-decode batches (q_len==1) never fault. b=1 never faults.
**Envelope (200-case fuzz, 24 crashes):** all `window=(-1,-1)`, b>=3 (two b=2
cases covered by issue 1), mixed decode+prefill or multi-prefill, max_q 128–1024,
num_splits=1, Hq=16/Hkv=4/D=128 (also reproduces at Hq=32/Hkv=8).
**Not the cause (controlled experiments):** num_stages 2→1 unchanged; page-table
select-after-load clamps in `paged_kv.py`/`flash_fwd.py
(_paged_load_page_table_and_ptrs)` applied and sentinel-verified live — rate
unchanged. Non-deterministic per identical shape (~37% on (6,512) pre-clamp),
suggesting an uninitialized/over-read whose fault depends on allocation layout.
**Workaround shipped:** batch-split prefill-containing batches into per-seq
(b=1) calls in the dispatch shim — 200/200 fuzz cases clean; numerics bit-exact.
**Repro:** tests/fa4_fuzz.py (seed 20260731); minimal: q=[6,512] kv=[6,512]
global window, paged, score_mod+aux — ~3/8 crash rate pre-clamp.

## Issue 3 — vLLM: inkling streaming tool parser leaks markup as content (agent-shaped requests)

**Repo:** vllm-project/vllm (vllm/parser/inkling.py engine + streaming path)
**Summary:** with `--tool-call-parser inkling` on an Inkling model, the
**streaming** chat-completions path returns the model's
`<|content_invoke_tool_json|>{"name":...,"args":{...}}` markup as plain content
(`finish_reason=stop`) instead of parsing it into `tool_calls`, for
agent-shaped requests (large system prompt ~26K chars, 30+ tool schemas).
The **non-streaming** path parses the identical payload correctly every time.
**Repro:** stream=True leaks 4/4; stream=False parses 6/6 (byte-identical
request otherwise; repro payload = Hermes agent's real request: 32 tools,
26K system prompt, no tool_choice, max_tokens 65536). Also reproduces with
text-before-call in streaming; non-streaming handles every tested shape
(text-before-call, 32 tools, tool_choice auto/required/absent).
**Impact:** OpenAI-compatible agents that hardcode `stream: true` (Hermes and
others) see raw markup and stall; the model itself is behaving correctly.
**Workaround (shipped in this repo):** de-streaming proxy
(`proxy/destream_proxy.py`) that strips `stream` before forwarding.
