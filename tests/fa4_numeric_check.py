# Numeric equivalence: batched varlen call vs per-seq (b=1) calls through the
# SM120 shim. b=1 is the bundle's validated envelope; if batched outputs differ
# beyond bf16 noise, the kernel's score_mod/aux indexing is semantically broken
# for b>1 and the serving fix must be batch-splitting, not bounds-clamping.
import sys

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q, inkling_fa4_num_splits, inkling_fa4_rel_attention)

D, REL_EXTENT, PAGE, DEV = 128, 1024, 128, "cuda"
HQ, HKV = 16, 4
torch.manual_seed(1234)


def run_batch(q_lens, kv_lens, window):
    b = len(q_lens)
    total_q = sum(q_lens)
    q = torch.randn(total_q, HQ, D, device=DEV, dtype=torch.bfloat16)
    npages = sum((l + PAGE - 1) // PAGE for l in kv_lens) + 8
    kc = torch.randn(npages, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    maxp = max((l + PAGE - 1) // PAGE for l in kv_lens)
    bt = torch.zeros(b, maxp, device=DEV, dtype=torch.int32)
    off = 0
    perm = torch.randperm(npages, device=DEV)
    for i, l in enumerate(kv_lens):
        p = (l + PAGE - 1) // PAGE
        bt[i, :p] = perm[off:off + p].to(torch.int32)
        off += p
    sl = torch.tensor(kv_lens, device=DEV, dtype=torch.int32)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)),
                      device=DEV, dtype=torch.int32)
    rel = torch.randn(total_q, HQ, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    mq = bucket_max_seqlen_q(max(q_lens))

    # batched
    out_b = torch.empty(total_q, HQ, D, device=DEV, dtype=torch.bfloat16)
    ns = inkling_fa4_num_splits(is_local=window != (-1, -1), batch_size=b,
                                max_query_len=mq, num_heads=HQ, num_kv_heads=HKV,
                                max_kv_len=131072)
    inkling_fa4_rel_attention(
        q, kc, vc, block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
        max_seqlen_q=mq, softmax_scale=D ** -0.5, causal=True, window_size=window,
        rel_extent=REL_EXTENT, rel_logits=rel, num_splits=ns, out=out_b)
    torch.cuda.synchronize()

    # per-seq (gold)
    outs = []
    for i in range(b):
        qs, ks = q_lens[i], kv_lens[i]
        p = (ks + PAGE - 1) // PAGE
        out_i = torch.empty(qs, HQ, D, device=DEV, dtype=torch.bfloat16)
        ns_i = inkling_fa4_num_splits(is_local=window != (-1, -1), batch_size=1,
                                      max_query_len=bucket_max_seqlen_q(qs),
                                      num_heads=HQ, num_kv_heads=HKV, max_kv_len=131072)
        inkling_fa4_rel_attention(
            q[cu[i]:cu[i + 1]], kc, vc,
            block_table=bt[i:i + 1], cache_seqlens=sl[i:i + 1],
            cu_seqlens_q=torch.tensor([0, qs], device=DEV, dtype=torch.int32),
            max_seqlen_q=bucket_max_seqlen_q(qs), softmax_scale=D ** -0.5,
            causal=True, window_size=window, rel_extent=REL_EXTENT,
            rel_logits=rel[cu[i]:cu[i + 1]], num_splits=ns_i, out=out_i)
        outs.append(out_i)
    torch.cuda.synchronize()
    out_g = torch.cat(outs, 0)
    diff = (out_b.float() - out_g.float()).abs()
    rel_err = (diff.max() / (out_g.float().abs().max() + 1e-6)).item()
    print(f"q={q_lens} kv={kv_lens} win={window}: max|d|={diff.max().item():.4f} "
          f"rel={rel_err:.4f} {'MATCH' if rel_err < 0.02 else 'MISMATCH'}", flush=True)
    return rel_err


cases = [
    ([1, 1, 1, 1], [10, 20, 30, 40], (-1, -1)),   # decode batch global
    ([1, 1, 1, 1], [10, 20, 30, 40], (511, 0)),  # decode batch SWA
    ([6, 512], [6, 512], (-1, -1)),              # the crashing case
    ([6, 512], [6, 512], (511, 0)),
    ([100, 512], [100, 512], (-1, -1)),
    ([1, 1024], [5000, 1024], (-1, -1)),
]
worst = 0.0
failures = 0
for ql, kl, w in cases:
    try:
        r = run_batch(ql, kl, w)
        worst = max(worst, r)
        if r >= 0.02:
            failures += 1
    except Exception as e:
        failures += 1
        print(f"q={ql} kv={kl} win={w}: EXCEPTION {type(e).__name__}", flush=True)
print(f"WORST rel err: {worst:.4f}  failures: {failures}")
sys.exit(1 if failures else 0)
