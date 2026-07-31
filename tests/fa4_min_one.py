# Minimal single-case driver for the Inkling FA4 relative-attention op through
# the SM120 shim — the repro shape for the varlen OOB (docs/UPSTREAM-ISSUES.md).
# Usage: fa4_min_one.py <q_lens_csv> <kv_lens_csv> <g|s>   (g=global, s=SWA-512)
import sys

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q, inkling_fa4_num_splits, inkling_fa4_rel_attention)

D, REL_EXTENT, PAGE, DEV = 128, 1024, 128, "cuda"
HQ, HKV = 16, 4  # TP=2 per-rank shard (use 32/8 for full model)
q_lens = [int(x) for x in sys.argv[1].split(",")]
kv_lens = [int(x) for x in sys.argv[2].split(",")]
window = (-1, -1) if sys.argv[3] == "g" else (511, 0)
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
out = torch.empty(total_q, HQ, D, device=DEV, dtype=torch.bfloat16)
mq = bucket_max_seqlen_q(max(q_lens))
ns = inkling_fa4_num_splits(is_local=window != (-1, -1), batch_size=b,
                            max_query_len=mq, num_heads=HQ, num_kv_heads=HKV,
                            max_kv_len=131072)
print(f"LAUNCHED q={q_lens} kv={kv_lens} win={window} mq={mq} ns={ns}", flush=True)
inkling_fa4_rel_attention(
    q, kc, vc, block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
    max_seqlen_q=mq, softmax_scale=D ** -0.5, causal=True, window_size=window,
    rel_extent=REL_EXTENT, rel_logits=rel, num_splits=ns, out=out)
torch.cuda.synchronize()
print("OK finite:", torch.isfinite(out).all().item(), flush=True)
