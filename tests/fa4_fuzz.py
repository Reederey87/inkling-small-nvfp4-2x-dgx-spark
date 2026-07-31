# Randomized fuzzer for the Inkling FA4 SM120 serving crash (cudaErrorIllegalAddress
# 700 in the bundle's _flash_attn_fwd via compile_cache). Uses the REAL serving
# bucketing (bucket_max_seqlen_q) and num_splits formula (inkling_fa4_num_splits)
# imported from vLLM, TP-sharded head counts by default, and serving-style paged
# metadata. Runs cases in-process until a fault poisons the CUDA context; the
# parent loop (fa4_fuzz_run.sh) restarts after each crash and resumes at the
# next case — the case that killed the process is the repro.
import argparse
import random
import sys
import time

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)

D, REL_EXTENT, PAGE = 128, 1024, 128
DEV = "cuda"


def one_case(idx, hq, hkv, rng):
    # serving-realistic batch shapes: decode-only, prefill-only, or mixed
    style = rng.choice(["decode", "prefill", "mixed"])
    b = rng.randint(1, 8)
    if style == "decode":
        q_lens = [1] * b
        kv_lens = [rng.choice([1, 7, 100, 511, 512, 513, 1000, 4096, 30000, 131072])
                   for _ in range(b)]
    elif style == "prefill":
        q_lens = [rng.choice([2, 6, 17, 100, 511, 512, 777, 1024]) for _ in range(b)]
        kv_lens = q_lens  # fresh prefill: seq == q
    else:  # mixed decode + chunked-prefill continuation
        q_lens = [1] * (b // 2) + [rng.choice([128, 511, 1024]) for _ in range(b - b // 2)]
        kv_lens = ([rng.choice([1, 100, 4096, 30000]) for _ in range(b // 2)]
                   + [q + rng.choice([0, 1024, 4096]) for q in q_lens[b // 2:]])
    total_q = sum(q_lens)
    if total_q > 8192:
        return None
    window = rng.choice([(-1, -1), (511, 0)])
    is_local = window != (-1, -1)
    max_q = bucket_max_seqlen_q(max(q_lens))           # serving buckets to pow2
    num_splits = inkling_fa4_num_splits(
        is_local=is_local, batch_size=b, max_query_len=max_q,
        num_heads=hq, num_kv_heads=hkv, max_kv_len=131072)

    q = torch.randn(total_q, hq, D, device=DEV, dtype=torch.bfloat16)
    npages = sum((l + PAGE - 1) // PAGE for l in kv_lens) + 8
    kc = torch.randn(npages, PAGE, hkv, D, device=DEV, dtype=torch.bfloat16)
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
    rel = torch.randn(total_q, hq, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(total_q, hq, D, device=DEV, dtype=torch.bfloat16)

    desc = (f"case={idx} style={style} b={b} q={q_lens} kv={kv_lens} "
            f"win={window} max_q={max_q} splits={num_splits} hq={hq} hkv={hkv}")
    print(f"LAUNCHED {desc}", flush=True)
    inkling_fa4_rel_attention(
        q, kc, vc,
        block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
        max_seqlen_q=max_q, softmax_scale=D ** -0.5, causal=True,
        window_size=window, rel_extent=REL_EXTENT, rel_logits=rel,
        num_splits=num_splits, out=out,
    )
    torch.cuda.synchronize()
    fin = torch.isfinite(out).all().item()
    print(f"OK case={idx} finite={fin}", flush=True)
    return 0 if fin else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hkv", type=int, default=4)
    args = ap.parse_args()
    rc = 0
    for idx in range(args.start, args.n):
        rng = random.Random(args.seed + idx)  # per-case seed → exact crash resume
        r = one_case(idx, args.hq, args.hkv, rng)
        if r is None:
            continue
        rc = rc or r
    return rc


if __name__ == "__main__":
    sys.exit(main())
