# Repro harness for the Inkling serving stall: drives the FA4 relative-attention
# op through the SM120 shim with SERVING shapes (not just the prefill smoke
# shape fa4_shim_check.py covers). One case per process; the parent wraps each
# case in `timeout`, so a hung kernel surfaces as exit 124 with the case name
# as the last LAUNCHED line. Compile (JIT) happens on first call per key and
# is CPU-bound; a real kernel hang is GPU-bound and never returns from sync.
import argparse
import sys
import time

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import inkling_fa4_rel_attention

HQ, HKV, D, REL_EXTENT, PAGE = 32, 8, 128, 1024, 128
DEV = "cuda"
# TP=2 serving runs each rank on its head SHARD: --hq 16 --hkv 4 reproduces the
# per-rank kernel config (GQA ratio 4 preserved; bundle's pack_gqa workaround
# path differs at Hkv=4 vs 8).


def mk_paged(kv_lens, npages_pool=4096):
    """Realistic paged KV: shuffled physical pages, rows padded with 0."""
    pages_per = [(l + PAGE - 1) // PAGE for l in kv_lens]
    need = sum(pages_per)
    assert need <= npages_pool, f"need {need} pages > pool {npages_pool}"
    kc = torch.randn(npages_pool, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    perm = torch.randperm(npages_pool, device=DEV)[:need]
    maxp = max(pages_per)
    bt = torch.zeros(len(kv_lens), maxp, device=DEV, dtype=torch.int32)
    off = 0
    for i, p in enumerate(pages_per):
        bt[i, :p] = perm[off:off + p].to(torch.int32)
        off += p
    sl = torch.tensor(kv_lens, device=DEV, dtype=torch.int32)
    return kc, vc, bt, sl


def run(name, q_lens, kv_lens, window, num_splits, iters=3, rel_dtype=torch.bfloat16):
    total_q = sum(q_lens)
    q = torch.randn(total_q, HQ, D, device=DEV, dtype=torch.bfloat16)
    kc, vc, bt, sl = mk_paged(kv_lens)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)),
                      device=DEV, dtype=torch.int32)
    rel = torch.randn(total_q, HQ, REL_EXTENT, device=DEV, dtype=rel_dtype)
    out = torch.empty(total_q, HQ, D, device=DEV, dtype=torch.bfloat16)
    msq = max(q_lens)
    print(f"LAUNCHED {name} q={q_lens} kv={kv_lens} win={window} "
          f"splits={num_splits} rel={rel_dtype}", flush=True)
    for i in range(iters):
        t0 = time.time()
        inkling_fa4_rel_attention(
            q, kc, vc,
            block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
            max_seqlen_q=msq, softmax_scale=D ** -0.5, causal=True,
            window_size=window, rel_extent=REL_EXTENT, rel_logits=rel,
            num_splits=num_splits, out=out,
        )
        torch.cuda.synchronize()  # a hung kernel never returns from here
        dt = time.time() - t0
        fin = torch.isfinite(out).all().item()
        print(f"  iter {i}: {dt:.2f}s finite={fin}", flush=True)
        if not fin:
            print(f"NONFINITE {name}", flush=True)
            return 2
    print(f"OK {name}", flush=True)
    return 0


# (name, q_lens, kv_lens, window, num_splits) — num_splits mirrors what
# inkling_fa4_num_splits() computes at serving for these shapes
# (Hq=32, Hkv=8, max_kv_len=131072).
CASES = {
    # tiny first-request prefill + first decode (the stall happened on req #1)
    "prefill-tiny-global":  ([6], [6], (-1, -1), 1),
    "decode-tiny-global":   ([1], [7], (-1, -1), 1),
    "decode-tiny-swa":      ([1], [7], (511, 0), 1),
    # decode, 1 seq: short ctx (no split) vs 4k/128k ctx (split-KV 16)
    "decode-1s-short-global": ([1], [100], (-1, -1), 1),
    "decode-1s-4k-global":    ([1], [4096], (-1, -1), 16),
    "decode-1s-128k-global":  ([1], [131072], (-1, -1), 16),
    "decode-1s-4k-swa":       ([1], [4096], (511, 0), 1),
    # decode, full batch of 8, mixed ctx lengths (continuous batching)
    "decode-8s-mixed-global": ([1] * 8, [10, 100, 500, 1000, 2048, 4096, 8192, 32768],
                              (-1, -1), 4),
    "decode-8s-mixed-swa":    ([1] * 8, [10, 100, 500, 1000, 2048, 4096, 8192, 32768],
                              (511, 0), 1),
    # chunked prefill at MAX_NUM_BATCHED_TOKENS=1024, both layer types
    "prefill-1024-global": ([1024], [1024], (-1, -1), 1),
    "prefill-1024-swa":    ([1024], [1024], (511, 0), 1),
    "prefill-1024-2nd-chunk-global": ([1024], [2048], (-1, -1), 1),
    # mixed prefill+decode in one varlen batch (continuous batching step)
    "mixed-decode-prefill-global": ([1, 1024], [5000, 1024], (-1, -1), 1),
    "mixed-decode-prefill-swa":    ([1, 1024], [5000, 1024], (511, 0), 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    help="case name, 'all', or comma list; 'list' prints names")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rel-fp32", action="store_true",
                    help="use fp32 rel_logits (fa4_shim_check used fp32; serving is bf16)")
    ap.add_argument("--hq", type=int, default=32, help="query heads (16 = TP2 shard)")
    ap.add_argument("--hkv", type=int, default=8, help="KV heads (4 = TP2 shard)")
    args = ap.parse_args()
    if args.case == "list":
        print("\n".join(CASES))
        return 0
    global HQ, HKV
    HQ, HKV = args.hq, args.hkv
    names = list(CASES) if args.case == "all" else args.case.split(",")
    rel_dtype = torch.float32 if args.rel_fp32 else torch.bfloat16
    rc = 0
    for n in names:
        q_lens, kv_lens, win, ns = CASES[n]
        r = run(n, q_lens, kv_lens, win, ns, iters=args.iters, rel_dtype=rel_dtype)
        rc = rc or r
    return rc


if __name__ == "__main__":
    sys.exit(main())
