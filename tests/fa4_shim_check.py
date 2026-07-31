# End-to-end proof: Inkling's FA4 relative-attention op through the SM120 shim.
# Mirrors the model's real call (paged KV + score_mod + aux_tensors) on GPU.
import torch
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import inkling_fa4_rel_attention

B, S, Hq, Hkv, D = 1, 512, 32, 8, 128
page_size = 128
npages = B * ((S + page_size - 1) // page_size)
q = torch.randn(B * S, Hq, D, device="cuda", dtype=torch.bfloat16)
kc = torch.randn(npages, page_size, Hkv, D, device="cuda", dtype=torch.bfloat16)
vc = torch.randn_like(kc)
bt = torch.arange(npages, device="cuda", dtype=torch.int32).view(B, -1)
sl = torch.tensor([S], device="cuda", dtype=torch.int32)
cu = torch.tensor([0, B * S], device="cuda", dtype=torch.int32)
rel = torch.randn(B * S, Hq, 1024, device="cuda", dtype=torch.float32)

out = inkling_fa4_rel_attention(
    q, kc, vc,
    block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
    max_seqlen_q=B * S, softmax_scale=D ** -0.5, causal=True,
    window_size=(-1, -1), rel_extent=1024, rel_logits=rel,
)
finite = torch.isfinite(out).all().item()
print("INKLING FA4 REL ATTN OK", tuple(out.shape), out.dtype, "finite:", finite)

# Second call with a sliding window (35 of 42 layers are SWA-512) — different code path.
out2 = inkling_fa4_rel_attention(
    q, kc, vc,
    block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu,
    max_seqlen_q=B * S, softmax_scale=D ** -0.5, causal=True,
    window_size=(511, 0), rel_extent=1024, rel_logits=rel,
)
print("SWA VARIANT OK", tuple(out2.shape), "finite:", torch.isfinite(out2).all().item())
