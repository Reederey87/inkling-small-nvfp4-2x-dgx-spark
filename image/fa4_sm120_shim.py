"""
Force vLLM on SM12x (DGX Spark GB10 / RTX 50 / RTX PRO Blackwell) to use the
SecondNatureComputing/flash-attn-4-sm120 HF kernel for the FA4 path.

Source: https://hackmd.io/uT8okJYPTIalj4vSTcr5oQ (verbatim shim, 2026-07-30).
Why: vLLM's bundled vllm_flash_attn/cute/interface.py gates SM12x with
  assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
and vLLM always serves paged KV. Inkling's relative attention hard-requires the
FA4 cute path (score_mod + aux_tensors), so without this shim the model cannot
boot on SM12x at all. The HF kernel bundles upstream flash-attention PRs #2348
(SM120 paged KV) and #2336 (SM120 split-KV), validated on SM121a (DGX Spark).

Disable by setting env var FA4_SM120_SHIM=0 before launching.
"""
from __future__ import annotations

import os
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder

if os.environ.get("FA4_SM120_SHIM", "1") != "0":

    _HF_KERNEL = None

    def _hf_kernel():
        # Load the SM120 FA4 bundle WITHOUT the `kernels` package: kernels>=0.4
        # resolves repos as repo_type="kernel" (HF's new registry) and 401s on this
        # model-type repo. Plain snapshot_download (model repo, offline-safe once
        # cached) + importlib file loading is version-independent.
        # FA4_SM120_BUNDLE_DIR overrides the source tree (e.g. a locally patched
        # copy — the HF cache blobs are root-owned read-only inside the container).
        global _HF_KERNEL
        if _HF_KERNEL is None:
            import importlib.util
            import os
            override = os.environ.get("FA4_SM120_BUNDLE_DIR")
            if override:
                pkg_dir = override
            else:
                from huggingface_hub import snapshot_download
                snap = snapshot_download(
                    "SecondNatureComputing/flash-attn-4-sm120",
                    revision="60117041e10fcc6f19882afd274318c755a5ef6e")
                pkg_dir = os.path.join(snap, "build", "torch-cuda")
            # cutlass-dsl 4.6.0 compat: the bundle (Mar-2026, written for ~4.4.x) uses
            # `cute.core.ThrMma` (29 sites) and `cute.make_fragment` (moved to
            # make_fragment_like — accepts Layout+dtype, exact superset). Verified these
            # are the only two cute.* names the bundle uses that 4.6.0 lacks (83 scanned).
            import cutlass.cute
            import cutlass.cute.core
            if not hasattr(cutlass.cute.core, "ThrMma") and hasattr(cutlass.cute, "ThrMma"):
                cutlass.cute.core.ThrMma = cutlass.cute.ThrMma
            if not hasattr(cutlass.cute, "make_fragment"):
                def _make_fragment(src, dtype=None, **kw):
                    # 4.6.0's make_fragment_like only takes Layout/ComposedLayout/Tensor;
                    # the old make_fragment also accepted tuple/list/int shapes.
                    from cutlass.cute.typing import ComposedLayout, Layout, Tensor
                    if not isinstance(src, (Layout, ComposedLayout, Tensor)):
                        src = cutlass.cute.make_layout(src)
                    return cutlass.cute.make_fragment_like(src, dtype, **kw)
                cutlass.cute.make_fragment = _make_fragment
            if pkg_dir not in sys.path:
                sys.path.insert(0, pkg_dir)
            spec = importlib.util.spec_from_file_location(
                "fa4_sm120_bundle",
                os.path.join(pkg_dir, "__init__.py"),
                submodule_search_locations=[pkg_dir])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["fa4_sm120_bundle"] = mod
            spec.loader.exec_module(mod)
            importlib.import_module("fa4_sm120_bundle.interface")
            # DSL 4.6.0 strict typing: the bundle internally reassigns window_size_*
            # to plain Python ints (e.g. causal → right=0), but its cute.compile /
            # launch params declare Int32|None (older DSL auto-coerced). Patch the
            # single choke point that produces those values. Verified Int32(0) is
            # falsy, so the SWA load-bound truthiness math keeps working.
            from cutlass.base_dsl.typing import Int32
            iface = mod.interface
            _orig_resolve = iface._resolve_causal_local_window
            def _resolve_typed(causal, wsl, wsr, mask_mod=None):
                causal, local, wsl, wsr = _orig_resolve(causal, wsl, wsr, mask_mod)
                conv = lambda v: Int32(v) if isinstance(v, int) and not isinstance(v, bool) else v
                return causal, local, conv(wsl), conv(wsr)
            iface._resolve_causal_local_window = _resolve_typed
            _HF_KERNEL = mod
        return _HF_KERNEL

    _PATCHED: set[str] = set()

    def _is_sm12x() -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            major, _ = torch.cuda.get_device_capability()
            return major == 12
        except Exception:
            return False

    def _patch_fa_iface(mod):
        def _is_fa4_supported_patched():
            if not getattr(mod, "FA4_AVAILABLE", False):
                return False, getattr(mod, "FA4_UNAVAILABLE_REASON", "FA4 unavailable")
            try:
                import torch
                major, _ = torch.cuda.get_device_capability()
            except Exception:
                return False, "no CUDA device"
            if major in (9, 10, 11, 12):
                return True, None
            return False, f"FA4 not supported on capability {major}.x"
        mod._is_fa4_supported = _is_fa4_supported_patched

    def _patch_cute_iface(mod):
        import inspect
        orig = mod._flash_attn_fwd
        bundle_params = {"_sentinel": None}

        def _split_per_seq(fn, args, kwargs):
            # DSPARK-FIX #2 (upstream-reportable): the bundle's varlen paged kernel
            # still faults (cudaErrorIllegalAddress, nondeterministic) for b>=3
            # global-window batches containing prefill-length queries — even with
            # the clamped score_mod. b=1 calls and all-decode batches never fault
            # (200-case fuzz, tests/fa4_fuzz.py), and batched vs per-seq outputs are
            # bit-exact (tests/fa4_numeric_check.py), so run prefill-containing
            # batches as per-seq calls. Decode batches (all q_len==1) stay batched.
            import torch
            q, k, v = args[0], args[1], args[2]
            cu = kwargs["cu_seqlens_q"]
            try:
                bounds = cu.tolist()
            except Exception:
                # FakeTensor compile path (warmup): trace-only, no real access.
                return fn(*args, **kwargs)
            out = kwargs.get("out")
            aux = kwargs.get("aux_tensors")
            pt = kwargs.get("page_table")
            sk = kwargs.get("seqused_k")
            last = None
            for i in range(len(bounds) - 1):
                s, e = bounds[i], bounds[i + 1]
                kw = dict(kwargs)
                sub_args = (q[s:e], k, v)
                cu_i = torch.tensor([0, e - s], device=cu.device, dtype=cu.dtype)
                kw["cu_seqlens_q"] = cu_i
                if out is not None:
                    kw["out"] = out[s:e]
                if aux is not None:
                    kw["aux_tensors"] = [t[s:e] for t in aux]
                if pt is not None:
                    kw["page_table"] = pt[i:i + 1]
                if sk is not None:
                    kw["seqused_k"] = sk[i:i + 1]
                last = fn(*sub_args, **kw)
            # vLLM's caller writes through the out= it passed; the per-seq calls
            # already filled every slice. Return the full-batch out so the
            # tuple-adapt below hands back the right tensor.
            if out is not None:
                return (out, None)
            return last

        def _dispatch(*args, **kwargs):
            if _is_sm12x():
                fn = _hf_kernel().interface._flash_attn_fwd
                if bundle_params.get("_sentinel") is None:
                    sig = inspect.signature(fn)
                    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                                     for p in sig.parameters.values())
                    bundle_params["_sentinel"] = None if has_var_kw else set(sig.parameters)
                allowed = bundle_params["_sentinel"]
                if allowed is not None:
                    dropped = [k for k in kwargs if k not in allowed]
                    if dropped:
                        warnings.warn(f"[fa4_sm120_shim] dropping unsupported kwargs: {dropped}")
                    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                cu = kwargs.get("cu_seqlens_q")
                msq = kwargs.get("max_seqlen_q") or 1
                if (cu is not None and getattr(cu, "numel", lambda: 0)() > 2
                        and msq > 1
                        and os.environ.get("FA4_SM120_BATCH_SPLIT", "1") == "1"):
                    ret = _split_per_seq(fn, args, kwargs)
                else:
                    ret = fn(*args, **kwargs)
                # Return-shape adapt: vendored callers unpack (out, lse, p, row_max);
                # the bundle returns (out, lse). Pad the fwd-only extras with None.
                if isinstance(ret, tuple):
                    if len(ret) == 2:
                        return (ret[0], ret[1], None, None)
                    return ret
                return (ret, None, None, None)
            return orig(*args, **kwargs)

        mod._flash_attn_fwd = _dispatch

    def _patch_fa_utils(mod):
        import functools
        orig = mod.get_flash_attn_version

        @functools.wraps(orig)
        def patched(requires_alibi: bool = False,
                    head_size: int | None = None,
                    head_size_v: int | None = None,
                    has_sinks: bool = False):
            if not _is_sm12x():
                return orig(requires_alibi=requires_alibi,
                            head_size=head_size,
                            head_size_v=head_size_v,
                            has_sinks=has_sinks)
            if requires_alibi:
                return 2
            if head_size is not None and head_size > 128:
                return 2
            try:
                from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
                if is_fa_version_supported(4):
                    return 4
            except Exception:
                pass
            return 2

        mod.get_flash_attn_version = patched

    def _patch_inkling_fa4_op(mod):
        # DSPARK-FIX (upstream-reportable): on SM12x the bundle kernel passes a
        # q_idx that, summed with seqlen_info.offset_q, can address PADDED tile
        # rows beyond total_q — the rel_logits gather in the score_mod then
        # reads out of bounds and faults intermittently with
        # cudaErrorIllegalAddress on b>=2 varlen batches (valid-row numerics are
        # bit-exact vs per-seq calls — verified by tests/fa4_numeric_check.py —
        # so clamping only touches discarded padding rows). Clamp the gather
        # index exactly like the existing rel_idx clamp.
        from functools import cache

        @cache
        def _get_score_mod_clamped(rel_extent: int):
            import cutlass.cute as cute
            from cutlass.cute import Float32

            from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

            @cute.jit
            def score_mod_rel_bias_clamped(
                scores: cute.TensorSSA,
                b_idx: cute.TensorSSA,
                h_idx: cute.TensorSSA,
                q_idx: cute.TensorSSA,
                kv_idx: cute.TensorSSA,
                seqlen_info: SeqlenInfoQK,
                aux_tensors: list[cute.Tensor],
            ) -> cute.TensorSSA:
                rel_logits = aux_tensors[0]

                seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
                rel_dist = (q_idx + seqlen_local_offset) - kv_idx
                global_q_idx = seqlen_info.offset_q + q_idx

                rel_dist_0 = rel_dist[0]
                rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
                rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

                n_rows = rel_logits.shape[0]
                gq_0 = global_q_idx[0]
                gq_0 = gq_0 if gq_0 >= 0 else 0
                gq_0 = gq_0 if gq_0 < n_rows else (n_rows - 1)

                rel_bias = rel_logits[gq_0, h_idx[0], rel_idx]
                rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
                return scores + rel_bias

            return score_mod_rel_bias_clamped

        mod._get_score_mod = _get_score_mod_clamped

    _DISPATCH = {
        "vllm.vllm_flash_attn.flash_attn_interface": _patch_fa_iface,
        "vllm.vllm_flash_attn.cute.interface":      _patch_cute_iface,
        "vllm.v1.attention.backends.fa_utils":      _patch_fa_utils,
        "vllm.models.inkling.nvidia.ops.fa4_rel_attention": _patch_inkling_fa4_op,
    }

    def _try_patch(name: str):
        if name in _PATCHED or name not in _DISPATCH:
            return
        mod = sys.modules.get(name)
        if mod is None:
            return
        try:
            _DISPATCH[name](mod)
            _PATCHED.add(name)
        except Exception as e:
            warnings.warn(f"[fa4_sm120_shim] failed to patch {name}: {e!r}")

    class _WrappedLoader(Loader):
        def __init__(self, real, name):
            self._real = real
            self._name = name

        def create_module(self, spec):
            if hasattr(self._real, "create_module"):
                return self._real.create_module(spec)
            return None

        def exec_module(self, module):
            self._real.exec_module(module)
            _try_patch(self._name)

    class _PatchingFinder(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in _DISPATCH or name in _PATCHED:
                return None
            for finder in list(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _WrappedLoader(spec.loader, name)
                    return spec
            return None

    if not getattr(sys, "_fa4_sm120_shim_installed", False):
        sys.meta_path.insert(0, _PatchingFinder())
        try:
            sys._fa4_sm120_shim_installed = True  # type: ignore[attr-defined]
        except Exception:
            pass

    for _n in list(_DISPATCH):
        _try_patch(_n)
