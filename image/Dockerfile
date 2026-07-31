# Thin overlay on the official arm64 v0.26.0 image. Additions:
#   1. scipy — the Inkling model class imports scipy.optimize.linear_sum_assignment
#      at construction (stock image lacks it; first boot died with ModuleNotFoundError).
#   2. fa4_sm120_shim (.py + .pth autoload) — redirects vLLM's bundled FA4 cute
#      _flash_attn_fwd to SecondNatureComputing/flash-attn-4-sm120 on SM12x (upstream
#      PRs #2348 paged KV / #2336 split-KV, validated on SM121a). Without it the boot
#      dies at cute-dsl warmup: "Paged KV not supported on SM 12.0 in this PR".
#      einops/apache-tvm-ffi are the bundle's declared python-depends (build.toml).
#      The bundle is loaded via huggingface_hub.snapshot_download (pinned revision) —
#      NOT the `kernels` package, whose >=0.4 loader 401s on this model-type repo.
#   3. nvidia-nccl-cu13==2.30.7 — the stock image ships 2.28.9, which intermittently
#      hangs cross-node collectives on this RoCE fabric under serving load
#      (#41725 class: workers wedge inside ncclAllGather, streams spin at 96% GPU).
#      The reference 2x-Spark production stack runs 2.30.7 daily without a hang.
#   4. py-spy — live-stack forensics for exactly the hang class above (not in the
#      stock image; docker exec needs --privileged for SYS_PTRACE).
# The HF kernel bundle itself is pre-downloaded into the node HF cache (runbook), NOT
# baked into the image. Disable the shim with FA4_SM120_SHIM=0.
# Build on spark1:  docker build -t vllm-inkling-runtime:v026-scipy-fa4sm120 image/
# Distribute:     docker save ... | zstd | ssh spark2 'unzstd | docker load'
FROM vllm/vllm-openai:v0.26.0
RUN pip install --no-cache-dir scipy einops apache-tvm-ffi "nvidia-nccl-cu13==2.30.7" py-spy
COPY fa4_sm120_shim.py /usr/local/lib/python3.12/dist-packages/fa4_sm120_shim.py
COPY zzz_fa4_sm120_shim.pth /usr/local/lib/python3.12/dist-packages/zzz_fa4_sm120_shim.pth
