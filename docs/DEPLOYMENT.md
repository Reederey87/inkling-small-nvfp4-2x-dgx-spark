# Deployment guide — Inkling-Small-NVFP4 on 2x DGX Spark (TP=2 over QSFP)

End-to-end bring-up, from bare nodes to a serving cluster. Tested on two NVIDIA
DGX Spark (GB10 Grace-Blackwell, 121 GiB unified memory each), DGX OS /
Ubuntu 24.04, driver 580.173.02, Docker 29.x, CUDA 13.0. Operated from a Mac
over SSH; everything runs on the Sparks — the Mac only orchestrates.

Estimated time: ~1.5 h plus ~320 GB of downloads.

## 0. Prerequisites (both nodes)

- A user (default `nvidia`) in the `docker` group with **linger enabled**
  (`sudo loginctl enable-linger nvidia`) and passwordless SSH from the Mac for
  that user on both nodes.
- Docker with the NVIDIA runtime (`docker run --gpus all hello-world`-class
  check works).
- ~180 GB free on each node's NVMe (weights ~160 GB + bundle + image).

## 1. QSFP fabric (once)

Direct QSFP DAC between the two Sparks. Each port shows up as two PCIe-twin
interfaces; we assign static IPs on both rails and let NCCL merge them.

- Rail 1 (`enp1s0f1np1`): head `192.168.177.10/24`, worker `192.168.177.11/24`
- Rail 2 (`enP2p1s0f1np1`): head `192.168.178.10/24`, worker `192.168.178.11/24`
- MTU 9000 on both.

Verify from the head: `ping -c2 192.168.177.11`. Details: docs/NETWORKING.md.
If your interface names differ, edit the `[SITE]` block of `cluster.env`.

## 2. Configure

```
cp cluster.env.example cluster.env
$EDITOR cluster.env      # [SITE] blocks: hosts, user, rail IPs, API port
set -a; source cluster.env; set +a   # the ssh one-liners below use these vars
```

Every knob is documented inline — the comments are the "why". The tuned
defaults are exactly what the reference cluster runs.

## 3. Install on both nodes (from the Mac)

```
bash install-services.sh
```

Rsyncs the repo to `$KIT_DIR` on both nodes, installs the systemd user units
(`vllm-inkling-head` / `vllm-inkling-worker`; installed, not enabled), checks
linger. Then render the node-local envs once to validate:

```
ssh $CLUSTER_USER@$HEAD_HOST   'cd /home/nvidia/inkling-small-nvfp4 && bash render-env.sh head'
ssh $CLUSTER_USER@$WORKER_HOST 'cd /home/nvidia/inkling-small-nvfp4 && bash render-env.sh worker'
```

## 4. Build the runtime image (on the head node, or both)

```
ssh $CLUSTER_USER@$HEAD_HOST 'cd /home/nvidia/inkling-small-nvfp4/image && docker build -t vllm-inkling-runtime:v026-scipy-fa4sm120 .'
```

Thin overlay on the official arm64 `vllm/vllm-openai:v0.26.0`: `scipy`,
`einops`, `apache-tvm-ffi`, and the **FA4 SM120 shim** (mandatory — see
docs/KERNEL-FIX.md). Copy or rebuild on the worker
(`docker save | ssh worker docker load`).

## 5. Download weights (both nodes)

~160 GB per node (the TP=2 shards read the same checkpoint):

```
ssh $CLUSTER_USER@$HEAD_HOST 'docker run --rm -v /home/nvidia/hf-cache-inkling:/cache/huggingface \
  -e HF_HOME=/cache/huggingface --entrypoint huggingface-cli \
  vllm-inkling-runtime:v026-scipy-fa4sm120 download thinkingmachines/Inkling-Small-NVFP4'
```

The SM120 FA4 bundle (`SecondNatureComputing/flash-attn-4-sm120`, revision
pinned in the shim) downloads automatically on first kernel use; to pre-stage
it, run `tests/fa4_shim_check.py` once (see §6). To seed the worker from the
head instead of re-downloading: rsync `$HF_CACHE/hub` between nodes.

## 6. Preflight

```
bash preflight.sh
```

Checks: both nodes reachable, fabric IPs up, prod-model conflicts (this kit
uses its own container name/port/cache), image present, weights present, bundle
present, sysctl parity. Fix whatever it flags.

Optional kernel self-test on either node (proves the FA4 SM120 path end to
end, incl. the shim fixes):

```
ssh $CLUSTER_USER@$HEAD_HOST 'docker run --rm --gpus all \
  -v /home/nvidia/hf-cache-inkling:/cache/huggingface \
  -v /home/nvidia/inkling-small-nvfp4/tests:/t:ro \
  -e HF_HOME=/cache/huggingface --entrypoint python3 \
  vllm-inkling-runtime:v026-scipy-fa4sm120 /t/fa4_shim_check.py'
```

## 7. Start the cluster

```
bash start-cluster.sh     # worker first, then head; polls /health up to 25 min
```

First boot: weights load (~3 min), engine init (~25 s), KV pool report. Expect
`GPU KV cache size: ~213K tokens` at the stock profile (see
docs/CONFIGURATION.md for the `MAX_NUM_BATCHED_TOKENS` ↔ KV-pool trade-off).

Stop: `bash stop-cluster.sh` (head first, then worker; never the reverse).

## 8. Validate

```
bash bench/smoke.sh       # health, models, short-qa, reasoning, long-gen, vision
bash bench/probe.sh       # C1 x3 + C8 x3 throughput (JSONL in bench/results/)
```

Smoke expectations: every request returns coherent text; thinking appears in
`message.reasoning` (the Inkling template streams reasoning separately from
`content`; with `reasoning_effort: high`, small `max_tokens` can be entirely
consumed by thinking → empty content + `finish_reason=length` — this is the
model, not a bug).

## 9. Troubleshooting

See docs/TROUBLESHOOTING.md — every failure we hit is catalogued there with
the fix (scipy ImportError, memory-util dip band, MoE backend rejection, FA4
paged-KV assert, varlen kernel faults, cudagraph cross-node crash, NCCL fabric
sanity test, py-spy-from-sidecar technique).
