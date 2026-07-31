# Hermes Agent on Inkling-Small-NVFP4

Everything needed to run [Hermes Agent](https://hermes-agent.nousresearch.com)
against this deployment — config profile, switch script, and the de-streaming
proxy that makes tool calls work.

## Why the extra pieces

1. **The profile (`config.yaml`)** — points Hermes at the local server with the
   right context window (131072), compaction tuned for it, and the vision
   auxiliary LOCAL (Inkling is multimodal — images never leave the box).
   Header comments document every choice.
2. **The de-streaming proxy (`proxy/destream_proxy.py`)** — vLLM 0.26's inkling
   tool parser leaks tool-call markup as plain content on the **streaming**
   path (measured: `stream=True` leaks 4/4, `stream=False` parses 6/6 with the
   identical request). Hermes hardcodes `"stream": true`. The proxy listens on
   `127.0.0.1:8002`, strips `stream`, and forwards to vLLM on `:8001` — tool
   calls then parse correctly. Cost: no token-level streaming in the TUI.
3. **The switch script (`switch-hermes.sh`)** — safe install/verify/rollback.

## Setup

Prereqs: the Inkling cluster serving (`bash start-cluster.sh` → `:8001`
healthy) and Hermes Agent installed on the head node (`hermes` on PATH).

```
# 1. install the proxy unit on the head node (once)
scp proxy/destream_proxy.py systemd/inkling-destream-proxy.service \
    "$CLUSTER_USER@$HEAD:/tmp/"
ssh "$CLUSTER_USER@$HEAD" '
  mkdir -p ~/inkling-small-nvfp4/proxy ~/.config/systemd/user
  cp /tmp/destream_proxy.py ~/inkling-small-nvfp4/proxy/
  cp /tmp/inkling-destream-proxy.service ~/.config/systemd/user/
  systemctl --user daemon-reload
  systemctl --user enable --now inkling-destream-proxy'

# 2. install the Hermes profile + verify
bash hermes/switch-hermes.sh inkling
```

The switch script backs up the live config to
`~/.hermes/config.yaml.bak-switch-<ts>` before installing — rollback is
`switch-hermes.sh ~/.hermes/config.yaml.bak-switch-<ts>` (on the node) or
re-running with any other profile file.

Env overrides: `HEAD` (default `spark1.local`), `CLUSTER_USER` (default
`nvidia`), `HERMES_SSH` if Hermes runs on a different host than the head.

## Gotchas (all probe-verified)

- **base_url fields must match** — `model.base_url` and
  `providers.custom.base_url` are both `:8002`; a mismatch silently detaches
  the provider block.
- **Reasoning level**: the serve-side default (`REASONING_EFFORT` in
  cluster.env, `high` stock) is what applies — Hermes does not forward
  `agent.reasoning_effort` for this provider (verified from a captured wire
  request). Change it on the server, not in Hermes.
- **Thinking eats the output budget** — reasoning streams into
  `message.reasoning`; small `max_tokens` yields empty content with
  `finish_reason=length`. The profile sets 65536.
- **Vision is local** — `auxiliary.vision: provider: main`. `web_extract` and
  `title_generation` default to an optional cloud model (needs an
  OpenAI-compatible key; swap to `provider: main` for fully-local).
- **Compaction is local** (`provider: main`), so `idle_compact_after_seconds`
  must stay 0 — otherwise session resume blocks on a local compaction.
