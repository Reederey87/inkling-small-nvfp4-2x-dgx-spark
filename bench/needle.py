#!/usr/bin/env python3
"""Needle-in-haystack long-context probe for the Inkling 2xSpark deployment.

Sends a synthetic haystack (repeated filler) with a unique needle sentence at
~90% depth, then asks for the needle. Stdlib-only so it runs directly on the
head node (the API is loopback-only):

  python3 bench/needle.py --depths 65536 262144 524288
  python3 bench/needle.py --depths 900000 --url http://127.0.0.1:8001

Gates per depth: needle phrase HIT in the reply AND the exact
usage.prompt_tokens readback >= ~depth (the --depth number is an estimate;
prompt_tokens is ground truth). An HTTP 400 "context length" error means the
server window is smaller than the requested depth (e.g. still on the 131K
default profile).

Note: on the 1M lane (MAX_NUM_BATCHED_TOKENS=1024) long prefills are slow —
a ~500K-token prompt can take several minutes. --timeout defaults to 1800 s.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = ("The archive contains ordinary field notes about weather, shipping "
          "lanes, and harbour schedules. ")
NEEDLE_ID = "ZEPHYR-QUILL-7429"
NEEDLE = ("The needle phrase is %s. It marks the hidden buoy in the archive. "
          % NEEDLE_ID)
CHARS_PER_TOKEN = 6.0  # calibrated for the built-in filler (2026-08-01, inkling tokenizer:
                        # repetitive BPE-friendly text tokenizes at ~6 chars/token;
                        # exact count always comes back in usage.prompt_tokens)


def build_prompt(depth_tokens):
    target_chars = int(depth_tokens * CHARS_PER_TOKEN)
    head = FILLER * max(1, (target_chars * 9 // 10) // len(FILLER))
    tail = FILLER * max(1, (target_chars * 1 // 10) // len(FILLER))
    question = ("Based only on the notes above: what is the needle phrase, "
                "and what does it mark? Answer in one short sentence.")
    return head + NEEDLE + tail + "\n\n" + question


def probe(url, model, depth, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You answer questions about long documents precisely."},
            {"role": "user", "content": build_prompt(depth)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        return {"depth": depth,
                "error": "HTTP %s: %.200s" % (e.code, e.read().decode("utf-8", "replace"))}
    except Exception as e:
        return {"depth": depth, "error": repr(e)}
    latency = time.time() - t0
    msg = out.get("choices", [{}])[0].get("message", {})
    text = msg.get("content") or ""
    usage = out.get("usage", {})
    return {"depth": depth,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(latency, 1),
            "hit": NEEDLE_ID in text,
            "answer": text.strip().replace("\n", " ")[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="inkling-small-nvfp4")
    ap.add_argument("--depths", type=int, nargs="+", default=[65536, 262144])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    rc = 0
    for d in args.depths:
        r = probe(args.url, args.model, d, args.max_tokens, args.timeout)
        if "error" in r:
            print("depth ~%-8d ERROR %s" % (d, r["error"]))
            rc = 1
            continue
        ok = r["hit"] and (r["prompt_tokens"] or 0) >= int(d * 0.85)
        print("depth ~%-8d prompt_tokens=%-8s latency=%-7ss needle=%s  %s"
              % (d, r["prompt_tokens"], r["latency_s"],
                 "HIT" if r["hit"] else "MISS", r["answer"]))
        if not ok:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
