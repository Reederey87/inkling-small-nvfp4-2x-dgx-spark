#!/usr/bin/env python3
"""Smoke/correctness checks for the Inkling cluster — stdlib only (runs on spark1 via ssh).

Usage: python3 smoke.py --url http://127.0.0.1:8001 --model inkling-small-nvfp4
Exit 0 = all checks passed. Prints one PASS/FAIL line per check.
"""
import argparse
import base64
import io
import json
import struct
import sys
import urllib.request
import urllib.error
import zlib

FAILURES = []


def http_json(method, url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def chat(url, model, messages, max_tokens):
    return http_json("POST", f"{url}/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": max_tokens,
    })


def red_png_b64(size=64):
    """Build a solid-red PNG in-memory (no PIL on the node)."""
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))  # filter byte + RGB
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def garbled(text):
    """Crude garble detector: pathological single-char runs or replacement chars."""
    if "�" in text:
        return True
    run = 1
    for a, b in zip(text, text[1:]):
        run = run + 1 if a == b else 1
        if run > 100:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="inkling-small-nvfp4")
    args = ap.parse_args()
    url, model = args.url.rstrip("/"), args.model

    # 1. health
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            report("health", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        report("health", False, str(e))
        print("Smoke aborted: server unreachable.")
        return 1

    # 2. served model id
    try:
        models = http_json("GET", f"{url}/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        report("models", model in ids, f"served={ids}")
    except Exception as e:
        report("models", False, str(e))

    # 3. short QA + reasoning presence (effort high is the lane-1 default)
    qa = None
    try:
        qa = chat(url, model, [{"role": "user", "content":
                                "Reply with just the answer, one word: what is the capital of France?"}],
                  max_tokens=4096)
        msg = qa["choices"][0]["message"]
        content = msg.get("content") or ""
        report("short-qa", "paris" in content.lower(), f"content={content[:80]!r}")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        report("reasoning-block", len(reasoning) > 0, f"reasoning_len={len(reasoning)}")
    except Exception as e:
        report("short-qa", False, str(e))
        report("reasoning-block", False, "no response")

    # 4. long generation: completes, non-null, not garbled
    try:
        r = chat(url, model, [{"role": "user", "content":
                               "Write a detailed essay about the history of computing."}],
                 max_tokens=2048)
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        fr = ch.get("finish_reason")
        report("long-gen-finish", fr in ("stop", "length"), f"finish_reason={fr}")
        report("long-gen-content", len(content) > 200 and not garbled(content),
               f"len={len(content)}")
    except Exception as e:
        report("long-gen-finish", False, str(e))
        report("long-gen-content", False, "no response")

    # 5. vision path (multimodal is a headline feature; vision tower is BF16)
    try:
        uri = "data:image/png;base64," + red_png_b64()
        r = chat(url, model, [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image? One word."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}], max_tokens=2048)
        content = r["choices"][0]["message"].get("content") or ""
        report("vision-red", "red" in content.lower(), f"content={content[:80]!r}")
    except urllib.error.HTTPError as e:
        report("vision-red", False, f"HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        report("vision-red", False, str(e))

    print(f"\n{'SMOKE PASS' if not FAILURES else 'SMOKE FAIL: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
