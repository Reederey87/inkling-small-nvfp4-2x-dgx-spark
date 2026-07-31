#!/usr/bin/env python3
"""De-streaming proxy for the Inkling vLLM server (127.0.0.1:8001 -> 8002).

Why this exists (2026-07-31, evidence in docs/TROUBLESHOOTING.md):
vLLM 0.26's inkling tool-call parser is broken on the STREAMING path for
Hermes-shaped requests — the model's <|content_invoke_tool_json|> markup is
returned as plain content (finish_reason=stop) instead of being parsed into
tool_calls. Measured: identical request, stream=True leaks 4/4, stream=False
parses 6/6. Hermes hardcodes "stream": True with no per-provider opt-out, so
this proxy strips streaming: it accepts Hermes' streaming request, forwards
it to vLLM with stream=false, and returns the complete JSON. Hermes handles
non-streaming responses natively (its documented fallback for adapters that
return them). Cost: no token-level streaming in the TUI (responses appear
when generation completes).

Only /v1/chat/completions is de-streamed; everything else (/v1/models,
/health, ...) is proxied verbatim. Stdlib only. 127.0.0.1:8002, loopback.
"""
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8001
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8002
TIMEOUT_S = 1800  # matches hermes' request_timeout_seconds


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep logs useful but quiet
        print(f"[destream] {fmt % args}", flush=True)

    def _forward(self, method, path, body, headers):
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=TIMEOUT_S)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp.getheaders(), data

    def _passthrough(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        try:
            status, resp_headers, data = self._forward(method, self.path, body, headers)
        except Exception as e:
            msg = json.dumps({"error": {"message": f"destream proxy upstream error: {e}",
                                        "type": "BadGateway", "code": 502}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(status)
        for k, v in resp_headers:
            if k.lower() not in ("transfer-encoding", "connection", "content-encoding"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._passthrough("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path.rstrip("/") == "/v1/chat/completions" and body:
            try:
                payload = json.loads(body)
            except Exception:
                payload = None
            if payload is not None and payload.get("stream"):
                payload["stream"] = False
                payload.pop("stream_options", None)
                body = json.dumps(payload).encode()
                print("[destream] stream stripped for chat/completions", flush=True)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=TIMEOUT_S)
            conn.request("POST", self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            status = resp.status
            conn.close()
        except Exception as e:
            msg = json.dumps({"error": {"message": f"destream proxy upstream error: {e}",
                                        "type": "BadGateway", "code": 502}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    srv = Server((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"[destream] listening on {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}",
          flush=True)
    srv.serve_forever()
