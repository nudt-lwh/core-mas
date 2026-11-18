"""OpenAI-compatible chat server backed by llama_cpp (for AutoDL 4090).

E0 finding: llama_cpp.server (0.3.36) returns Qwen3 tool calls as raw
`<tool_call>...</tool_call>` text in `content` instead of structured
`tool_calls`, which breaks E-mem's tool loop. This thin server wraps
llama_cpp.create_chat_completion and converts the raw text into the
structured OpenAI format E-mem expects (mirrors local_llm_server.py parsing).

Also strips Qwen3 <think> blocks from plain content.
"""

import argparse
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("llamacpp_server")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int):
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_gpu_layers=99, n_ctx=n_ctx, verbose=False)
        self.lock = threading.Lock()

    @staticmethod
    def _preprocess(messages):
        """Adapt OpenAI structured tool_calls to Qwen3's raw <tool_call> format
        and disable thinking via /no_think on the last user message."""
        out = []
        for msg in messages:
            m = dict(msg)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                blocks = []
                for tc in m["tool_calls"]:
                    try:
                        obj = {"name": tc["function"]["name"],
                               "arguments": json.loads(tc["function"]["arguments"])}
                    except Exception:
                        obj = {"name": tc["function"]["name"], "arguments": {}}
                    blocks.append("<tool_call>\n%s\n</tool_call>" % json.dumps(obj, ensure_ascii=False))
                m["content"] = "\n".join(blocks)
                m.pop("tool_calls", None)
                m.pop("refusal", None)
            out.append(m)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user" and isinstance(out[i].get("content"), str):
                out[i]["content"] = out[i]["content"].rstrip() + "\n/no_think"
                break
        return out

    def chat(self, payload: dict):
        with self.lock:
            messages = self._preprocess(payload.get("messages", []))
            tools = payload.get("tools") or None
            max_tokens = int(payload.get("max_tokens", 1024))
            temperature = float(payload.get("temperature", 0.0))
            t0 = time.time()
            r = self.llm.create_chat_completion(
                messages=messages, tools=tools, temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = r["choices"][0]["message"].get("content") or ""
            calls = []
            for m in TOOL_CALL_RE.finditer(raw):
                try:
                    obj = json.loads(m.group(1).strip())
                    calls.append({
                        "id": f"call_{len(calls):04d}",
                        "type": "function",
                        "function": {
                            "name": obj.get("name", ""),
                            "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False),
                        },
                    })
                except json.JSONDecodeError:
                    log.warning("unparsable tool_call: %r", m.group(1)[:200])
            content = TOOL_CALL_RE.sub("", raw)
            content = THINK_RE.sub("", content)
            content = re.sub(r"</?think>", "", content).strip()
            message = {"role": "assistant", "content": content or None}
            if calls:
                message["content"] = None
                message["tool_calls"] = calls
            log.info("chat ok %.1fs tools=%d content_len=%d",
                     time.time() - t0, len(calls), len(content or ""))
            return {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "qwen3-4b-local"),
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": r.get("usage", {}),
            }


def make_handler(backend: LlamaCppBackend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._send(200, {"object": "list", "data": [{"id": "qwen3-4b-local", "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length))
            except Exception as e:
                self._send(400, {"error": {"message": str(e)}})
                return
            try:
                self._send(200, backend.chat(payload))
            except Exception as e:
                log.exception("chat failed")
                self._send(500, {"error": {"message": str(e)}})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=6006)
    args = ap.parse_args()
    backend = LlamaCppBackend(args.model, args.n_ctx)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend))
    log.info("serving on http://127.0.0.1:%d/v1", args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
