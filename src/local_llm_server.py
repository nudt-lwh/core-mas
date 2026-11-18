"""Minimal OpenAI-compatible chat server for E-mem probe (E0).

Serves a Qwen3-4B Q4_K_M GGUF via transformers on CUDA. Supports the subset of
the OpenAI Chat Completions API that E-mem uses (src/agent/base.py):
  - model / messages / tools / max_tokens / temperature
  - extra_body.repetition_penalty
  - tool-call loop (Qwen3 chat template tool format)

Probe-only engineering shim; replaces llama-server (GitHub release binaries are
unreachable from this network). Sequential generation (one lock).
"""

import argparse
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("local_llm_server")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class QwenServer:
    def __init__(self, gguf_dir: str, gguf_file: str, tokenizer_dir: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # E0 finding: fp16 dequant of Q4_K_M (~8.3GB) overflows 8GB VRAM with
        # >4k context (CUDA spills to RAM, ~0.35 tok/s). CPU fp32 ~37 tok/s.
        if device == "cpu":
            torch.set_num_threads(20)
            dtype = torch.float32
        else:
            dtype = torch.float16
        self.device = device
        log.info("loading GGUF %s from %s on %s (dtype=%s)", gguf_file, gguf_dir, self.device, dtype)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            gguf_dir, gguf_file=gguf_file, torch_dtype=dtype
        )
        self.model.eval()
        if device == "cuda":
            self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.lock = threading.Lock()
        log.info("model loaded in %.1fs; device=%s", time.time() - t0, self.device)

    @staticmethod
    def _extract_tool_calls(text: str):
        """Extract Qwen3 <tool_call> JSON blocks from generated text."""
        calls = []
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
            try:
                obj = json.loads(m.group(1).strip())
                calls.append(
                    {
                        "id": f"call_{len(calls):04d}",
                        "type": "function",
                        "function": {
                            "name": obj.get("name", ""),
                            "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                )
            except json.JSONDecodeError:
                log.warning("unparsable tool_call block: %r", m.group(1)[:200])
        return calls

    def chat(self, payload: dict):
        with self.lock:
            return self._chat_locked(payload)

    def _chat_locked(self, payload: dict):
        import torch

        messages = payload.get("messages", [])
        tools = payload.get("tools") or None
        max_tokens = int(payload.get("max_tokens", 1024))
        temperature = float(payload.get("temperature", 0.0))
        extra = payload.get("extra_body") or {}
        repetition_penalty = extra.get("repetition_penalty")

        # Standardize assistant tool_calls messages (openai SDK may serialize them)
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                pass  # template handles standard format
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = float(repetition_penalty)
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        # strip special tokens and think blocks for content
        content = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        content = re.sub(r"</?think>", "", content)
        for tok in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            content = content.replace(tok, "")
        content = content.replace("<tool_call>", "").replace("</tool_call>", "").strip()

        tool_calls = self._extract_tool_calls(text) if tools else []
        message = {"role": "assistant", "content": None if tool_calls else content}
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "qwen3-4b-local"),
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}
            ],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(new_ids.shape[0])},
        }


def make_handler(server: QwenServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
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
            t0 = time.time()
            try:
                resp = server.chat(payload)
                ct = resp["usage"]["completion_tokens"]
                log.info("chat ok %.1fs (%d tok, %.1f tok/s)", time.time() - t0, ct,
                         ct / max(time.time() - t0, 0.01))
                self._send(200, resp)
            except Exception as e:
                log.exception("chat failed")
                self._send(500, {"error": {"message": str(e)}})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", default="tools/models")
    ap.add_argument("--gguf-file", default="Qwen3-4B-Q4_K_M.gguf")
    ap.add_argument("--tokenizer-dir", default="tools/models/qwen3-tokenizer")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    server = QwenServer(args.gguf_dir, args.gguf_file, args.tokenizer_dir, args.device)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(server))
    log.info("serving on http://127.0.0.1:%d/v1", args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
