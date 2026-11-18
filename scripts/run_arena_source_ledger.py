# -*- coding: utf-8 -*-
"""MemoryArena formal_reasoning 双分支对照：original（CWRA 记忆）vs SourceLedger
（保留原始背景与题目、不写入模型生成历史）。首题共享原版生成。

移植自外部实验 fresh_chain_20261110（已测 22 题，独立复核 8/22->11/22）。
本版用 deepseek-chat（论文表 12 同模型），全部 24 题，调用上限 340。

Run: python scripts/run_arena_source_ledger.py
输出: experiment_data/runs/run_arena_ledger/
"""
import ast
import json
import re
import sys
import threading
import types
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_data" / "runs" / "run_arena_ledger"
RUN = "arena_source_ledger"
CALL_CAP = 340
API_MODEL = "deepseek-chat"

sys.path.insert(0, str(ROOT / "references" / "memoryarena"))
sys.path.insert(0, str(ROOT / "src"))

KEY = re.search(r"sk-[A-Za-z0-9_-]+",
                (ROOT.parent / "DeepSeek_key.txt").read_text(encoding="utf-8-sig")).group()

lock = threading.Lock()
calls = 0
usage = Counter()


def unpack(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    return json.loads(text)


def token_count(text):
    import tiktoken
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return len(text.split())
    return len(enc.encode(text or ""))


def api(messages, max_tokens=4096, tools_arg=None):
    global calls
    with lock:
        if calls >= CALL_CAP:
            raise RuntimeError("API call cap reached")
        calls += 1
    payload = {"model": API_MODEL, "messages": messages,
               "temperature": 0, "max_tokens": max_tokens}
    if tools_arg:
        payload.update(tools=tools_arg, tool_choice="auto")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP " + str(e.code)) from None
    with lock:
        for k, v in d.get("usage", {}).items():
            if isinstance(v, (int, float)):
                usage[k] += v
    return d["choices"][0]


class Metered:
    """Minimal OpenAI-shaped client routing to api()."""

    def __init__(self):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        messages = kw.get("messages")
        tools = kw.get("tools")
        max_tokens = kw.get("max_tokens", 4096)
        ch = api(messages, max_tokens, tools_arg=tools)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=ch["message"].get("content") or "",
                tool_calls=ch["message"].get("tool_calls")),
            finish_reason=ch.get("finish_reason"))])


fake = types.ModuleType("openai")
fake.OpenAI = lambda **kw: Metered()
sys.modules["openai"] = fake

# ---- load our arena memory system + the env's MathAgent (package imports) ----
from core_mas.arena_cwra_mem import CwraMemorySystem
from agent.math import MathAgent

# ---- safety patch: bounded execute_code ----
_original_execute = MathAgent.execute_code


def bounded(self, code):
    try:
        tree = ast.parse(code)
        body = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ("math", "numpy"):
                        return "Error: only math and numpy are available."
                    src = "math" if alias.name == "math" else "np"
                    tgt = alias.asname or alias.name
                    if tgt != src:
                        body.extend(ast.parse(tgt + " = " + src).body)
            elif isinstance(node, ast.ImportFrom):
                if node.module not in ("math", "numpy") or node.level:
                    return "Error: unsupported import"
                for alias in node.names:
                    if alias.name.startswith("_") or alias.name == "*":
                        return "Error: unsupported import"
                    body.extend(ast.parse(
                        (alias.asname or alias.name) + " = " +
                        ("math" if node.module == "math" else "np") +
                        "." + alias.name).body)
            else:
                body.append(node)
        tree.body = body
        code = ast.unparse(tree)
    except SyntaxError:
        return "Error: invalid Python syntax"
    prior = sys.gettrace()
    count = [0]

    def trace(frame, event, arg):
        count[0] += 1
        if count[0] > 200000:
            raise RuntimeError("Computation step limit exceeded")
        return trace
    try:
        sys.settrace(trace)
        return _original_execute(self, code)
    finally:
        sys.settrace(prior)


MathAgent.execute_code = bounded


def make_agent():
    a = MathAgent.__new__(MathAgent)
    a.model_name = API_MODEL
    a.temperature = 0.0
    a.backend_name = "openai"
    a.llm_backend = types.SimpleNamespace(
        chat=lambda messages: Metered().create(messages=messages).choices[0].message.content)
    return a


local = threading.local()


def assess(c, answer, finish):
    ch = api([{"role": "system", "content":
               "Assess EVERY independently required item in the reference. "
               "Do not accept only one matching part or mere topical "
               "similarity. Check all assumptions, constants, signs, "
               "exponents and quantifiers. Accept justified mathematical "
               "equivalence. Return JSON {\"checks\":[{\"requirement\":"
               "\"required statement\",\"satisfied\":true or false,"
               "\"reason\":\"why\"}]}. Never treat absence of a required "
               "statement as satisfying it."},
              {"role": "user", "content":
               "QUESTION:\n" + c["question"] + "\nREFERENCE:\n" +
               str(c["reference"]) + "\nCANDIDATE:\n" + answer}], 2048)
    try:
        checks = unpack(ch["message"].get("content") or "")["checks"]
        ok = bool(checks) and all(x.get("satisfied") is True for x in checks)
    except Exception:
        checks = []
        ok = False
    return bool(ok and finish == "stop" and answer), checks


def judge(q, ref, answer):
    msg = [{"role": "system", "content":
            "Assess mathematical equivalence against the reference. Check "
            "every requested part, definitions, assumptions, signs, "
            "exponents, factors and bound regimes. A correct stronger bound "
            "is acceptable only if justified for the same assumptions. "
            "Reject unfinished answers or partial derivations. Output "
            "CORRECT or INCORRECT only. Candidate identity is hidden."},
           {"role": "user", "content":
            "QUESTION:\n" + q + "\nREFERENCE:\n" + str(ref) +
            "\nCANDIDATE:\n" + answer}]
    return (api(msg, 16)["message"].get("content") or "").strip()


def chain(name):
    p = official[name]
    agent = make_agent()
    mem = CwraMemorySystem()
    mem.add_chunk("Initial result: Empty\n")
    prior = []
    seen = set()
    results = []
    for i, q in enumerate(p["questions"]):
        query = agent.build_prompt(q, p["backgrounds"][i] or
                                   "No information provided.")
        prompts = {"original": mem.wrap_user_prompt(query),
                   "source_ledger": ("<memory_context>\n" +
                                     "\n\n".join(prior) +
                                     "\n</memory_context>\n" + query)
                   if i else mem.wrap_user_prompt(query)}
        answers = {}
        actions = {}
        for method in ["original", "source_ledger"]:
            if i == 0 and method == "source_ledger":
                actions[method] = actions["original"]
                answers[method] = dict(answers["original"])
                continue
            local.finish = "stop"
            action = agent.act(prompts[method])
            answer = action.get("answer", action.get("result", "")) or ""
            actions[method] = action
            answers[method] = {"answer": answer, "finish": local.finish,
                               "tool_calls": len(action.get("tool_info", []))}
        for method in ["original", "source_ledger"]:
            if i == 0 and method == "source_ledger":
                answers[method] = dict(answers["original"])
                continue
            c = dict(dataset="MemoryArena", question=q, reference=p["answers"][i])
            a = answers[method]
            a["correct"], a["checks"] = assess(c, a["answer"], a["finish"])
        out = dict(id=name + ":" + str(i), paper=name, index=i, question=q,
                   reference=p["answers"][i], answers=answers,
                   shared_initial=(i == 0),
                   prompt_tokens={m: token_count(v) for m, v in prompts.items()})
        with lock:
            with (OUT / (RUN + ".jsonl")).open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print("RESULT", json.dumps({"id": out["id"],
                                    "correct": {m: a["correct"]
                                                for m, a in answers.items()}}),
              flush=True)
        results.append(out)
        action = actions["original"]
        mem.add_chunk(agent.build_memory_entry(
            q, action, observation={"tool_info": action.get("tool_info", [])},
            reward=None))
        bg = p["backgrounds"][i] or ""
        if bg and bg not in seen:
            prior.append("ORIGINAL BACKGROUND, task " + str(i) + ":\n" + bg)
            seen.add(bg)
        prior.append("PREVIOUS TASK " + str(i) + ":\n" + q)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    official_rows = ROOT / "experiment_data" / "runs" / "official_math_rows.json"
    global official
    official = {x["row"]["paper_name"]: x["row"]
                for x in json.loads(official_rows.read_text(encoding="utf-8"))["rows"]}
    names = ["2402.05814", "2507.18621", "2312.01127"]
    (OUT / (RUN + "_manifest.json")).write_text(json.dumps({
        "seed": "2026-11-10", "model": API_MODEL,
        "papers": names,
        "n": sum(len(official[n]["questions"]) for n in names),
        "memory": "original CWRA memory vs SourceLedger (raw backgrounds + "
                  "questions only, no model-generated history)",
        "q0_shared": True}, indent=2), encoding="utf-8")
    results = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        fs = {pool.submit(chain, n): n for n in names}
        for fut in as_completed(fs):
            n = fs[fut]
            try:
                results.extend(fut.result())
            except Exception as e:
                print("FAIL", n, e, flush=True)
    results.sort(key=lambda r: r["id"])
    # ---- independent blind audit (pairwise, identity hidden) ----
    audit = {}
    for r in results:
        if r["shared_initial"]:
            continue
        pair = {}
        for method in ["original", "source_ledger"]:
            a = r["answers"][method]
            if not a.get("answer"):
                pair[method] = False
                continue
            verdict = judge(r["question"], r["reference"], a["answer"])
            pair[method] = (verdict == "CORRECT")
        audit[r["id"]] = pair
    first = {m: sum(1 for r in results if not r["shared_initial"] and
                    r["answers"][m].get("correct")) for m in
             ["original", "source_ledger"]}
    audit_n = sum(1 for r in results if not r["shared_initial"])
    summary = {
        "completed": len(results),
        "model": API_MODEL,
        "first_judge": first,
        "blind_audit": {m: sum(1 for r in results if not r["shared_initial"]
                               and audit[r["id"]][m])
                        for m in ["original", "source_ledger"]},
        "audit_n": audit_n,
        "usage": dict(usage),
        "calls": calls,
    }
    (OUT / (RUN + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
