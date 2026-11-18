"""E7 v2 protocol: fault injection on E-mem's own per-agent pipeline.

E7 v1 (naive protocol: agents returned raw session text) gave F1 ~0.1 — the
pipeline was too weak to carry signal (recorded as a failed iteration in
RESEARCH_REVIEW). This version gives EACH agent an E-mem text-mode
ChatManager (block summaries + hybrid router, validated in E0 at F1 0.445)
storing its allocated session texts, and the coordinator merges each
surviving agent's search_memory(question) output.

Fault: victim agent's manager is not queried (and its memory never reaches
the coordinator prompt). Allocations (emem-like origin / frequency / core_mas)
as in locomo_fault.

Run (cloud): PYTHONPATH=src:emem python -m core_mas.locomo_emem_fault \
  --config emem/evaluation/locomo/config.yaml --out-dir run_e7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import fractions

_ORIG = fractions.Fraction.__new__


def _patched(cls, *a, **kw):
    kw.pop("_normalize", None)
    return _ORIG(cls, *a, **kw)


fractions.Fraction.__new__ = _patched

from .eval_agents import HTTPOpenAILLM
from .locomo_fault import (build_allocations, load_locomo, make_units,
                           partition_sessions, qa_witness_bundles)

LOCOMO_COORD_SYSTEM = (
    "You are the team coordinator. Answer the question using ONLY the "
    "evidence below, in the form of a SHORT PHRASE, using exact wording from "
    "the evidence whenever possible. Do not use outside knowledge. Reply "
    "exactly in the form: ANSWER: <short phrase>. Only if no evidence is "
    "relevant at all, reply: ANSWER: INSUFFICIENT_EVIDENCE")

# official E-mem/LoCoMo category instructions (replicated from
# evaluation/locomo/eval_locomo.py so the coordinator answers the same way the
# official eval's question_answer_model does)
CATEGORY_INSTRUCTIONS = {
    1: """Based on the text below, analyze the context to provide the best answer to the question.
### PROCESS
1. Understand the specific details requested. 2. Reason over the text. 3. Extract the final answer using exact wording and terminology from the context whenever possible.
4. For date/time questions, strictly follow the format "15 July 2023" and give a specific date whenever possible.
### CRITICAL CONSTRAINTS
* Strict Relevance: NEVER output content unrelated to the question.
* Anti-Refusal: NEVER say "not mentioned". Even if not explicitly stated, you MUST infer the most probable answer from the provided evidence.
* Format: the final OUTPUT must be a short phrase (under 10 words). NO sentences.""",
    2: """You are a precise date extraction and normalization assistant.
Answer based on the document, STRICTLY adhering to the formatting rules:
MODE A (question starts with "When"): convert to [Day] [Full Month Name] [Year] (e.g. "6 July 2023"; month/year or year only allowed).
MODE B (other time questions like "How long"): output a duration or relative interval like "3 years", "6 months", "10 years ago".
CRITICAL: NEVER output "not mentioned"/"unknown". If the exact answer is missing, you MUST infer the most reasonable date/duration and format it.
Output ONLY the final string, no punctuation at the end.""",
    3: """Based on the text below, write an answer in the form of a short phrase, not a sentence.
For "Would" questions choosing between options: output the specific option selected.
For yes/no prediction questions: start with "Likely yes" or "Likely no".
NEVER output refusal phrases. ALWAYS infer the best possible answer based on available evidence.""",
    4: """You are an extractive QA assistant. Extract the exact answer substring from the text.
1. Locate the exact answer in the TEXT. 2. Output ONLY the key entity, date, name, or short phrase.
3. DO NOT use full sentences or filler words like "The answer is", "a", "the", "she is".
4. Keep it as short as possible (ideally less than 10 words). 5. Use EXACT wording from the text if possible.""",
    5: """### CRITICAL: THE "REJECTION FIRST" PROTOCOL
Your goal is to PROVE that the answer is MISSING. Actively try to disqualify any potential evidence.
1. Default Verdict: Start with the assumption that the correct answer is "Not mentioned".
2. Exact Match Test: Does the text contain the exact keywords or synonyms for the specific detail asked? If NO -> Select "Not mentioned". If YES but different context/person/time -> Select "Not mentioned".
3. No Inference Firewall: even if an answer is 99% likely, if it is not 100% stated -> Select "Not mentioned".
4. Final Check: can you underline the exact sentence that proves the answer? If not -> Select "Not mentioned".
You MUST answer with the selected answer only, without explanation.""",
}


def coord_system_for(category: int) -> str:
    instr = CATEGORY_INSTRUCTIONS.get(category)
    if instr is None:
        return LOCOMO_COORD_SYSTEM
    return ("You are an expert at answering questions based on conversation "
            "history. Instructions: " + instr)


def build_user_prompt(task, ev_text):
    """Official-style user prompt; category 5 gets the randomized
    two-option construction exactly like the official eval."""
    if task.get("category") == 5:
        ref = task["answer"]
        wrong = task.get("adversarial") or "Unknown"
        if ref != "Not mentioned in the conversation":
            opts = [ref, wrong] if random.random() < 0.5 else [wrong, ref]
        else:
            opts = [wrong, ref] if random.random() < 0.5 else [ref, wrong]
        return (f"### CRITICAL: THE \"REJECTION FIRST\" PROTOCOL\n"
                f"Your goal is to PROVE that the answer is MISSING...\n"
                f"YOU MUST Select the correct answer: '{opts[0]}' or "
                f"'{opts[1]}'. Provide ONLY the selected answer without "
                f"explanation.\n\nQuestion: {task['question']}\n\n"
                f"Context:\n{ev_text[:10000]}")
    return (f"Question: {task['question']}\n\nContext:\n{ev_text[:10000]}")


def extract_answer(text):
    m = re.search(r"ANSWER\s*[:：]\s*(.*)", (text or "").strip(), re.I | re.S)
    return m.group(1).strip() if m else (text or "").strip()


RELAY_SELECT_SYSTEM = (
    "Select source passages needed to answer the question completely. "
    "Search for the correct entity, relation, time, and every requested "
    "part. Include definitions, notation, assumptions and earlier referenced "
    "procedures on which the answer depends. Do NOT solve the question. "
    'Return JSON {"ids":[integer passage IDs],"missing":["any unresolved '
    'dependency"]}. Select at most 24 passages; avoid irrelevant passages '
    "and generic preambles.")


def relay_select(llm_sel, source: str, question: str):
    """Passage selector over the numbered source (protocol-constrained: the
    source is the union of SURVIVING agents' allocated units only). Returns
    (packet, selection). Falls back to the full source on parse failure or
    empty selection."""
    blocks = [x for x in re.split(r"\n\s*\n", source) if x.strip()]
    if not blocks:
        return "", {"ids": []}
    numbered = "\n\n".join(f"PASSAGE {i}\n{s}" for i, s in enumerate(blocks))
    user = f"QUESTION:\n{question}\nNUMBERED SOURCE:\n{numbered}"
    out = llm_sel.chat(RELAY_SELECT_SYSTEM, user)
    selection = {}
    ids = []
    try:
        m = re.search(r"\{.*\}", out, re.S)
        selection = json.loads(m.group(0)) if m else {}
        raw_ids = selection.get("ids", [])
        ids = [i for i in raw_ids if isinstance(i, int) and 0 <= i < len(blocks)]
    except Exception:
        selection = {"parse_error": True}
    selected = set()
    for i in ids:
        selected.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(blocks))
    packet = "\n\n".join(f"PASSAGE {i}\n{blocks[i]}" for i in sorted(selected)) \
        if selected else source
    return packet, selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="/root/autodl-tmp/core-mas/locomo10.json")
    ap.add_argument("--emem-repo", default="/root/autodl-tmp/core-mas/emem")
    ap.add_argument("--config", default="/root/autodl-tmp/core-mas/emem/evaluation/locomo/config.yaml")
    ap.add_argument("--out-dir", default="run_e7")
    ap.add_argument("--convs", type=int, default=4)
    ap.add_argument("--questions-per-conv", type=int, default=8)
    ap.add_argument("--budget-ratio", type=float, default=0.6)
    ap.add_argument("--victims", nargs="+", default=["s0", "agent_c"])
    ap.add_argument("--include-relay", action="store_true",
                    help="add the source_relay strategy (passage selection over "
                         "surviving units, then category-specific answering)")
    args = ap.parse_args()

    sys.path.insert(0, args.emem_repo)
    from config import ensure_app_config, load_raw_config
    from src.conversation_manager.factory import create_chat_manager
    from evaluation.locomo.utils import calculate_metrics

    llm = HTTPOpenAILLM()
    data = load_locomo(args.locomo, args.emem_repo)
    app_cfg = ensure_app_config(load_raw_config(args.config))
    kw = app_cfg.to_chat_manager_kwargs()

    all_rows = []
    t0 = time.time()
    for ci in range(args.convs):
        conv = data[ci]
        parts = partition_sessions(conv.conversation.sessions)
        units, dia_to_unit = make_units(parts, f"conv{ci}")
        qa_tasks = qa_witness_bundles(conv.qa, dia_to_unit)
        if not qa_tasks:
            continue
        strats, _ = build_allocations(units, {i: t for i, t in enumerate(qa_tasks)},
                                      args.budget_ratio, 100 + ci)
        # protocol ceiling reference: ONE manager with the FULL history
        # (no partition, no budget) — no-fault only
        strats["full_ref"] = {"agent_a": units, "agent_b": [], "agent_c": []}
        for sid, alloc in strats.items():
            # build per-agent E-mem managers over allocated sessions
            managers = {}
            for agent in ("agent_a", "agent_b", "agent_c"):
                text = "\n\n".join(u.content for u in alloc.get(agent, []))
                if not text:
                    managers[agent] = None
                    continue
                mgr = create_chat_manager(**kw)
                mgr.add_memory(text)
                managers[agent] = mgr
            conds = ["s0"] if sid == "full_ref" else args.victims
            for t in qa_tasks[:args.questions_per_conv]:
                for victim in conds:
                    evidence = []
                    for agent, mgr in managers.items():
                        if agent == victim or mgr is None:
                            continue
                        try:
                            resp = mgr.search_memory(t["question"])
                        except Exception as e:
                            resp = ""
                        if resp:
                            evidence.append(f"Agent {agent}:\n{resp}")
                    ev_text = "\n\n".join(evidence) if evidence else "(no evidence available)"
                    sys_msg = coord_system_for(t.get("category", 0))
                    user = build_user_prompt(t, ev_text)
                    coord = llm.chat(sys_msg, user)
                    pred = extract_answer(coord)
                    if t.get("category", 0) not in (2, 5):
                        from evaluation.locomo.utils import extract_answer_from_xml
                        pred = extract_answer_from_xml(pred, t.get("category", 0))
                    metrics = calculate_metrics(pred, t["answer"])
                    all_rows.append({"conv": ci, "strategy": sid, "victim": victim,
                                     "question": t["question"][:80],
                                     "prediction": pred[:200], "reference": t["answer"][:120],
                                     **{k: v for k, v in metrics.items()}})
            print(f"conv{ci} {sid} done ({time.time()-t0:.0f}s)", flush=True)
        if args.include_relay:
            # source_relay: passage selection over the SURVIVING agents'
            # allocated units (same core_mas allocation as the CORE-MAS
            # strategy), then the category-specific coordinator. No E-mem
            # managers, no global-text reading: the fault protocol applies
            # exactly as in the other strategies.
            alloc = strats["core_mas"]
            llm_sel = HTTPOpenAILLM(max_tokens=2048)
            for t in qa_tasks[:args.questions_per_conv]:
                for victim in args.victims:
                    srcs = []
                    for agent in ("agent_a", "agent_b", "agent_c"):
                        if agent == victim:
                            continue
                        txt = "\n\n".join(u.content for u in alloc.get(agent, []))
                        if txt:
                            srcs.append(txt)
                    source = "\n\n".join(srcs)
                    packet, _ = relay_select(llm_sel, source, t["question"])
                    sys_msg = coord_system_for(t.get("category", 0))
                    user = build_user_prompt(
                        t, packet if packet else "(no evidence available)")
                    coord = llm.chat(sys_msg, user)
                    pred = extract_answer(coord)
                    if t.get("category", 0) not in (2, 5):
                        from evaluation.locomo.utils import extract_answer_from_xml
                        pred = extract_answer_from_xml(pred, t.get("category", 0))
                    metrics = calculate_metrics(pred, t["answer"])
                    all_rows.append({"conv": ci, "strategy": "source_relay",
                                     "victim": victim,
                                     "question": t["question"][:80],
                                     "prediction": pred[:200],
                                     "reference": t["answer"][:120],
                                     **{k: v for k, v in metrics.items()}})
            print(f"conv{ci} source_relay done ({time.time()-t0:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e7_emem.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved {len(all_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
