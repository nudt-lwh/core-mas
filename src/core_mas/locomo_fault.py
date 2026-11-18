"""E7: LoCoMo adapted fault protocol (allocation comparison under failure).

Design: writing/notes/E7_PROTOCOL.md
- history partitioned by session into 3 agent memories (+ coordinator index);
- allocation strategies (E-mem-like origin sharding / frequency replication /
  CORE-MAS witness-aware) under an equal token budget;
- s0 / s1(victim) failures; ECCR evidence protocol (same as E2);
- grading with E-mem's official metric functions (EM/F1/ROUGE/BLEU/METEOR)
  imported from the E-mem repo present on the cloud.

Not the official leaderboard: self-built fault protocol on official data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from .eval_agents import (AGENT_SYSTEM, COORD_SYSTEM, EvalAgents,
                          HTTPOpenAILLM, extract_answer)
from .memory_units import MemoryUnit, token_count

# nltk 3.8.1 calls Fraction(..., _normalize=False); Python 3.12 removed that
# kwarg (normalization is unconditional). Patch before importing the E-mem
# metric utilities.
import fractions

_ORIG_FRACTION_NEW = fractions.Fraction.__new__


def _patched_fraction_new(cls, *args, **kwargs):
    kwargs.pop("_normalize", None)
    return _ORIG_FRACTION_NEW(cls, *args, **kwargs)


fractions.Fraction.__new__ = _patched_fraction_new

# three agents + coordinator
LOCOMO_AGENTS = ["agent_a", "agent_b", "agent_c", "coordinator"]

LOCOMO_COORD_SYSTEM = (
    "You are the team coordinator. Answer the question using ONLY the "
    "evidence below, in the form of a SHORT PHRASE, using exact wording from "
    "the evidence whenever possible. Do not use outside knowledge. Reply "
    "exactly in the form: ANSWER: <short phrase>. Only if no evidence is "
    "relevant at all, reply: ANSWER: INSUFFICIENT_EVIDENCE")


def extract_loco_answer(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"ANSWER\s*[:：]\s*(.*)", t, re.I | re.S)
    return m.group(1).strip() if m else t


def load_locomo(path: str, emem_repo: str = None):
    """Load LoCoMo via the official E-mem parser (conversation.sessions)."""
    import sys
    if emem_repo and emem_repo not in sys.path:
        sys.path.insert(0, emem_repo)
    from evaluation.locomo.load_dataset import load_locomo_dataset
    return load_locomo_dataset(path)


def partition_sessions(sessions, block_tokens=4096):
    """Session-level memory units (LoCoMo's natural granularity).

    Session indices split into three contiguous thirds -> agent_a/b/c
    (documented adaptation; not speaker-based). Returns
    {agent: [(session_text, [dia_ids])]}. Oversized sessions are chunked;
    their dia_ids map to the first chunk's unit.
    """
    nums = sorted(int(k.split("_")[1]) for k in sessions
                  if k.startswith("session_"))
    n_max = max(nums) if nums else 1
    t1, t2 = n_max // 3, (2 * n_max) // 3
    parts = {"agent_a": [], "agent_b": [], "agent_c": []}
    for skey in sorted(sessions.keys()):
        if not skey.startswith("session_"):
            continue
        num = int(skey.split("_")[1])
        agent = "agent_a" if num <= t1 else ("agent_b" if num <= t2 else "agent_c")
        # official E-mem per-turn format: the session date is critical for the
        # hybrid router's temporal matching (E7 v7 diagnosis)
        session_text = "\n".join(f"[{sessions[skey].date_time}] {t.speaker}: {t.text}"
                                 for t in sessions[skey].turns)
        dia_ids = [t.dia_id for t in sessions[skey].turns]
        while token_count(session_text) > block_tokens:
            cut = session_text[: int(len(session_text) * 0.8)]
            idx = cut.rfind("\n")
            parts[agent].append((cut[:idx], dia_ids))
            session_text = session_text[idx + 1:]
        parts[agent].append((session_text, dia_ids))
    return parts


def make_units(parts_by_agent, conv_id):
    """Memory units from session partitions. Returns (units, dia_to_unit)."""
    units = []
    dia_to_unit = {}
    for agent, items in parts_by_agent.items():
        for i, (text, dia_ids) in enumerate(items):
            u = MemoryUnit(
                unit_id=f"{conv_id}_{agent}_s{i}",
                fact_id=f"{conv_id}_{agent}_s{i}",  # session unit as its own "fact"
                kind="raw",
                content=text,
                source_id=f"{conv_id}-{agent}",
                rule_id=None,
                supports=[],
                not_sufficient_for=[],
                access_scope=[agent, "coordinator"],
                token_cost=token_count(text) + 8,
            )
            units.append(u)
            for d in dia_ids:
                dia_to_unit[d] = u
    return units, dia_to_unit


def qa_witness_bundles(qa_list, dia_to_unit):
    """Witness bundles from official QA evidence turn ids (e.g., "D1:3"): the
    session units containing the evidenced turns jointly support the question
    (AND structure; alternative paths are rare in official evidence lists)."""
    tasks = []
    for qa in qa_list:
        facts = []
        for ev in qa.evidence or []:
            u = dia_to_unit.get(ev.strip())
            if u is not None:
                facts.append(u.fact_id)
        if not facts:
            continue
        # one AND bundle over the evidenced facts (multiple evidence = AND;
        # alternative paths rarely present in official evidence lists)
        tasks.append({
            "question": qa.question,
            "answer": str(qa.answer) if qa.answer is not None
                      else "Not mentioned in the conversation",
            "category": qa.category,
            "adversarial": (str(qa.adversarial_answer)
                            if qa.adversarial_answer else None),
            "facts": list(dict.fromkeys(facts)),
        })
    return tasks


def build_allocations(units, tasks_by_question, budget_ratio, seed):
    """Three strategies under equal budget.

    Returns dict strategy -> {agent: [units]}.
    """
    rng = random.Random(seed)
    total_cost = sum(u.token_cost for u in units)
    budget = int(total_cost * budget_ratio)
    per_agent = budget // 3

    # 1) origin sharding (E-mem-like: each agent keeps its own blocks)
    origin = {}
    for u in units:
        agent = u.fact_id.split("_")[1] if "_" in u.fact_id else "agent_a"
        # fact_id = conv_agent_bN -> agent part
        parts = u.fact_id.split("_")
        agent = parts[1] if len(parts) > 1 and parts[1] in LOCOMO_AGENTS else "agent_a"
        if agent == "coordinator":
            agent = "agent_a"
        origin.setdefault(agent, [])
        if sum(x.token_cost for x in origin[agent]) + u.token_cost <= per_agent:
            origin[agent].append(u)

    # 2) frequency replication: replicate blocks by dev QA evidence frequency
    freq = Counter()
    for t in tasks_by_question.values():
        for f in t["facts"]:
            freq[f] += 1
    by_fact = {u.fact_id: u for u in units}
    ranked = sorted(units, key=lambda u: -freq.get(u.fact_id, 0))
    frep = {}
    for u in ranked:
        for agent in u.access_scope:
            if agent == "coordinator":
                continue
            frep.setdefault(agent, [])
            if sum(x.token_cost for x in frep[agent]) + u.token_cost <= per_agent and \
               sum(x.token_cost for us in frep.values() for x in us) + u.token_cost <= budget:
                frep[agent].append(u)
                break

    # 3) CORE-MAS witness-aware: reuse the allocator machinery by treating each
    # question's evidenced facts as an AND bundle; bundle-aware greedy over
    # fault-weighted expected coverage
    core = _core_mas_alloc(units, tasks_by_question, budget, per_agent, rng)
    return {"emem_like": origin, "frequency": frep, "core_mas": core}, \
        {"emem_like": origin, "frequency": frep, "core_mas": core}


def _core_mas_alloc(units, tasks_by_question, budget, per_agent, rng):
    from .failure_scenarios import FailureSpec  # not used; simple MC inside
    specs = [None, "s1"]

    def coverage(alloc):
        # expected witness coverage over questions: a question covered if all
        # its evidenced facts present among survivors' units
        total = 0.0
        for t in tasks_by_question.values():
            if not t["facts"]:
                continue
            for spec in specs:
                if spec is None:
                    have = {u.fact_id for us in alloc.values() for u in us}
                    if all(f in have for f in t["facts"]):
                        total += 0.5
                else:
                    for victim in ("agent_a", "agent_b", "agent_c"):
                        have = {u.fact_id for a, us in alloc.items()
                                if a != victim for u in us}
                        if all(f in have for f in t["facts"]):
                            total += 0.5 / 3
        return total / max(1, len(tasks_by_question))

    by_fact = {u.fact_id: u for u in units}
    alloc = {}
    # bundle actions: single facts + full question bundles
    actions = []
    seen = set()
    for t in tasks_by_question.values():
        fs = t["facts"]
        if fs and tuple(fs) not in seen and len(fs) <= 4:
            seen.add(tuple(fs))
            for agent in ("agent_a", "agent_b", "agent_c"):
                actions.append(([by_fact[f] for f in fs if f in by_fact], agent))
    for u in units:
        # origin agent for this unit (coordinator is the merger, not a store)
        agent = u.fact_id.split("_")[1] if len(u.fact_id.split("_")) > 1 else "agent_a"
        if agent not in ("agent_a", "agent_b", "agent_c"):
            agent = u.access_scope[0] if u.access_scope else "agent_a"
        actions.append(([u], agent))

    def fits(action):
        us, agent = action
        new = sum(u.token_cost for u in us)
        cur_agent = sum(u.token_cost for u in alloc.get(agent, []))
        cur_total = sum(u.token_cost for us2 in alloc.values() for u in us2)
        return cur_agent + new <= per_agent and cur_total + new <= budget

    while True:
        best, best_score = None, 0.0
        base = coverage(alloc)
        for action in actions:
            if not fits(action):
                continue
            trial = {k: list(v) for k, v in alloc.items()}
            trial.setdefault(action[1], []).extend(action[0])
            gain = coverage(trial) - base
            cost = sum(u.token_cost for u in action[0]) or 1
            if gain / cost > best_score:
                best, best_score = action, gain / cost
        if best is None or best_score <= 0:
            break
        alloc.setdefault(best[1], []).extend(best[0])
    return alloc


def run_locomo(conv, qa_tasks, strategy_alloc, llm, metric_fn, victim=None,
               n_questions=None):
    """ECCR protocol on one conversation; grades with official metric_fn."""
    rows = []
    tasks = qa_tasks if n_questions is None else qa_tasks[:n_questions]
    for t in tasks:
        accessible = {a: strategy_alloc.get(a, []) for a in LOCOMO_AGENTS
                      if a != victim}
        evidence = []
        for a in LOCOMO_AGENTS:
            if a not in accessible or not accessible[a]:
                continue
            mem = "\n".join(f"[{u.unit_id}|{u.source_id}] {u.content}"
                            for u in accessible[a])[:6000]
            resp = llm.chat(AGENT_SYSTEM,
                            f"Your memory:\n{mem}\n\nQuestion: {t['question']}")
            if "NO_RELEVANT_EVIDENCE" in resp.upper():
                continue
            evidence.append(f"Agent {a}:\n{resp}")
        ev_text = "\n\n".join(evidence) if evidence else "(no evidence available)"
        coord = llm.chat(LOCOMO_COORD_SYSTEM,
                         f"Question: {t['question']}\n\nEvidence:\n{ev_text[:8000]}")
        pred = extract_loco_answer(coord)
        metrics = metric_fn(pred, t["answer"])
        rows.append({"question": t["question"][:80], "category": t["category"],
                     "victim": victim or "s0", "prediction": pred[:200],
                     "reference": t["answer"][:120], **{k: v for k, v in metrics.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="/root/autodl-tmp/core-mas/locomo10.json")
    ap.add_argument("--emem-repo", default="/root/autodl-tmp/core-mas/emem")
    ap.add_argument("--out-dir", default="run_e7")
    ap.add_argument("--convs", type=int, default=5)
    ap.add_argument("--questions-per-conv", type=int, default=12)
    ap.add_argument("--budget-ratio", type=float, default=0.6)
    ap.add_argument("--llm", choices=["mock", "http"], default="http")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.emem_repo)
    from evaluation.locomo.utils import calculate_metrics  # official metrics

    llm = HTTPOpenAILLM() if args.llm == "http" else None
    data = load_locomo(args.locomo)
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
        for sid, alloc in strats.items():
            for victim in [None, "agent_a", "agent_b", "agent_c"]:
                rows = run_locomo(conv, qa_tasks, alloc, llm, calculate_metrics,
                                  victim=victim, n_questions=args.questions_per_conv)
                for r in rows:
                    r["conv"] = ci
                    r["strategy"] = sid
                all_rows.extend(rows)
        print(f"conv {ci} done ({time.time()-t0:.0f}s elapsed)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e7_locomo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(out / "e7_locomo.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=1)
    print(f"saved {len(all_rows)} rows")


if __name__ == "__main__":
    main()
