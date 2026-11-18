"""C-A gap attribution with agent-level logging (Type I/II/III classification).

For structurally-sufficient-but-LLM-wrong tasks (C=1 & A=0), classify the
failure into:
  Type I   agent omission: a surviving agent holding relevant evidence
           returned NO_RELEVANT_EVIDENCE (or an empty reply)
  Type II  coordinator false refusal: agents returned evidence, but the
           coordinator still answered INSUFFICIENT_EVIDENCE
  Type III reasoning error: coordinator answered but wrongly

Run (env-configured LLM, e.g. cloud llama 4B on 6006 or DeepSeek API):
  python -m core_mas.diag_attribution --out experiment_data/runs/diag_attribution.csv \
      --seed 200 --tasks 40 --caps 0.8 1.0 --strategy core_mas \
      --conditions s0 s1_finance s1_logistics
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import (AGENT_SYSTEM, COORD_SYSTEM, EvalAgents, HTTPOpenAILLM,
                          extract_answer, grade)
from .eval_structural import AGENT_SET, make_budgets
from .witness_graph import register_req_kinds, requirements_satisfied

VICTIM_MAP = {"s0": None, "s1_finance": "finance",
              "s1_logistics": "logistics", "s1_compliance": "compliance"}


def holds_any_req(units, task) -> bool:
    """Does this agent's memory satisfy at least one (fact, kind) requirement
    of any witness bundle of the task?"""
    satisfied = requirements_satisfied(units, None)
    for bundle in task.witness_bundles:
        for req in bundle:
            if req in satisfied:
                return True
    return False


def run_task_diag(ev: EvalAgents, task, victim) -> dict:
    accessible = ev.accessible_units(victim)
    agent_log = []
    type1 = False
    for a in ev.agents:
        units = accessible.get(a)
        if not units:
            continue
        holds = holds_any_req(units, task)
        mem_text = "\n".join(
            f"[{u.unit_id}|src:{u.source_id}|rule:{u.rule_id or '-'}] "
            f"{u.content}" for u in units)
        resp = ev.llm.chat(AGENT_SYSTEM,
                           f"Your memory:\n{mem_text}\n\nQuestion: {task.question}")
        omitted = ("NO_RELEVANT_EVIDENCE" in resp.upper()) or not resp.strip()
        agent_log.append({"agent": a, "holds_req": holds, "omitted": omitted,
                          "resp": resp[:200]})
        if holds and omitted:
            type1 = True
    # coordinator step (same as run_task)
    evidence = [f"Agent {l['agent']}:\n{l['resp']}"
                for l in agent_log if not l["omitted"]]
    ev_text = "\n\n".join(evidence) if evidence else "(no evidence available)"
    coord = ev.llm.chat(COORD_SYSTEM,
                        f"Question: {task.question}\n\nEvidence:\n{ev_text}")
    ans = extract_answer(coord)
    refused = ans.startswith("insufficient")
    structural = ev.structural_answer(task, accessible)
    wrong = not grade(coord, task.correct_answer)
    if grade(structural, task.correct_answer) and wrong:
        if type1:
            etype = "I_agent_omission"
        elif refused:
            etype = "II_coord_false_refusal"
        else:
            etype = "III_reasoning_error"
    else:
        etype = "n/a"
    return {"task_id": task.task_id, "victim": victim, "coord_answer": coord,
            "correct": task.correct_answer, "correct_llm": not wrong,
            "correct_structural": grade(structural, task.correct_answer),
            "error_type": etype, "type1_flag": int(type1),
            "n_agents_queried": len(agent_log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment_data/runs/diag_attribution.csv")
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--caps", nargs="+", type=float, default=[0.8, 1.0])
    ap.add_argument("--strategy", default="core_mas")
    ap.add_argument("--conditions", nargs="+",
                    default=["s0", "s1_finance", "s1_logistics"])
    args = ap.parse_args()

    world = generate(args.seed)
    register_req_kinds(world.test_tasks + world.dev_tasks)
    llm = HTTPOpenAILLM()
    print(f"llm -> {llm.base_url} model={llm.model}", flush=True)

    rows = []
    t0 = time.time()
    for cap in args.caps:
        budgets = make_budgets(world, cap)
        alloc = STRATEGIES[args.strategy](world.units, AGENT_SET, budgets,
                                          world.dev_tasks, world)
        ev = EvalAgents(world, alloc, llm, AGENT_SET)
        for task in world.test_tasks[:args.tasks]:
            for cond in args.conditions:
                r = run_task_diag(ev, task, VICTIM_MAP[cond])
                rows.append({"seed": args.seed, "capacity": cap,
                             "condition": cond, "task_id": task.task_id,
                             "structural": int(r["correct_structural"]),
                             "llm_correct": int(r["correct_llm"]),
                             "error_type": r["error_type"],
                             "type1_flag": r["type1_flag"],
                             "n_agents_queried": r["n_agents_queried"]})
        print(f"cap {cap} done ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    c1a0 = [r for r in rows if r["structural"] and not r["llm_correct"]]
    types = Counter(r["error_type"] for r in c1a0)
    n_c1 = sum(1 for r in rows if r["structural"])
    n_a1 = sum(1 for r in rows if r["structural"] and r["llm_correct"])
    print(f"\nsummary: n={len(rows)} C=1: {n_c1} C=1→A=1: {n_a1} "
          f"({n_a1/max(1,n_c1):.3f})", flush=True)
    print(f"C=1&A=0 (n={len(c1a0)}): {dict(types)}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
