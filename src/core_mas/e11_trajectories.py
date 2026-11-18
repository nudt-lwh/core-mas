"""E11: rebuild case-study trajectories with full logging (DeepSeek API).

Runs the E2 protocol (ECCR) on the selected case IDs and logs every
prompt/response so the appendix can show real success/failure trajectories.
Case pool: experiment_data/runs/run_e2_full_cloud_cases.json.

Run (local, API): LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat
  LLM_API_KEY=$(cat <project>/deepseek_key.txt) PYTHONPATH=src
  python -m core_mas.e11_trajectories --out-dir experiment_data/runs/run_e11
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import (AGENT_SYSTEM, COORD_SYSTEM_V2, HTTPOpenAILLM,
                          extract_answer, full_gate, get_records,
                          record_line, targeted_repair)
from .eval_structural import AGENT_SET, make_budgets


def run_case(world, strategy, cap, task, victim, llm, log_lines):
    budgets = make_budgets(world, cap)
    alloc = STRATEGIES[strategy](world.units, AGENT_SET, budgets,
                                 world.dev_tasks, world)
    log_lines.append(f"=== CASE: task={task.task_id} strategy={strategy} "
                     f"cap={cap} victim={victim} ===")
    log_lines.append(f"QUESTION: {task.question} | correct: {task.correct_answer}")
    accessible = {a: alloc.get(a, []) for a in AGENT_SET if a != victim}
    responses = []
    for a in AGENT_SET:
        if a not in accessible or not accessible[a]:
            if a == victim:
                log_lines.append(f"[{a}] MEMORY UNAVAILABLE (failure)")
            continue
        mem_text = "\n".join(
            f"[{u.unit_id}|src:{u.source_id}|rule:{u.rule_id or '-'}] {u.content}"
            for u in accessible[a])
        resp = llm.chat(AGENT_SYSTEM,
                        f"Your memory:\n{mem_text}\n\nQuestion: {task.question}")
        log_lines.append(f"[{a}] units={len(accessible[a])}\n  -> {resp[:500]}")
        if "NO_RELEVANT_EVIDENCE" in resp.upper():
            continue
        responses.append((a, resp))
    # ECCR-v2: verify -> gate -> targeted repair -> answer/refuse
    verified = []
    for a, resp in responses:
        verified.extend(get_records(resp, accessible.get(a, [])))
    log_lines.append(f"VERIFIED records: {len(verified)}")
    sufficient = full_gate(task, verified)
    repaired, sufficient2 = (verified, True) if sufficient else \
        targeted_repair(task, verified, accessible)
    log_lines.append(f"GATE: {'PASS' if sufficient else 'FAIL'} -> "
                     f"repair +{len(repaired)-len(verified)} -> "
                     f"{'SUFFICIENT' if sufficient2 else 'INSUFFICIENT'}")
    if sufficient2:
        ev_text = "\n\n".join(record_line(u) for u in repaired)
        coord = llm.chat(COORD_SYSTEM_V2,
                         f"Question: {task.question}\n\nEvidence:\n{ev_text[:8000]}")
    else:
        coord = "ANSWER: INSUFFICIENT_EVIDENCE"
    log_lines.append(f"COORDINATOR -> {coord[:400]}")
    from .eval_agents import grade
    log_lines.append(f"graded: {grade(coord, task.correct_answer)} "
                     f"(answer={extract_answer(coord)!r}, correct={task.correct_answer!r})")
    log_lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="experiment_data/runs/run_e2_full_cloud_cases.json")
    ap.add_argument("--out-dir", default="experiment_data/runs/run_e11")
    ap.add_argument("--cap", type=float, default=0.8)
    args = ap.parse_args()
    cases = json.load(open(args.cases, encoding="utf-8"))
    llm = HTTPOpenAILLM()
    world = generate(200)
    tasks = {t.task_id: t for t in world.test_tasks}
    log_lines = []
    for cls, keys in cases.items():
        for (seed, task_id) in keys:
            if seed != "200":
                continue
            t = tasks[task_id]
            # one no-fault and one fault trajectory per case, alternating strategy
            run_case(world, "core_mas", args.cap, t, None, llm, log_lines)
            run_case(world, "task_aware_item", args.cap, t, "finance", llm, log_lines)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"saved {len(log_lines)} lines to {out}/trajectories.txt")


if __name__ == "__main__":
    main()
