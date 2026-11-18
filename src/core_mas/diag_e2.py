"""E2 diagnostics: dump raw agent/coordinator exchanges for manual inspection.

Run: PYTHONPATH=src python -m core_mas.diag_e2 --tasks 5
Prints every prompt and response so we can see WHERE the LLM layer breaks.
"""

from __future__ import annotations

import argparse

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import (AGENT_SYSTEM, COORD_SYSTEM, EvalAgents,
                          HTTPOpenAILLM, grade)
from .eval_structural import AGENT_SET, make_budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--strategy", default="core_mas")
    ap.add_argument("--capacity", type=float, default=1.0)
    args = ap.parse_args()

    world = generate(200)
    budgets = make_budgets(world, args.capacity)
    alloc = STRATEGIES[args.strategy](world.units, AGENT_SET, budgets,
                                      world.dev_tasks, world)
    llm = HTTPOpenAILLM()
    ev = EvalAgents(world, alloc, llm, AGENT_SET)

    for task in world.test_tasks[:args.tasks]:
        print("=" * 100)
        print(f"TASK: {task.question}  | correct: {task.correct_answer}")
        accessible = ev.accessible_units(None)
        for a in AGENT_SET:
            if not accessible.get(a):
                print(f"[{a}]: (no units)")
                continue
            mem_text = "\n".join(
                f"[{u.unit_id}|src:{u.source_id}|rule:{u.rule_id or '-'}] {u.content}"
                for u in accessible[a])
            resp = llm.chat(AGENT_SYSTEM, f"Your memory:\n{mem_text}\n\nQuestion: {task.question}")
            print(f"[{a} units={len(accessible[a])}] -> {resp[:400]}")
        r = ev.run_task(task, victim=None)
        print(f"COORDINATOR: {r['coord_answer'][:300]!r}")
        print(f"structural={r['structural_answer']} correct={r['correct']} "
              f"llm_correct={r['correct_llm']} unsupported={r['unsupported_llm']}")
        print()


if __name__ == "__main__":
    main()
