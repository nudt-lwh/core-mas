"""E2 pilot runner (cloud): LLM-layer main comparison on synthetic tasks.

Strategies: frequency / task-aware / core_mas (oracle optional) at capacities
0.8/1.0, failures s0 + s1 (all victims), N pilot tasks per seed.
Metrics: acc_normal (M1), acc_fault (M2), paired drop (M3), capability
retention (M4), insufficient-evidence rate (M7), unsupported answers (M6
proxy), solver-vs-LLM gap (M13).

Run (cloud, with local llama server on 6006):
  python -m core_mas.run_e2_pilot --out-dir /root/autodl-tmp/core-mas/run_e2 \
      --seeds 200 --tasks 10 --llm http
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import EvalAgents, HTTPOpenAILLM, MockLLM
from .eval_structural import AGENT_SET, make_budgets

STRAT_SUBSET = ["uniform_sharding", "frequency_replication",
                 "task_aware_item", "core_mas"]
# oracle uses TEST witnesses: non-deployable upper bound, cap 1.0 only
ORACLE_CAPS = {1.0}
VICTIMS = ["finance", "compliance", "logistics"]


def run_seed(seed: int, caps, n_tasks: int, llm_mode: str,
             strategies=None, conditions=None, repair_flow=False) -> list:
    world = generate(seed)
    llm = HTTPOpenAILLM() if llm_mode == "http" else MockLLM()
    sids = strategies or STRAT_SUBSET
    conds = conditions or ["s0"] + [f"s1_{v}" for v in VICTIMS]
    rows = []
    for cap in caps:
        budgets = make_budgets(world, cap)
        for sid in sids:
            if sid == "oracle_witness" and cap not in ORACLE_CAPS:
                continue
            alloc = STRATEGIES[sid](world.units, AGENT_SET, budgets,
                                    world.dev_tasks, world)
            ev = EvalAgents(world, alloc, llm, AGENT_SET)
            for task in world.test_tasks[:n_tasks]:
                if "s0" in conds:
                    rows.append(_row(seed, cap, sid, task,
                                     ev.run_task(task, victim=None,
                                                 repair_flow=repair_flow), "s0"))
                for victim in VICTIMS:
                    if f"s1_{victim}" in conds:
                        rows.append(_row(seed, cap, sid, task,
                                         ev.run_task(task, victim=victim,
                                                     repair_flow=repair_flow),
                                         f"s1_{victim}"))
    return rows


def _row(seed, cap, sid, task, r, cond):
    row = {
        "seed": seed, "capacity": cap, "strategy": sid, "condition": cond,
        "task_id": task.task_id, "correct": r["correct"],
        "llm_correct": int(r["correct_llm"]),
        "structural_correct": int(r["correct_structural"]),
        "insufficient": int(r["insufficient_llm"]),
        "unsupported": int(r["unsupported_llm"]),
    }
    if "repair_answer" in r:
        row.update({
            "llm_correct_repair": int(r["correct_llm_repair"]),
            "repair_verified_n": r["repair_verified_n"],
            "repair_repaired_n": r["repair_repaired_n"],
            "repair_gate_pass": int(bool(r["repair_gate_pass"])),
            "insufficient_repair": int(r["insufficient_llm_repair"]),
            "raw_original_answer": r["coord_answer"],
            "raw_repair_answer": r["repair_answer"],
        })
    return row


def summarize(rows) -> str:
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["strategy"], r["condition"])].append(r)
    lines = []
    for (sid, cond), rs in sorted(by.items()):
        if cond == "s0":
            continue
        acc = sum(r["llm_correct"] for r in rs) / len(rs)
        lines.append(f"  {sid:24s} {cond:14s} acc={acc:.3f} n={len(rs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiment_data/runs/run_e2")
    ap.add_argument("--seeds", nargs="+", type=int, default=[200])
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--capacities", nargs="+", type=float, default=[0.8, 1.0])
    ap.add_argument("--llm", choices=["mock", "http"], default="http")
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--repair-flow", action="store_true",
                    help="ECCR v2: verify -> gate -> targeted repair -> answer "
                         "(records both original and repair answers)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for seed in args.seeds:
        rows.extend(run_seed(seed, args.capacities, args.tasks, args.llm,
                             args.strategies, args.conditions,
                             repair_flow=args.repair_flow))
    print(summarize(rows))
    with open(out / f"e2_pilot_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(out / "e2_pilot_latest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
