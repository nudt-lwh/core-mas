# -*- coding: utf-8 -*-
"""core_mas vs traditional_placement（及 central_replication）配对 bootstrap CI。

任务级配对、1000 次重采样、95% CI；每容量每种子 + 种子合并。
输出: experiment_data/runs/run_e1/ci_vs_traditional.json
"""
import json
import random
import sys
from pathlib import Path

from core_mas.allocators import STRATEGIES
from core_mas.datagen import generate
from core_mas.e1_stats import per_task_coverage, bootstrap_diff_paired
from core_mas.eval_structural import AGENT_SET, make_budgets
from core_mas.failure_scenarios import FailureSpec

SEEDS = [200, 201, 202]
CAPS = [0.4, 0.6, 0.8, 1.0]
BASELINES = ["traditional_placement", "central_replication", "frequency_replication"]


def main():
    out = {"seeds": SEEDS, "note": "task-level paired bootstrap, R=1000, "
           "fw=(s1 exact + s2 MC32)/2 per task"}
    for cap in CAPS:
        out[cap] = {}
        for bl in BASELINES:
            out[cap][bl] = {}
            for seed in SEEDS:
                world = generate(seed)
                budgets = make_budgets(world, cap)
                a = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                           world.dev_tasks, world)
                b = STRATEGIES[bl](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
                va1 = per_task_coverage(world.test_tasks, a, FailureSpec("s1"),
                                        AGENT_SET, 42)
                va2 = per_task_coverage(world.test_tasks, a, FailureSpec("s2"),
                                        AGENT_SET, 42)
                vb1 = per_task_coverage(world.test_tasks, b, FailureSpec("s1"),
                                        AGENT_SET, 42)
                vb2 = per_task_coverage(world.test_tasks, b, FailureSpec("s2"),
                                        AGENT_SET, 42)
                va = [(x + y) / 2 for x, y in zip(va1, va2)]
                vb = [(x + y) / 2 for x, y in zip(vb1, vb2)]
                lo, hi = bootstrap_diff_paired(va, vb, random.Random(999))
                d = sum(x - y for x, y in zip(va, vb)) / len(va)
                out[cap][bl][str(seed)] = {"diff_pp": round(d * 100, 2),
                                           "ci95": [round(lo * 100, 2),
                                                    round(hi * 100, 2)]}
                print(f"cap={cap:.0%} {bl} seed={seed}: "
                      f"{out[cap][bl][str(seed)]}", flush=True)
    Path("experiment_data/runs/run_e1/ci_vs_traditional.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ci_vs_traditional.json")


if __name__ == "__main__":
    main()
