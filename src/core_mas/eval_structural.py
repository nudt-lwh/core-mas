"""E1 structural evaluation: witness coverage of all strategies under budget
and failure, over frozen test tasks. Pure CPU, deterministic (fixed rng seeds).

Multiprocessing over (seed, capacity) with a capped worker count so the run
stays in the background without saturating the machine.

Run: python -m core_mas.eval_structural --out-dir <dir> [--seeds 200 201 202] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
import time
from pathlib import Path

from .allocators import STRATEGIES, Budgets, expected_coverage
from .datagen import AGENTS, generate
from .failure_scenarios import FailureSpec

AGENT_SET = AGENTS + ["coordinator"]
CAPACITIES = [0.4, 0.6, 0.8, 1.0]
TEST_SEEDS = [200, 201, 202]


def make_budgets(world, cap: float) -> Budgets:
    total = int(world.baseline_cost * cap)
    per = max(1, total // len(AGENT_SET))
    return Budgets(per_agent={a: per for a in AGENT_SET}, total=total)


def run_combo(seed: int, cap: float) -> list:
    world = generate(seed)
    budgets = make_budgets(world, cap)
    rows = []
    for sid, fn in STRATEGIES.items():
        t0 = time.time()
        alloc = fn(world.units, AGENT_SET, budgets, world.dev_tasks, world)
        rng = random.Random(42)
        cov_s0 = expected_coverage(world.test_tasks, alloc,
                                   FailureSpec("s0"), AGENT_SET, rng)
        cov_s1 = expected_coverage(world.test_tasks, alloc,
                                   FailureSpec("s1"), AGENT_SET, rng)
        cov_s2 = expected_coverage(world.test_tasks, alloc,
                                   FailureSpec("s2"), AGENT_SET, rng,
                                   s2_samples=32)
        rows.append({
            "seed": seed, "capacity": cap, "strategy": sid,
            "cov_s0": round(cov_s0, 4), "cov_s1": round(cov_s1, 4),
            "cov_s2": round(cov_s2, 4),
            "fault_weighted": round((cov_s1 + cov_s2) / 2, 4),
            "tokens_used": budgets.total_used(alloc), "budget": budgets.total,
            "alloc_seconds": round(time.time() - t0, 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiment_data/runs/run_e1")
    ap.add_argument("--seeds", nargs="+", type=int, default=TEST_SEEDS)
    ap.add_argument("--capacities", nargs="+", type=float, default=CAPACITIES)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    combos = [(s, c) for s in args.seeds for c in args.capacities]
    t_start = time.time()
    rows = []
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            for r in pool.starmap(run_combo, combos):
                rows.extend(r)
    else:
        for s, c in combos:
            rows.extend(run_combo(s, c))

    with open(out / "e1_coverage.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "seeds": args.seeds, "capacities": args.capacities,
                   "agents": AGENT_SET,
                   "note": "structural witness coverage; s1 exact over victims, "
                           "s2 MC 32 samples seed 42"}, f, ensure_ascii=False, indent=1)
    with open(out / "e1_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # quick summary for the log
    print(f"\n=== E1 summary ({time.time()-t_start:.0f}s) ===")
    for s in args.seeds:
        for c in args.capacities:
            sub = [r for r in rows if r["seed"] == s and r["capacity"] == c]
            cm = next(r for r in sub if r["strategy"] == "core_mas")
            ta = next(r for r in sub if r["strategy"] == "task_aware_item")
            fr = next(r for r in sub if r["strategy"] == "frequency_replication")
            print(f"seed={s} cap={c:.0%}: core_mas {cm['fault_weighted']:.3f} | "
                  f"task_aware {ta['fault_weighted']:.3f} | freq {fr['fault_weighted']:.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
