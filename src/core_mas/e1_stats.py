"""E1 statistics: per-scenario breakdown + paired bootstrap CIs.

Recomputes per-task coverage vectors (deterministic) for the three key
strategies, then does task-level paired bootstrap on the fault-weighted
difference (core_mas - baseline). R=1000, fixed seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .allocators import STRATEGIES, expected_coverage
from .datagen import generate
from .eval_structural import AGENT_SET, make_budgets
from .failure_scenarios import FailureSpec

SEEDS = [200, 201, 202]
CAPS = [0.4, 0.6, 0.8, 1.0]
BASELINES = ["frequency_replication", "task_aware_item"]


def per_task_coverage(tasks, alloc, spec, agents, rng_seed, s2_samples=32):
    """Per-task coverage vector under one failure spec (deterministic)."""
    from .witness_graph import register_req_kinds, requirements_satisfied
    from .failure_scenarios import accessible_units
    from .memory_units import AgentMemory

    register_req_kinds(tasks)
    mems = {a: AgentMemory(a, {u.unit_id: u for u in alloc.get(a, [])}) for a in agents}
    rng = random.Random(rng_seed)
    out = []
    for task in tasks:
        if spec.kind == "s0":
            units = accessible_units(mems, spec, rng)
            out.append(1.0 if task.covered_by(requirements_satisfied(units, None)) else 0.0)
        elif spec.kind == "s1":
            hits = 0
            for victim in agents:
                mems_v = {a: mems[a] for a in agents if a != victim}
                units = [u for m in mems_v.values() for u in m.units.values()]
                hits += task.covered_by(requirements_satisfied(units, None))
            out.append(hits / len(agents))
        else:
            hits = 0
            for _ in range(s2_samples):
                units = accessible_units(mems, spec, rng)
                hits += task.covered_by(requirements_satisfied(units, None))
            out.append(hits / s2_samples)
    return out


def bootstrap_diff_paired(a, b, rng, R=1000):
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    means = []
    for _ in range(R):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(d[i] for i in idx) / n)
    means.sort()
    return means[int(0.025 * R)], means[int(0.975 * R)]


def main():
    out = {}
    for seed in SEEDS:
        world = generate(seed)
        out[seed] = {}
        for cap in CAPS:
            budgets = make_budgets(world, cap)
            vecs = {}
            for sid in ["core_mas"] + BASELINES:
                alloc = STRATEGIES[sid](world.units, AGENT_SET, budgets,
                                        world.dev_tasks, world)
                v1 = per_task_coverage(world.test_tasks, alloc,
                                       FailureSpec("s1"), AGENT_SET, 42)
                v2 = per_task_coverage(world.test_tasks, alloc,
                                       FailureSpec("s2"), AGENT_SET, 42)
                v0 = per_task_coverage(world.test_tasks, alloc,
                                       FailureSpec("s0"), AGENT_SET, 42)
                vecs[sid] = {"s0": v0, "s1": v1, "s2": v2,
                             "fw": [(a + b) / 2 for a, b in zip(v1, v2)]}
            cell = {}
            for sid in ["core_mas"] + BASELINES:
                cell[sid] = {k: round(sum(v) / len(v), 4) for k, v in vecs[sid].items()}
            rng = random.Random(999)
            for bl in BASELINES:
                lo, hi = bootstrap_diff_paired(vecs["core_mas"]["fw"],
                                               vecs[bl]["fw"], rng)
                cell[f"fw_diff_vs_{bl}_ci95"] = [round(lo, 4), round(hi, 4)]
            out[seed][cap] = cell
            print(f"seed={seed} cap={cap:.0%}: cm s0/s1/s2 = "
                  f"{cell['core_mas']['s0']}/{cell['core_mas']['s1']}/{cell['core_mas']['s2']} | "
                  f"vs ta CI95 {cell['fw_diff_vs_task_aware_item_ci95']} | "
                  f"vs freq CI95 {cell['fw_diff_vs_frequency_replication_ci95']}")
    Path("experiment_data/runs/run_e1/e1_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved e1_stats.json")


if __name__ == "__main__":
    main()
