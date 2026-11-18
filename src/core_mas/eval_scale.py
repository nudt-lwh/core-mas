"""E1_scale: structural scalability grid.

M suppliers x N department sub-team agents, capacity 0.8, failures
s0 / s1 (single amnesia) / s1x2 (double amnesia) / s2 (unit loss).
Strategies: core_mas, frequency_replication, uniform_sharding everywhere;
task_aware_item for M <= 32. CP-SAT exact (60s timeout) at (M,N) in
CP_SAT_CELLS for the scalability-vs-exactness comparison.

Run: python -m core_mas.eval_scale --out-dir experiment_data/runs/run_e1_scale
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import random
import time
from pathlib import Path

from .allocators import STRATEGIES, Budgets, expected_coverage
from .datagen import _agent_team, generate
from .failure_scenarios import FailureSpec

M_GRID = [8, 16, 32, 64, 128]
N_GRID = [3, 6, 9]
CAPACITY = 0.8
# CP-SAT cells: (n_suppliers, n_agents); timeout 60s
CP_SAT_CELLS = [(32, 3), (64, 3)]
CP_SAT_TIMEOUT = 60.0


def make_budgets_n(world, cap: float, agent_set) -> Budgets:
    total = int(world.baseline_cost * cap)
    per = max(1, total // len(agent_set))
    return Budgets(per_agent={a: per for a in agent_set}, total=total)


def run_cell(job) -> list:
    n_supp, n_ag, seed = job
    t_cell = time.time()
    world = generate(seed, n_agents=n_ag, n_suppliers=n_supp)
    agents = _agent_team(n_ag) + ["coordinator"]
    budgets = make_budgets_n(world, CAPACITY, agents)
    strategies = ["core_mas", "frequency_replication", "uniform_sharding"]
    if n_supp <= 32:
        strategies.append("task_aware_item")
    rows = []
    for sid in strategies:
        t0 = time.time()
        alloc = STRATEGIES[sid](world.units, agents, budgets, world.dev_tasks, world)
        dt = time.time() - t0
        rng = random.Random(42)
        cov_s0 = expected_coverage(world.test_tasks, alloc, FailureSpec("s0"), agents, rng)
        cov_s1 = expected_coverage(world.test_tasks, alloc, FailureSpec("s1"), agents, rng)
        cov_s1x2 = expected_coverage(world.test_tasks, alloc, FailureSpec("s1x2"), agents, rng)
        cov_s2 = expected_coverage(world.test_tasks, alloc, FailureSpec("s2"), agents, rng,
                                   s2_samples=32)
        rows.append({
            "n_suppliers": n_supp, "n_agents": n_ag, "seed": seed,
            "capacity": CAPACITY, "strategy": sid,
            "n_units": len(world.units), "baseline_cost": world.baseline_cost,
            "cov_s0": round(cov_s0, 4), "cov_s1": round(cov_s1, 4),
            "cov_s1x2": round(cov_s1x2, 4), "cov_s2": round(cov_s2, 4),
            "fw": round((cov_s1 + cov_s2) / 2, 4),
            "tokens_used": budgets.total_used(alloc), "budget": budgets.total,
            "alloc_seconds": round(dt, 2),
        })
    print(f"[cell] M={n_supp} N={n_ag} seed={seed} done in {time.time()-t_cell:.1f}s",
          flush=True)
    return rows


def run_cp_sat(job) -> dict:
    n_supp, n_ag, seed = job
    from .optimality_gap import exact_solution, reduce_world
    t0 = time.time()
    world = generate(seed, n_agents=n_ag, n_suppliers=n_supp)
    agents = _agent_team(n_ag) + ["coordinator"]
    budgets = make_budgets_n(world, CAPACITY, agents)
    # reduce to the same first-n suppliers instance used by M12 (whole world here:
    # at scale we solve the FULL generated world, not a supplier subset)
    units, dev_tasks = world.units, world.dev_tasks
    exact, status = exact_solution(units, dev_tasks, budgets, agents,
                                   timeout=CP_SAT_TIMEOUT)
    heur_alloc = STRATEGIES["core_mas"](units, agents, budgets, dev_tasks, world)
    rng = random.Random(42)
    heur = expected_coverage(dev_tasks, heur_alloc, FailureSpec("s0"), agents, rng)
    return {
        "n_suppliers": n_supp, "n_agents": n_ag, "seed": seed,
        "cp_sat_status": status,
        "cp_sat_seconds": round(time.time() - t0, 1),
        "exact_s0": round(exact, 4) if exact is not None else None,
        "heur_s0": round(heur, 4),
        "gap": round((exact - heur) / exact, 4) if exact else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiment_data/runs/run_e1_scale")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(m, n, args.seed) for m in M_GRID for n in N_GRID if not (m == 128 and n == 9)]
    print(f"grid: {len(jobs)} cells", flush=True)
    t_start = time.time()
    rows = []
    with multiprocessing.Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            rows.extend(r)
    rows.sort(key=lambda r: (r["n_suppliers"], r["n_agents"], r["strategy"]))
    with open(out / "e1_scale.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"grid done in {time.time()-t_start:.0f}s -> {out / 'e1_scale.csv'}", flush=True)

    cp_jobs = [(m, n, args.seed) for m, n in CP_SAT_CELLS]
    cp_rows = []
    for j in cp_jobs:
        print(f"CP-SAT cell {j}", flush=True)
        cp_rows.append(run_cp_sat(j))
    with open(out / "e1_scale_cpsat.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cp_rows[0].keys()))
        w.writeheader()
        w.writerows(cp_rows)
    print(f"cp-sat done -> {out / 'e1_scale_cpsat.csv'}", flush=True)


if __name__ == "__main__":
    main()
