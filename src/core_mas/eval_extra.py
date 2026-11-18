"""E4 ablations + E5 sensitivity + E10 cost (structural layer, local CPU).

E4: full vs w/o-witness / w/o-failure / w/o-granularity / w/o-bundles /
    AND-only / OR-only / w/o-recovery / w/o-provenance
E5: capacity fine grid; heterogeneous agent failure; s2 loss-prob sweep
E10: per-stage CPU cost (unit extraction, witness building, allocation,
     failure simulation, provenance checks)

Run: python -m core_mas.eval_extra --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
import time
from copy import deepcopy
from pathlib import Path

from .allocators import STRATEGIES, expected_coverage
from .datagen import generate
from .eval_structural import AGENT_SET, make_budgets
from .failure_scenarios import FailureSpec
from .witness_graph import Task, register_req_kinds, requirements_satisfied

SEEDS = [200, 201, 202]


def and_only(tasks):
    return [Task(t.task_id + "_and", t.query_type, t.witness_bundles[:1], t.is_dev)
            for t in tasks]


def or_only(tasks):
    out = []
    for t in tasks:
        bundles = [frozenset({r}) for b in t.witness_bundles for r in b]
        out.append(Task(t.task_id + "_or", t.query_type, bundles, t.is_dev))
    return out


def coverage_no_provenance(tasks, alloc, spec, agents, rng, s2_samples=32):
    """Coverage when predicate provenance checks are skipped (predicates as raw)."""
    from .failure_scenarios import accessible_units
    from .memory_units import AgentMemory

    register_req_kinds(tasks)
    mems = {a: AgentMemory(a, {u.unit_id: u for u in alloc.get(a, [])}) for a in agents}
    total = 0.0
    for task in tasks:
        if spec.kind == "s0":
            units = accessible_units(mems, spec, rng)
            sat = set()
            for u in units:
                for fid in u.covered_facts:
                    for k in _kinds(fid):
                        sat.add((fid, k))
            total += task.covered_by(sat)
        elif spec.kind == "s1":
            hits = 0
            for victim in agents:
                sat = set()
                for a, m in mems.items():
                    if a == victim:
                        continue
                    for u in m.units.values():
                        for fid in u.covered_facts:
                            for k in _kinds(fid):
                                sat.add((fid, k))
                hits += task.covered_by(sat)
            total += hits / len(agents)
        else:
            hits = 0
            for _ in range(s2_samples):
                units = accessible_units(mems, spec, rng)
                sat = set()
                for u in units:
                    for fid in u.covered_facts:
                        for k in _kinds(fid):
                            sat.add((fid, k))
                hits += task.covered_by(sat)
            total += hits / s2_samples
    return total / len(tasks)


_K = {}


def _kinds(fid):
    from .witness_graph import _KINDS_BY_FACT
    return _KINDS_BY_FACT.get(fid, [])


def coverage_no_recovery(tasks, alloc, spec, agents, rng, s2_samples=32):
    """Coverage when survivors do NOT share evidence (single-agent view only).

    s1: exact over victims, each survivor checked alone.
    s2: MC over per-unit loss, each surviving agent checked alone (no sharing).
    s0: any single agent covers.
    """
    register_req_kinds(tasks)
    total = 0.0
    if spec.kind == "s1":
        for task in tasks:
            hits_sum = 0.0
            for victim in agents:
                hit = 0.0
                for survivor in agents:
                    if survivor == victim:
                        continue
                    units = alloc.get(survivor, [])
                    if task.covered_by(requirements_satisfied(units, None)):
                        hit = 1.0
                        break
                hits_sum += hit
            total += hits_sum / len(agents)
        return total / len(tasks)
    if spec.kind == "s2":
        for task in tasks:
            hits = 0
            for _ in range(s2_samples):
                hit = 0.0
                for a in agents:
                    units = [u for u in alloc.get(a, [])
                             if rng.random() >= spec.unit_loss_prob]
                    if task.covered_by(requirements_satisfied(units, None)):
                        hit = 1.0
                        break
                hits += hit
            total += hits / s2_samples
        return total / len(tasks)
    # s0: any single agent covers
    for task in tasks:
        best = 0.0
        for a in agents:
            if task.covered_by(requirements_satisfied(alloc.get(a, []), None)):
                best = 1.0
                break
        total += best
    return total / len(tasks)


ABLATIONS = {
    "full": dict(),
    "w_o_witness": dict(use_witness=False),
    "w_o_failure": dict(use_failure=False),
    "w_o_granularity": dict(use_granularity=False),
    "w_o_bundles": dict(use_bundles=False, use_local_search=False),
}


def run_ablations(seed: int, cap: float) -> list:
    world = generate(seed)
    budgets = make_budgets(world, cap)
    rows = []
    for name, kw in ABLATIONS.items():
        t0 = time.time()
        alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                       world.dev_tasks, world, **kw)
        rng = random.Random(42)
        fw = (expected_coverage(world.test_tasks, alloc, FailureSpec("s1"),
                                AGENT_SET, rng) +
              expected_coverage(world.test_tasks, alloc, FailureSpec("s2"),
                                AGENT_SET, rng, s2_samples=32)) / 2
        rows.append({"exp": "E4", "seed": seed, "capacity": cap,
                     "variant": name, "fw": round(fw, 4),
                     "seconds": round(time.time() - t0, 2)})
    # structural variants of witness shape
    for name, transform in (("and_only", and_only), ("or_only", or_only)):
        dev_t = transform(world.dev_tasks)
        test_t = transform(world.test_tasks)
        t0 = time.time()
        alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets, dev_t, world)
        rng = random.Random(42)
        fw = (expected_coverage(test_t, alloc, FailureSpec("s1"), AGENT_SET, rng) +
              expected_coverage(test_t, alloc, FailureSpec("s2"), AGENT_SET, rng,
                                s2_samples=32)) / 2
        rows.append({"exp": "E4", "seed": seed, "capacity": cap,
                     "variant": name, "fw": round(fw, 4),
                     "seconds": round(time.time() - t0, 2)})
    # w/o recovery + w/o provenance on the FULL allocation
    alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    rng = random.Random(42)
    # w/o recovery under the SAME fault-weighted metric as the full method:
    # (s1 exact no-sharing + s2 MC no-sharing) / 2
    fw_nr = (coverage_no_recovery(world.test_tasks, alloc,
                                  FailureSpec("s1"), AGENT_SET, rng) +
             coverage_no_recovery(world.test_tasks, alloc,
                                  FailureSpec("s2"), AGENT_SET, rng,
                                  s2_samples=32)) / 2
    rows.append({"exp": "E4", "seed": seed, "capacity": cap, "variant": "w_o_recovery",
                 "fw": round(fw_nr, 4),
                 "seconds": 0.0})
    rows.append({"exp": "E4", "seed": seed, "capacity": cap, "variant": "w_o_provenance",
                 "fw": round(coverage_no_provenance(world.test_tasks, alloc,
                                                    FailureSpec("s1"), AGENT_SET, rng), 4),
                 "seconds": 0.0})
    return rows


def run_sensitivity(seed: int) -> list:
    world = generate(seed)
    rows = []
    # capacity fine grid at fault-weighted coverage
    for cap in [0.3, 0.5, 0.7, 0.9]:
        budgets = make_budgets(world, cap)
        alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                       world.dev_tasks, world)
        rng = random.Random(42)
        fw = (expected_coverage(world.test_tasks, alloc, FailureSpec("s1"),
                                AGENT_SET, rng) +
              expected_coverage(world.test_tasks, alloc, FailureSpec("s2"),
                                AGENT_SET, rng, s2_samples=32)) / 2
        rows.append({"exp": "E5", "seed": seed, "axis": "capacity",
                     "value": cap, "fw": round(fw, 4)})
    # heterogeneous agent failure probabilities (s1)
    het = FailureSpec("s1", agent_probs={"finance": 0.4, "compliance": 0.2,
                                         "logistics": 0.2, "coordinator": 0.2})
    budgets = make_budgets(world, 0.8)
    alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    rng = random.Random(42)
    fw_het = (expected_coverage(world.test_tasks, alloc, het, AGENT_SET, rng) +
              expected_coverage(world.test_tasks, alloc, FailureSpec("s2"),
                                AGENT_SET, rng, s2_samples=32)) / 2
    rows.append({"exp": "E5", "seed": seed, "axis": "het_failure", "value": "0.4/0.2/0.2/0.2",
                 "fw": round(fw_het, 4)})
    # s2 loss probability sweep
    for p in [0.1, 0.3, 0.5]:
        rng = random.Random(42)
        fw_p = expected_coverage(world.test_tasks, alloc, FailureSpec("s2", unit_loss_prob=p),
                                 AGENT_SET, rng, s2_samples=32)
        rows.append({"exp": "E5", "seed": seed, "axis": "s2_loss_p", "value": p,
                     "fw": round(fw_p, 4)})
    return rows


def run_cost(seed: int) -> list:
    world = generate(seed)
    budgets = make_budgets(world, 0.8)
    rows = []
    t0 = time.perf_counter()
    units = deepcopy(world.units)
    t_extract = time.perf_counter() - t0
    t0 = time.perf_counter()
    register_req_kinds(world.dev_tasks)
    t_witness = time.perf_counter() - t0
    t0 = time.perf_counter()
    alloc = STRATEGIES["core_mas"](units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    t_alloc = time.perf_counter() - t0
    t0 = time.perf_counter()
    rng = random.Random(42)
    expected_coverage(world.test_tasks, alloc, FailureSpec("s1"), AGENT_SET, rng)
    expected_coverage(world.test_tasks, alloc, FailureSpec("s2"), AGENT_SET, rng,
                      s2_samples=32)
    t_fail = time.perf_counter() - t0
    t0 = time.perf_counter()
    # provenance verification: check rule/source fields on every stored unit
    for us in alloc.values():
        for u in us:
            assert u.source_id and (u.kind != "predicate" or u.rule_id)
    t_prov = time.perf_counter() - t0
    rows.append({"exp": "E10", "seed": seed, "stage": "unit_extraction",
                 "cpu_seconds": round(t_extract, 4)})
    rows.append({"exp": "E10", "seed": seed, "stage": "witness_build",
                 "cpu_seconds": round(t_witness, 4)})
    rows.append({"exp": "E10", "seed": seed, "stage": "allocation",
                 "cpu_seconds": round(t_alloc, 4)})
    rows.append({"exp": "E10", "seed": seed, "stage": "failure_sim",
                 "cpu_seconds": round(t_fail, 4)})
    rows.append({"exp": "E10", "seed": seed, "stage": "provenance_check",
                 "cpu_seconds": round(t_prov, 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default="experiment_data/runs/run_e1")
    args = ap.parse_args()
    out = Path(args.out_dir)

    combos = [(s, c) for s in SEEDS for c in [0.4, 0.6, 0.8, 1.0]]
    rows = []
    with multiprocessing.Pool(args.workers) as pool:
        for r in pool.starmap(run_ablations, combos):
            rows.extend(r)
        for r in pool.map(run_sensitivity, SEEDS):
            rows.extend(r)
        for r in pool.map(run_cost, SEEDS):
            rows.extend(r)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e1_extra.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(out / "e1_extra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "seed", "capacity", "variant",
                                          "axis", "value", "fw", "seconds",
                                          "stage", "cpu_seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows to {out}/e1_extra.csv")


if __name__ == "__main__":
    main()
