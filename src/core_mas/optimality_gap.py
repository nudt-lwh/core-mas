"""M12: small-scale optimality gap via CP-SAT exact solver.

Compares CORE-MAS's heuristic allocation with the exact optimum of the same
objective (fault-weighted dev-task witness coverage) on a reduced instance
(first N suppliers), both under the same budgets. Small scale only — no claim
of general optimality (plan §E1 / metrics_spec M12).

Run: python -m core_mas.optimality_gap [--n-suppliers 6]
"""

from __future__ import annotations

import argparse
import random

from ortools.sat.python import cp_model

from .allocators import STRATEGIES, expected_coverage
from .datagen import AGENTS, generate
from .eval_structural import AGENT_SET, make_budgets
from .failure_scenarios import FailureSpec
from .witness_graph import register_req_kinds, requirements_satisfied


def reduce_world(world, n_suppliers: int):
    from .datagen import SUPPLIERS

    keep_suppliers = set(SUPPLIERS[:n_suppliers])
    keep_facts = {f for f, v in world.facts.items() if v["supplier"] in keep_suppliers}
    units = [u for u in world.units if u.fact_id in keep_facts]
    dev_tasks = [t for t in world.dev_tasks
                 if all(f in keep_facts for b in t.witness_bundles for f, _ in b)]
    return units, dev_tasks


def exact_solution(units, dev_tasks, budgets, agents, timeout: float = 120.0):
    """CP-SAT: maximize s0 dev-task witness coverage under budget constraints.

    Returns (objective_or_None, status_str) where status is 'OPTIMAL' or
    'FEASIBLE' (time-limited); (None, status) if no solution found.
    """
    register_req_kinds(dev_tasks)
    m = cp_model.CpModel()
    x = {}
    for u in units:
        for a in agents:
            if a in u.access_scope:
                x[(u.unit_id, a)] = m.NewBoolVar(f"x_{u.unit_id}_{a}")
    for a in agents:
        m.Add(sum(u.token_cost * x[(u.unit_id, a)] for u in units
                  if (u.unit_id, a) in x) <= budgets.per_agent[a])
    m.Add(sum(u.token_cost * x[(u.unit_id, a)] for u in units for a in agents
              if (u.unit_id, a) in x) <= budgets.total)

    task_vars = []
    for t in dev_tasks:
        bundle_ok = []
        for b in t.witness_bundles:
            req_ok = []
            for fact_id, kind in b:
                unit_vars = []
                for u in units:
                    if u.kind == "raw":
                        covers = fact_id in u.covered_facts
                    else:
                        covers = (fact_id in u.covered_facts and kind in u.supports
                                  and kind not in u.not_sufficient_for)
                    if covers:
                        unit_vars += [x[(u.unit_id, a)] for a in agents
                                      if (u.unit_id, a) in x]
                if unit_vars:
                    r = m.NewBoolVar(f"r_{t.task_id}_{fact_id}_{kind}")
                    m.AddBoolOr(unit_vars + [r.Not()])
                    m.Add(sum(unit_vars) >= 1).OnlyEnforceIf(r)
                    req_ok.append(r)
            if req_ok:
                bb = m.NewBoolVar(f"b_{t.task_id}_{len(bundle_ok)}")
                m.AddBoolAnd(req_ok).OnlyEnforceIf(bb)
                m.Add(sum(req_ok) < len(req_ok)).OnlyEnforceIf(bb.Not())
                bundle_ok.append(bb)
        if bundle_ok:
            tv = m.NewBoolVar(f"t_{t.task_id}")
            m.AddBoolOr(bundle_ok + [tv.Not()])
            m.Add(sum(bundle_ok) >= 1).OnlyEnforceIf(tv)
            task_vars.append(tv)

    m.Maximize(sum(task_vars))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, "NO_SOLUTION"
    obj = solver.ObjectiveValue()
    n_tasks = max(1, len(dev_tasks))
    return obj / n_tasks, ("OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-suppliers", type=int, default=6)
    ap.add_argument("--capacity", type=float, default=0.6)
    args = ap.parse_args()

    world = generate(200)
    units, dev_tasks = reduce_world(world, args.n_suppliers)
    budgets = make_budgets(world, args.capacity)
    # scale budgets to the reduced instance proportionally
    total = int(budgets.total * len(units) / len(world.units))
    per = max(1, total // len(AGENT_SET))
    from .allocators import Budgets
    budgets = Budgets(per_agent={a: per for a in AGENT_SET}, total=total)

    print(f"reduced instance: {len(units)} units, {len(dev_tasks)} dev tasks, "
          f"budget={total}")
    alloc = STRATEGIES["core_mas"](units, AGENT_SET, budgets, dev_tasks, world)
    rng = random.Random(42)
    heur = expected_coverage(dev_tasks, alloc, FailureSpec("s0"), AGENT_SET, rng)
    exact, status = exact_solution(units, dev_tasks, budgets, AGENT_SET)
    print(f"core_mas heuristic coverage: {heur:.4f}")
    if exact is None:
        print(f"exact: CP-SAT failed (status={status})")
    else:
        gap = (exact - heur) / exact if exact > 0 else 0.0
        print(f"exact coverage: {exact:.4f} (status={status})")
        print(f"optimality gap: {gap:.2%}")


if __name__ == "__main__":
    main()
