"""Allocation strategies for E1 structural experiment.

All strategies receive the same candidate units, access scopes, agent budgets
and total budget. Deployable strategies may use DEV tasks only; the Oracle uses
TEST tasks and is reported as a non-deployable upper bound.

Strategy ids:
  no_memory, uniform_sharding (E-mem-like origin storage), independent_summary,
  equal_replication, frequency_replication, semantic_dedup,
  task_aware_item, central_replication, traditional_placement,
  core_mas, oracle_witness
"""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .failure_scenarios import FailureSpec, accessible_units
from .memory_units import AgentMemory, MemoryUnit
from .witness_graph import Task, register_req_kinds, requirements_satisfied


@dataclass
class Budgets:
    per_agent: Dict[str, int]
    total: int

    def room(self, alloc: Dict[str, List[MemoryUnit]], agent: str) -> int:
        used = sum(u.token_cost for u in alloc.get(agent, []))
        return self.per_agent.get(agent, 0) - used

    def total_used(self, alloc: Dict[str, List[MemoryUnit]]) -> int:
        return sum(u.token_cost for us in alloc.values() for u in us)


def legal_actions(units: List[MemoryUnit], agents: List[str]) -> List[Tuple[MemoryUnit, str]]:
    out = []
    for u in units:
        for a in agents:
            if a in u.access_scope:
                out.append((u, a))
    return out


# --------------------------------------------------------------------------
# coverage evaluation (shared)
# --------------------------------------------------------------------------

def expected_coverage(
    tasks: List[Task], alloc: Dict[str, List[MemoryUnit]],
    spec: FailureSpec, agents: List[str], rng: random.Random,
    s1_exact: bool = True, s2_samples: int = 16,
) -> float:
    """E_failure[ P(task has >=1 fully satisfied witness bundle) ] over tasks."""
    register_req_kinds(tasks)
    mems = {a: AgentMemory(a, {u.unit_id: u for u in alloc.get(a, [])}) for a in agents}
    total = 0.0
    for task in tasks:
        if spec.kind == "s0":
            units = accessible_units(mems, spec, rng)
            if task.covered_by(requirements_satisfied(units, None)):
                total += 1.0
        elif spec.kind == "s1" and s1_exact:
            # exact: average over each agent as victim; heterogeneous failure
            # probabilities weight the victims when provided (E5 het_failure).
            # For uniform probs this reduces to hits / len(agents), so the main
            # E1/E2 results are unchanged.
            probs = spec.agent_probs or {a: 1.0 / len(agents) for a in agents}
            hits = 0.0
            for victim in agents:
                mems_v = {a: mems[a] for a in agents if a != victim}
                units = []
                for m in mems_v.values():
                    units.extend(m.units.values())
                if task.covered_by(requirements_satisfied(units, None)):
                    hits += probs.get(victim, 0.0)
            total += hits / sum(probs.get(a, 0.0) for a in agents)
        elif spec.kind == "s1x2":
            # E1_scale extension: exactly two agents lose memory simultaneously,
            # exact average over all unordered victim pairs
            hits = 0
            pairs = 0
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    victims = {agents[i], agents[j]}
                    mems_v = {a: mems[a] for a in agents if a not in victims}
                    units = []
                    for m in mems_v.values():
                        units.extend(m.units.values())
                    if task.covered_by(requirements_satisfied(units, None)):
                        hits += 1
                    pairs += 1
            total += hits / pairs
        else:  # s2 Monte Carlo
            hits = 0
            for _ in range(s2_samples):
                units = accessible_units(mems, spec, rng)
                if task.covered_by(requirements_satisfied(units, None)):
                    hits += 1
            total += hits / s2_samples
    return total / len(tasks)


def fact_frequency(tasks: List[Task]) -> Dict[str, int]:
    freq = defaultdict(int)
    for t in tasks:
        for b in t.witness_bundles:
            for fact_id, _ in b:
                freq[fact_id] += 1
    return freq


# --------------------------------------------------------------------------
# baseline allocators
# --------------------------------------------------------------------------

def _place(alloc, unit, agent, budgets):
    if budgets.room(alloc, agent) >= unit.token_cost and \
       budgets.total_used(alloc) + unit.token_cost <= budgets.total:
        alloc.setdefault(agent, []).append(unit)
        return True
    return False


def allocate_no_memory(units, agents, budgets, dev_tasks, world, **kw):
    return {}


def allocate_uniform_sharding(units, agents, budgets, dev_tasks, world, **kw):
    alloc = {}
    for u in units:
        if u.kind != "raw":
            continue
        _place(alloc, u, world.facts[u.fact_id]["origin_agent"], budgets)
    return alloc


def allocate_independent_summary(units, agents, budgets, dev_tasks, world, **kw):
    """Per-agent summary: cheapest predicate per fact (fallback raw), on origin."""
    alloc = {}
    best = {}
    for u in units:
        if u.fact_id not in best or u.token_cost < best[u.fact_id].token_cost:
            if u.kind == "predicate" or (u.fact_id not in best and u.kind == "raw"):
                best[u.fact_id] = u
    for fid, u in best.items():
        _place(alloc, u, world.facts[fid]["origin_agent"], budgets)
    return alloc


def allocate_equal_replication(units, agents, budgets, dev_tasks, world, **kw):
    alloc = {}
    raws = [u for u in units if u.kind == "raw"]
    while True:
        placed_any = False
        for u in raws:
            for a in u.access_scope:
                if _place(alloc, u, a, budgets):
                    placed_any = True
        if not placed_any:
            break
    return alloc


def allocate_frequency_replication(units, agents, budgets, dev_tasks, world, **kw):
    alloc = {}
    freq = fact_frequency(dev_tasks)
    raws = sorted([u for u in units if u.kind == "raw"],
                  key=lambda u: -freq.get(u.fact_id, 0))
    for u in raws:
        for a in u.access_scope:
            _place(alloc, u, a, budgets)
    return alloc


def allocate_semantic_dedup(units, agents, budgets, dev_tasks, world, **kw):
    """Dedupe identical (attribute,value) facts into shared units, then shard."""
    alloc = {}
    groups = defaultdict(list)
    for u in units:
        if u.kind == "raw":
            v = world.facts[u.fact_id]
            groups[(v["attribute"], v["value"])].append(u)
    # shared unit: keep one raw per group, re-linked to all member facts
    # (deepcopy: the re-linked unit must not mutate the shared world state that
    #  later strategies in the same matrix reuse)
    for key, raws in groups.items():
        kept = deepcopy(min(raws, key=lambda u: u.token_cost))
        if len(raws) > 1:
            kept.covers_facts = [u.fact_id for u in raws]
        _place(alloc, kept, world.facts[kept.fact_id]["origin_agent"], budgets)
    # remaining budget: cheapest predicates
    preds = sorted([u for u in units if u.kind == "predicate"], key=lambda u: u.token_cost)
    for u in preds:
        _place(alloc, u, u.access_scope[0], budgets)
    return alloc


def allocate_task_aware_item(units, agents, budgets, dev_tasks, world, **kw):
    """Per-item task value (single-unit marginal coverage), no complementarity."""
    alloc = {}
    rng = random.Random(0)
    base_cov = expected_coverage(dev_tasks, {}, FailureSpec("s0"), agents, rng)
    actions = legal_actions(units, agents)
    while True:
        best = None
        best_score = 0.0
        for u, a in actions:
            if not _can_place(alloc, u, a, budgets):
                continue
            trial = {k: list(v) for k, v in alloc.items()}
            trial.setdefault(a, []).append(u)
            cov = expected_coverage(dev_tasks, trial, FailureSpec("s0"), agents, rng)
            score = (cov - base_cov) / u.token_cost
            if score > best_score:
                best, best_score = (u, a), score
        if best is None or best_score <= 0:
            break
        u, a = best
        _place(alloc, u, a, budgets)
    return alloc


def allocate_central_replication(units, agents, budgets, dev_tasks, world, **kw):
    """Central store (coordinator) first, then frequency-based replicas."""
    alloc = {}
    freq = fact_frequency(dev_tasks)
    raws = sorted([u for u in units if u.kind == "raw"],
                  key=lambda u: -freq.get(u.fact_id, 0))
    for u in raws:
        _place(alloc, u, "coordinator", budgets)
    for u in raws:
        for a in u.access_scope:
            if a != "coordinator":
                _place(alloc, u, a, budgets)
    return alloc


def allocate_traditional_placement(units, agents, budgets, dev_tasks, world, **kw):
    """Webcache-style greedy: place by access frequency onto lowest-failure agent."""
    alloc = {}
    freq = fact_frequency(dev_tasks)
    raws = sorted([u for u in units if u.kind == "raw"],
                  key=lambda u: -freq.get(u.fact_id, 0))
    # order agents by assumed reliability (coordinator most reliable in s1-free design)
    order = ["coordinator"] + [a for a in agents if a != "coordinator"]
    for u in raws:
        for a in order:
            if a in u.access_scope and _place(alloc, u, a, budgets):
                break
    return alloc


# --------------------------------------------------------------------------
# CORE-MAS (CWRA): witness-aware, fault-weighted, bundle-aware greedy + local search
# --------------------------------------------------------------------------

def _can_place(alloc, unit, agent, budgets):
    return budgets.room(alloc, agent) >= unit.token_cost and \
        budgets.total_used(alloc) + unit.token_cost <= budgets.total


def _bundle_actions(world, dev_tasks, agents, max_facts=3):
    """Small witness bundles: for each dev-task bundle, complementary unit sets
    (2-max_facts facts, default 3; raw or cheapest variant per fact) that would
    complete it, as a joint action placed on any agent allowed to hold all of
    them."""
    variants_by_fact = {}
    for u in world.units:
        variants_by_fact.setdefault(u.fact_id, []).append(u)
    actions = []
    seen = set()
    for t in dev_tasks:
        for b in t.witness_bundles:
            if 2 <= len(b) <= max_facts:
                key = frozenset(f for f, _ in b)
                if key in seen:
                    continue
                seen.add(key)
                facts = sorted(key)
                # raw variant of every fact (guaranteed legal for some agent)
                raw_units = [min((u for u in variants_by_fact[f] if u.kind == "raw"),
                                 key=lambda u: u.token_cost) for f in facts]
                for a in agents:
                    if all(a in u.access_scope for u in raw_units):
                        actions.append((list(raw_units), a))
                        break
                # cheapest mixed variant (granularity choice) for each fact,
                # only if the chosen predicates actually hold (supports nonempty
                # beyond raw fallback)
                mixed = []
                for f in facts:
                    preds = [u for u in variants_by_fact[f] if u.kind == "predicate" and u.supports]
                    mixed.append(min(preds, key=lambda u: u.token_cost) if preds
                                 else raw_units[facts.index(f)])
                for a in agents:
                    if all(a in u.access_scope for u in mixed):
                        actions.append((list(mixed), a))
                        break
    return actions


def allocate_core_mas(units, agents, budgets, dev_tasks, world, **kw):
    """CWRA: witness-aware fault-weighted bundle greedy + local search.

    Ablation flags (E4): use_witness / use_failure / use_granularity /
    use_bundles / use_local_search.
    """
    use_witness = kw.get("use_witness", True)
    use_failure = kw.get("use_failure", True)
    use_granularity = kw.get("use_granularity", True)
    use_bundles = kw.get("use_bundles", True)
    use_local_search = kw.get("use_local_search", True)
    use_budget_fill = kw.get("use_budget_fill", False)  # diagnostic variant (E1_scale)

    units = [u for u in units if use_granularity or u.kind == "raw"]
    alloc: Dict[str, List[MemoryUnit]] = {}
    rng = random.Random(0)
    specs = [FailureSpec("s0"), FailureSpec("s1")] if use_failure else [FailureSpec("s0")]

    if use_witness:
        def obj(a):
            return sum(expected_coverage(dev_tasks, a, s, agents, rng) for s in specs) / len(specs)
    else:
        # single-fact importance: per-item additive coverage proxy (no witness
        # structure, no complementarity)
        register_req_kinds(dev_tasks)
        fact_reqs = {}
        for t in dev_tasks:
            for b in t.witness_bundles:
                for f, k in b:
                    fact_reqs.setdefault(f, set()).add((f, k))

        def obj(a):
            satisfied = requirements_satisfied(
                [u for us in a.values() for u in us], None)
            total = 0.0
            for f, reqs in fact_reqs.items():
                total += len(reqs & satisfied) / max(1, len(reqs))
            return total / max(1, len(fact_reqs))

    def delta(a, action):
        before = obj(a)
        trial = {k: list(v) for k, v in a.items()}
        for u in action[0]:
            trial.setdefault(action[1], []).append(u)
        return obj(trial) - before

    single_actions = legal_actions(units, agents)
    bundle_actions = (_bundle_actions(world, dev_tasks, agents,
                                      max_facts=kw.get("bundle_max_facts", 3))
                      if use_bundles else [])

    # bundle-aware greedy: evaluate single units AND small bundles
    while True:
        best = None
        best_score = 0.0
        for action in [( [u], a) for u, a in single_actions] + \
                      [([u for u in us], a) for us, a in bundle_actions]:
            placed_ids = {u.unit_id for us in alloc.values() for u in us}
            fresh = [u for u in action[0] if u.unit_id not in placed_ids]
            if not fresh:
                continue
            if not all(_can_place(alloc, u, action[1], budgets) for u in fresh):
                continue
            gain = delta(alloc, (fresh, action[1]))
            cost = sum(u.token_cost for u in fresh)
            score = gain / max(cost, 1)
            if score > best_score:
                best, best_score = (fresh, action[1]), score
        if best is None or best_score <= 0:
            break
        for u in best[0]:
            _place(alloc, u, best[1], budgets)

    if not use_local_search:
        if use_budget_fill:
            _fill_budget(alloc, units, agents, budgets, dev_tasks, world)
        return alloc

    # local search: (a) swap two placed units, (b) swap a placed unit with an
    # unplaced one (budget- and scope-legal, objective-improving moves only)
    def try_improve():
        nonlocal alloc
        placed = [(a, u) for a, us in alloc.items() for u in us]
        unplaced = [u for u in units if all(
            u.unit_id not in {x.unit_id for x in us} for us in alloc.values())]
        base = obj(alloc)
        # (a) placed-placed swap
        for i, (a_i, u_i) in enumerate(placed):
            for j in range(i + 1, len(placed)):
                a_j, u_j = placed[j]
                if not (a_j in u_i.access_scope and a_i in u_j.access_scope):
                    continue
                trial = {k: list(v) for k, v in alloc.items()}
                trial[a_i].remove(u_i); trial[a_j].remove(u_j)
                trial[a_i].append(u_j); trial[a_j].append(u_i)
                if obj(trial) > base + 1e-9:
                    return trial
        # (b) placed-unplaced swap
        for (a_i, u_i) in placed:
            for u_j in unplaced:
                if a_i not in u_j.access_scope:
                    continue
                if budgets.room(alloc, a_i) + u_i.token_cost < u_j.token_cost:
                    continue  # would exceed per-agent budget
                trial = {k: list(v) for k, v in alloc.items()}
                trial[a_i].remove(u_i)
                trial[a_i].append(u_j)
                if budgets.total_used(trial) <= budgets.total and \
                   obj(trial) > base + 1e-9:
                    return trial
        return None

    while True:
        better = try_improve()
        if better is None:
            break
        alloc = better
    if use_budget_fill:
        _fill_budget(alloc, units, agents, budgets, dev_tasks, world)
    return alloc


def _fill_budget(alloc, units, agents, budgets, dev_tasks, world):
    """Diagnostic variant (E1_scale): once the witness-greedy exhausts
    positive-gain actions, fill the remaining budget with development-frequency-
    ranked raw units on their origin agents (respecting access scopes). This
    isolates the early-stopping failure mode at M=128."""
    placed_ids = {u.unit_id for us in alloc.values() for u in us}
    freq = fact_frequency(dev_tasks)
    remaining = sorted(
        [u for u in units if u.kind == "raw" and u.unit_id not in placed_ids],
        key=lambda u: -freq.get(u.fact_id, 0))
    for u in remaining:
        if _place(alloc, u, world.facts[u.fact_id]["origin_agent"], budgets):
            pass


def allocate_oracle_witness(units, agents, budgets, dev_tasks, world, **kw):
    """Same CWRA objective but estimated on TEST tasks (upper bound, NOT deployable)."""
    return allocate_core_mas(units, agents, budgets, world.test_tasks, world)


STRATEGIES = {
    "no_memory": allocate_no_memory,
    "uniform_sharding": allocate_uniform_sharding,
    "independent_summary": allocate_independent_summary,
    "equal_replication": allocate_equal_replication,
    "frequency_replication": allocate_frequency_replication,
    "semantic_dedup": allocate_semantic_dedup,
    "task_aware_item": allocate_task_aware_item,
    "central_replication": allocate_central_replication,
    "traditional_placement": allocate_traditional_placement,
    "core_mas": allocate_core_mas,
    "oracle_witness": allocate_oracle_witness,
}
