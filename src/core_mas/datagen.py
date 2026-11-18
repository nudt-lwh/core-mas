"""Programmatic synthetic procurement task generator (seeded, reproducible).

3 agents: finance (quotes), compliance (certificates), logistics (delivery).
8 suppliers; facts = price / compliance / delivery per supplier (24 facts).
Memory unit variants per fact: raw + verified predicates (budget/delivery
thresholds). Tasks carry witness bundles (AND within a bundle, OR across
bundles); dev/test split by generator seed. Test tasks are frozen — allocators
may only use dev tasks (Oracle uses test witnesses as an upper bound only).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .memory_units import MemoryUnit
from .witness_graph import Bundle, Task

AGENTS = ["finance", "compliance", "logistics"]
SUPPLIERS = [f"S{i}" for i in range(1, 9)]
BUDGET_THRESHOLDS = [80, 100, 120]  # 万
DELIVERY_THRESHOLDS = [15, 30]  # days


def _agent_team(n_agents: int) -> List[str]:
    """E1_scale: department sub-teams. n_agents=3 -> exact original names;
    6/9 -> each department split into 2/3 sub-teams (finance_a, ...)."""
    if n_agents == 3:
        return list(AGENTS)
    if n_agents in (6, 9):
        k = n_agents // 3
        names = []
        for dept in AGENTS:
            for i in range(k):
                names.append(f"{dept}_{chr(ord('a') + i)}")
        return names
    raise ValueError(f"unsupported n_agents={n_agents} (use 3/6/9)")


@dataclass
class World:
    """One generated dataset instance."""

    seed: int
    suppliers: Dict[str, dict]  # sid -> {price, compliant, delivery_days}
    facts: Dict[str, dict]  # fact_id -> {supplier, attribute, value, origin_agent}
    units: List[MemoryUnit]  # all candidate units (raw + predicates)
    dev_tasks: List[Task] = field(default_factory=list)
    test_tasks: List[Task] = field(default_factory=list)
    # cost of the deduplicated raw set (capacity baseline 100%)
    baseline_cost: int = 0

    def facts_by_supplier(self, sid: str) -> List[str]:
        return [f for f, v in self.facts.items() if v["supplier"] == sid]


def _price_predicates(world: World, fact: str, sid: str, price: int,
                      owner: str) -> None:
    for b in BUDGET_THRESHOLDS:
        holds = price < b
        unit = MemoryUnit(
            unit_id=f"{fact}_under_{b}",
            fact_id=fact,
            kind="predicate",
            content=f"{sid} price < {b} (10k RMB), verified",
            source_id=f"quote-{sid}-07",
            rule_id="budget_v1",
            supports=["budget_check"] if holds else [],
            not_sufficient_for=["exact_price", "cheapest_pair"] if holds else [],
            access_scope=[owner, "coordinator"],
        )
        if holds:  # only store predicates that actually hold (verified)
            world.units.append(unit)


def _delivery_predicates(world: World, fact: str, sid: str, days: int,
                         owner: str) -> None:
    for d in DELIVERY_THRESHOLDS:
        holds = days <= d
        unit = MemoryUnit(
            unit_id=f"{fact}_by_{d}",
            fact_id=fact,
            kind="predicate",
            content=f"{sid} delivers within {d} days, verified",
            source_id=f"log-{sid}-03",
            rule_id="delivery_v1",
            supports=["deliver_by"] if holds else [],
            not_sufficient_for=["exact_delivery"] if holds else [],
            access_scope=[owner, "coordinator"],
        )
        if holds:
            world.units.append(unit)


def generate(seed: int, n_agents: int = 3, n_suppliers: int = 8) -> World:
    """Generate one dataset instance.

    n_agents=3 / n_suppliers=8 reproduces the main-experiment configuration
    exactly (same names, same rng sequence). For E1_scale, suppliers are
    distributed round-robin across the department sub-teams of their
    attribute's department (finance_* / compliance_* / logistics_*).
    """
    rng = random.Random(seed)
    world = World(seed=seed, suppliers={}, facts={}, units=[])

    suppliers = [f"S{i}" for i in range(1, n_suppliers + 1)]
    team = _agent_team(n_agents)
    k = n_agents // 3  # sub-teams per department

    for sid in suppliers:
        price = rng.randint(60, 150)
        compliant = rng.random() < 0.6
        delivery = rng.randint(7, 45)
        world.suppliers[sid] = {"price": price, "compliant": compliant, "delivery_days": delivery}

    for i, sid in enumerate(suppliers):
        for attr, dept in (("price", "finance"), ("compliant", "compliance"),
                           ("delivery_days", "logistics")):
            owner = dept if k == 1 else f"{dept}_{chr(ord('a') + i % k)}"
            vals = world.suppliers[sid]
            fact_id = f"f_{attr}_{sid}"
            world.facts[fact_id] = {
                "supplier": sid, "attribute": attr,
                "value": vals[attr], "origin_agent": owner,
            }
            raw_content = {
                "price": f"{sid} price = {vals['price']} (10k RMB)",
                "compliant": f"{sid} compliant = {vals['compliant']}, cert {sid}-09 v3",
                "delivery_days": f"{sid} delivery = {vals['delivery_days']} days",
            }[attr]
            world.units.append(MemoryUnit(
                unit_id=f"{fact_id}_raw",
                fact_id=fact_id,
                kind="raw",
                content=raw_content,
                source_id=f"src-{sid}-{owner}",
                rule_id=None,
                supports=[],  # raw satisfies everything (see witness_graph)
                not_sufficient_for=[],
                access_scope=[owner, "coordinator"],
            ))
            if attr == "price":
                _price_predicates(world, fact_id, sid, vals["price"], owner)
            elif attr == "delivery_days":
                _delivery_predicates(world, fact_id, sid, vals["delivery_days"], owner)

    world.baseline_cost = sum(
        u.token_cost for u in world.units if u.kind == "raw")

    # ---- dev tasks (fixed sub-seed 100; for allocation-time estimation) ----
    dev_rng = random.Random(100)
    world.dev_tasks = _make_tasks(world, dev_rng, 150, dev=True)
    # ---- test tasks (frozen per world seed; never used by deployable allocators) ----
    test_rng = random.Random(seed)
    world.test_tasks = _make_tasks(world, test_rng, 120, dev=False)
    return world


def _make_tasks(world: World, rng: random.Random, n: int, dev: bool) -> List[Task]:
    tasks = []
    for i in range(n):
        kind = rng.choice(["budget_check", "cheapest_pair", "cheapest_compliant",
                           "deliver_pair", "budget_and_delivery"])
        s1, s2 = rng.sample(list(world.suppliers.keys()), 2)
        f1p = f"f_price_{s1}"; f2p = f"f_price_{s2}"
        f1c = f"f_compliant_{s1}"; f2c = f"f_compliant_{s2}"
        f1d = f"f_delivery_days_{s1}"; f2d = f"f_delivery_days_{s2}"
        b = rng.choice(BUDGET_THRESHOLDS)
        d = rng.choice(DELIVERY_THRESHOLDS)

        if kind == "budget_check":
            bundles = [
                frozenset({(f1p, "exact_price")}),
                frozenset({(f1p, "budget_check")}),
            ]
        elif kind == "cheapest_pair":
            bundles = [frozenset({(f1p, "exact_price"), (f2p, "exact_price")})]
        elif kind == "cheapest_compliant":
            bundles = [frozenset({(f1p, "exact_price"), (f1c, "exact_comp"),
                                  (f2p, "exact_price"), (f2c, "exact_comp")})]
        elif kind == "deliver_pair":
            bundles = [
                frozenset({(f1d, "exact_delivery"), (f2d, "exact_delivery")}),
                frozenset({(f1d, "deliver_by"), (f2d, "deliver_by")}),
            ]
        else:  # budget_and_delivery: 4-fact AND with an alternative path
            bundles = [
                frozenset({(f1p, "exact_price"), (f1d, "exact_delivery"),
                           (f2p, "exact_price"), (f2d, "exact_delivery")}),
                frozenset({(f1p, "budget_check"), (f1d, "deliver_by"),
                           (f2p, "budget_check"), (f2d, "deliver_by")}),
            ]
        question, correct = _question_answer(
            world, kind, s1, s2,
            b if kind in ("budget_check", "budget_and_delivery") else None,
            d if kind in ("deliver_pair", "budget_and_delivery") else None)
        tasks.append(Task(task_id=f"{'dev' if dev else 'test'}_{i}_{kind}_{s1}_{s2}",
                          query_type=kind, witness_bundles=bundles, is_dev=dev,
                          question=question, correct_answer=correct))
    return tasks


def _question_answer(world, kind, s1, s2, budget=None, delivery=None):
    """Question text + programmatic correct answer for an answerable task."""
    if kind == "budget_check":
        price = world.suppliers[s1]["price"]
        q = f"Is {s1}'s price below {budget}0,000 RMB?"
        return q, ("yes" if price < budget else "no")
    if kind == "cheapest_pair":
        p1, p2 = world.suppliers[s1]["price"], world.suppliers[s2]["price"]
        q = f"Which supplier has the lower price: {s1} or {s2}?"
        return q, (s1 if p1 < p2 else s2)
    if kind == "cheapest_compliant":
        c1 = world.suppliers[s1]["compliant"]
        c2 = world.suppliers[s2]["compliant"]
        p1, p2 = world.suppliers[s1]["price"], world.suppliers[s2]["price"]
        q = f"Among compliant suppliers, which is cheaper: {s1} or {s2}?"
        if c1 and c2:
            return q, (s1 if p1 <= p2 else s2)
        if c1:
            return q, s1
        if c2:
            return q, s2
        return q, "none"
    if kind == "deliver_pair":
        d1 = world.suppliers[s1]["delivery_days"]
        d2 = world.suppliers[s2]["delivery_days"]
        q = f"Do both {s1} and {s2} deliver within {delivery} days?"
        return q, ("yes" if d1 <= delivery and d2 <= delivery else "no")
    # budget_and_delivery
    p1, p2 = world.suppliers[s1]["price"], world.suppliers[s2]["price"]
    d1, d2 = world.suppliers[s1]["delivery_days"], world.suppliers[s2]["delivery_days"]
    q = (f"Which of {s1} and {s2} is under budget {budget}0,000 RMB "
         f"and delivers within {delivery} days?")
    ok1 = p1 < budget and d1 <= delivery
    ok2 = p2 < budget and d2 <= delivery
    if ok1 and not ok2:
        return q, s1
    if ok2 and not ok1:
        return q, s2
    return q, "none"
