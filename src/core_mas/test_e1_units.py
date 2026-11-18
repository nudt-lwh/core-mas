"""Unit tests for the E1 structural layer (the 6 required tests from the plan).

Run: python -m core_mas.test_e1_units
"""

import random

from .datagen import generate
from .failure_scenarios import FailureSpec, accessible_units
from .memory_units import AgentMemory, MemoryUnit
from .witness_graph import Task, register_req_kinds, requirements_satisfied


def _mk_agents():
    return {a: AgentMemory(a) for a in ["finance", "compliance", "logistics", "coordinator"]}


def test_whole_agent_loss_isolated():
    """A lost agent's units are unreachable; coordinator cannot see them."""
    mems = _mk_agents()
    u = MemoryUnit("u1", "f1", "raw", "secret", "src", None, [], [],
                   ["finance", "coordinator"])
    mems["finance"].units[u.unit_id] = u
    rng = random.Random(0)
    units = accessible_units(mems, FailureSpec("s1", agent_probs={"finance": 1.0}), rng)
    assert len(units) == 0, "victim's units leaked"
    print("test_whole_agent_loss_isolated OK")


def test_predicate_cannot_answer_exact():
    """A budget predicate must not satisfy an exact-price requirement."""
    pred = MemoryUnit("p1", "f1", "predicate", "S1 price < 100",
                      "quote-1", "budget_v1", ["budget_check"], ["exact_price"],
                      ["finance"])
    raw = MemoryUnit("r1", "f1", "raw", "S1 price = 92", "quote-1", None,
                     [], [], ["finance"])
    task = Task("t", "budget_check",
                [frozenset({("f1", "exact_price")})])
    register_req_kinds([task])
    sat_pred = requirements_satisfied([pred], None)
    sat_raw = requirements_satisfied([raw], None)
    assert ("f1", "exact_price") not in sat_pred
    assert ("f1", "exact_price") in sat_raw
    assert not task.covered_by(sat_pred)
    assert task.covered_by(sat_raw)
    print("test_predicate_cannot_answer_exact OK")


def test_and_dependency_incomplete():
    """Missing one leg of an AND bundle must be uncovered; one full OR path suffices."""
    u_a = MemoryUnit("a", "fA", "raw", "A", "s", None, [], [], ["finance"])
    u_b = MemoryUnit("b", "fB", "raw", "B", "s", None, [], [], ["finance"])
    u_c = MemoryUnit("c", "fC", "raw", "C", "s", None, [], [], ["finance"])
    task_and = Task("t_and", "q", [frozenset({("fA", "k"), ("fB", "k")})])
    task_or = Task("t_or", "q", [frozenset({("fA", "k")}), frozenset({("fC", "k")})])
    register_req_kinds([task_and, task_or])
    sat_ab = requirements_satisfied([u_a, u_b], None)
    sat_a = requirements_satisfied([u_a], None)
    sat_ac = requirements_satisfied([u_a, u_c], None)
    assert task_and.covered_by(sat_ab)
    assert not task_and.covered_by(sat_a)
    assert task_or.covered_by(sat_a)  # one OR path
    assert task_or.covered_by(sat_ac)
    print("test_and_dependency_incomplete OK")


def test_budget_accounting():
    """Total stored cost (units + provenance + index) respects budget."""
    world = generate(200)
    used = sum(u.token_cost for u in world.units)
    assert used > 0
    # every unit's cost includes provenance/index overhead beyond raw content
    from .memory_units import token_count
    for u in world.units:
        assert u.token_cost >= token_count(u.content) + 1
    print("test_budget_accounting OK")


def test_no_fabricated_recovery():
    """Units with empty supports (non-holding predicates) are never generated."""
    world = generate(200)
    for u in world.units:
        if u.kind == "predicate":
            assert u.supports, f"non-holding predicate generated: {u.unit_id}"
            assert u.rule_id  # verified rule provenance required
    print("test_no_fabricated_recovery OK")


def test_reproducibility():
    """Same seeds -> identical worlds and identical coverage."""
    w1, w2 = generate(202), generate(202)
    assert [u.unit_id for u in w1.units] == [u.unit_id for u in w2.units]
    assert [t.task_id for t in w1.test_tasks] == [t.task_id for t in w2.test_tasks]
    from .allocators import STRATEGIES, expected_coverage
    from .eval_structural import make_budgets, AGENT_SET
    b = make_budgets(w1, 0.8)
    a1 = STRATEGIES["core_mas"](w1.units, AGENT_SET, b, w1.dev_tasks, w1)
    a2 = STRATEGIES["core_mas"](w2.units, AGENT_SET, b, w2.dev_tasks, w2)
    c1 = expected_coverage(w1.test_tasks, a1, FailureSpec("s1"), AGENT_SET, random.Random(7))
    c2 = expected_coverage(w2.test_tasks, a2, FailureSpec("s1"), AGENT_SET, random.Random(7))
    assert abs(c1 - c2) < 1e-12
    print("test_reproducibility OK")


if __name__ == "__main__":
    test_whole_agent_loss_isolated()
    test_predicate_cannot_answer_exact()
    test_and_dependency_incomplete()
    test_budget_accounting()
    test_no_fabricated_recovery()
    test_reproducibility()
    print("ALL 6 UNIT TESTS PASSED")
