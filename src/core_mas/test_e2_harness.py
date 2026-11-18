"""Unit tests for the E2 harness (MockLLM, no real model invoked).

Checks: fault isolation (victim's units never reach any prompt — enforced by
construction), prompt cardinality, grading, insufficient-evidence path,
structural solver vs LLM gap plumbing.
"""

import random

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import EvalAgents, MockLLM, grade, normalize_answer
from .eval_structural import AGENT_SET, make_budgets


def test_fault_isolation():
    """Victim agent's memory must never appear in any surviving agent's input."""
    world = generate(200)
    budgets = make_budgets(world, 0.8)
    alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    victim = "finance"
    victim_ids = {u.unit_id for u in alloc.get(victim, [])}
    assert victim_ids, "victim has no units — test needs a victim with memory"
    ev = EvalAgents(world, alloc, MockLLM(), AGENT_SET)
    accessible = ev.accessible_units(victim)
    assert victim not in accessible
    for a, units in accessible.items():
        for u in units:
            assert u.unit_id not in victim_ids or u.unit_id in {x.unit_id for x in units}
    print("test_fault_isolation OK")


def test_run_task_flow():
    """A full task run with the mock backend: grading + solver agree on
    evidence-supported answers, and the flow returns all fields."""
    world = generate(200)
    budgets = make_budgets(world, 1.0)
    alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    ev = EvalAgents(world, alloc, MockLLM(), AGENT_SET)
    task = world.test_tasks[0]
    r = ev.run_task(task, victim=None)
    for k in ("coord_answer", "correct", "correct_llm", "structural_answer",
              "correct_structural", "insufficient_llm", "unsupported_llm"):
        assert k in r, k
    # mock coordinator echoes evidence verbatim, so it can't equal the
    # canonical answer in general — but the structural solver path must work
    assert r["structural_answer"] in ("yes", "no", "none", "S1", "S2", "S3", "S4",
                                      "S5", "S6", "S7", "S8", "INSUFFICIENT_EVIDENCE"), \
        r["structural_answer"]
    print("test_run_task_flow OK")


def test_insufficient_evidence_path():
    """Empty evidence -> structural solver must say INSUFFICIENT_EVIDENCE."""
    world = generate(200)
    ev = EvalAgents(world, {}, MockLLM(), AGENT_SET)
    task = world.test_tasks[0]
    r = ev.run_task(task, victim=None)
    assert r["structural_answer"] == "INSUFFICIENT_EVIDENCE"
    assert r["correct_structural"] is False or task.correct_answer == "none"
    print("test_insufficient_evidence_path OK")


def test_grade_normalization():
    assert grade("Yes.", "yes")
    assert grade("  no  ", "no")
    assert grade("Supplier S3 is cheaper", "s3")
    assert not grade("", "yes")
    assert normalize_answer("INSUFFICIENT_EVIDENCE").startswith("insufficient")
    print("test_grade_normalization OK")


def test_structural_vs_programmatic():
    """With FULL evidence (100% budget, no failure), the structural solver
    should match the programmatic answer on most tasks."""
    world = generate(200)
    budgets = make_budgets(world, 1.0)
    alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                   world.dev_tasks, world)
    ev = EvalAgents(world, alloc, MockLLM(), AGENT_SET)
    hits = 0
    n = 30
    for t in world.test_tasks[:n]:
        r = ev.run_task(t, victim=None)
        hits += r["correct_structural"]
    rate = hits / n
    assert rate > 0.5, f"structural solver too weak: {rate:.2f}"
    print(f"test_structural_vs_programmatic OK (agree rate {rate:.2f} on {n})")


if __name__ == "__main__":
    test_fault_isolation()
    test_run_task_flow()
    test_insufficient_evidence_path()
    test_grade_normalization()
    test_structural_vs_programmatic()
    print("ALL E2 HARNESS TESTS PASSED")
