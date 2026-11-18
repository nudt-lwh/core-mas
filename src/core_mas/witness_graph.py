"""Task witness structures: which evidence combinations suffice for a task.

A task q has a set of witness bundles W(q). Each bundle is a set of
requirements (fact_id, req_kind) that must ALL be satisfied (AND); any single
bundle suffices (OR). req_kind is a task-type id that raw facts always satisfy
and predicate units satisfy only when listed in their `supports`.

Oracle witnesses (test tasks) must never be used by deployable allocators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

Req = Tuple[str, str]  # (fact_id, req_kind)
Bundle = FrozenSet[Req]


@dataclass
class Task:
    task_id: str
    query_type: str  # e.g. "budget_check", "cheapest_pair", ...
    witness_bundles: List[Bundle]
    # metadata for allocation-time task distribution estimation
    is_dev: bool = True
    # answerable task fields (E2 LLM layer; optional)
    question: str = ""
    correct_answer: str = ""  # programmatic answer for grading

    def covered_by(self, satisfied: Set[Req]) -> bool:
        """True if at least one bundle is fully satisfied."""
        for b in self.witness_bundles:
            if b.issubset(satisfied):
                return True
        return False


def requirements_satisfied(units, fact_id_to_units) -> Set[Req]:
    """Compute which (fact_id, req_kind) requirements the given units satisfy.

    For each stored unit, all its `supports` task types are satisfied for its
    fact (raw units satisfy every req_kind used by any task — handled at the
    caller by trying each req_kind appearing in the tasks).
    """
    satisfied: Set[Req] = set()
    for u in units:
        for fid in u.covered_facts:
            if u.kind == "raw":
                # raw satisfies any requirement kind on its fact; kinds enumerated
                # by the caller via all_req_kinds
                for k in _KINDS_BY_FACT.get(fid, []):
                    satisfied.add((fid, k))
            else:
                for k in u.supports:
                    if k not in u.not_sufficient_for:
                        satisfied.add((fid, k))
    return satisfied


_KINDS_BY_FACT: Dict[str, List[str]] = {}


def register_req_kinds(tasks: List[Task]) -> None:
    """Precompute req kinds per fact so raw units can satisfy them all."""
    global _KINDS_BY_FACT
    _KINDS_BY_FACT = {}
    for t in tasks:
        for b in t.witness_bundles:
            for fact_id, kind in b:
                _KINDS_BY_FACT.setdefault(fact_id, [])
                if kind not in _KINDS_BY_FACT[fact_id]:
                    _KINDS_BY_FACT[fact_id].append(kind)
