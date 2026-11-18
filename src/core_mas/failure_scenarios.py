"""Failure scenarios and post-failure accessible-memory computation.

s0: no failure.
s1: exactly one agent loses its memory (uniform over agents).
s2: each memory unit lost independently with probability p (default 0.3).
Heterogeneous per-agent failure probabilities supported for E5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .memory_units import AgentMemory


@dataclass
class FailureSpec:
    kind: str  # "s0" | "s1" | "s2"
    agent_probs: Optional[Dict[str, float]] = None  # heterogeneous, for s1/s2
    unit_loss_prob: float = 0.3  # for s2


def sample_failure(spec: FailureSpec, agents: List[str], rng: random.Random) -> Set[str]:
    """Return the set of memory unit ids that are LOST under one sampled failure."""
    if spec.kind == "s0":
        return set()
    if spec.kind == "s1":
        probs = spec.agent_probs or {a: 1.0 / len(agents) for a in agents}
        names = list(probs.keys())
        weights = [probs[n] for n in names]
        victim = rng.choices(names, weights=weights, k=1)[0]
        return {f"{victim}::ALL"}  # sentinel: whole-agent loss handled by caller
    if spec.kind == "s2":
        return {"unit"}
    raise ValueError(spec.kind)


def accessible_units(
    agents: Dict[str, AgentMemory], spec: FailureSpec, rng: random.Random
) -> List:
    """Units reachable after the sampled failure.

    s1 uses the '::ALL' sentinel: the victim agent's units are all removed.
    s2 drops each unit with unit_loss_prob.
    """
    if spec.kind == "s0":
        lost: Set[str] = set()
        victims = set()
    elif spec.kind == "s1":
        lost = set()
        probs = spec.agent_probs or {a: 1.0 / len(agents) for a in agents}
        names = list(probs.keys())
        weights = [probs[n] for n in names]
        victims = {rng.choices(names, weights=weights, k=1)[0]}
    elif spec.kind == "s2":
        victims = set()
        lost = set()
    else:
        raise ValueError(spec.kind)

    out = []
    for a_id, mem in agents.items():
        if a_id in victims:
            continue
        for uid, unit in mem.units.items():
            if spec.kind == "s2" and rng.random() < spec.unit_loss_prob:
                continue
            out.append(unit)
    return out
