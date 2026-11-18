"""Memory units: raw facts, verified predicates, and provenance.

A unit is the atomic object an allocation places on an agent. Costs are real
token counts (tiktoken cl100k_base as deterministic proxy; the E-mem pipeline's
Qwen3 tokenizer is used for end-to-end accounting in E2+, recorded per run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken

_ENC = None


def tokenizer():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # fallback: character count / 4 (rough, deterministic)
            _ENC = None
    return _ENC


def token_count(text: str) -> int:
    enc = tokenizer()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


@dataclass
class MemoryUnit:
    """One storable memory unit (a fact variant)."""

    unit_id: str
    fact_id: str  # which underlying fact this unit describes
    kind: str  # "raw" | "predicate"
    content: str
    source_id: str  # provenance
    rule_id: Optional[str]  # e.g. "budget_v1" for verified predicates
    supports: List[str]  # task-type ids this unit is sufficient evidence for
    not_sufficient_for: List[str]  # explicit negative capability (anti-overclaim)
    access_scope: List[str]  # agent ids allowed to hold this unit
    token_cost: int = 0
    covers_facts: Optional[List[str]] = None  # deduped units covering several facts

    def __post_init__(self):
        if not self.token_cost:
            # cost includes content + provenance/index overhead, per plan
            self.token_cost = token_count(self.content) + token_count(
                self.source_id + (self.rule_id or "") + "".join(self.access_scope)) + 2

    @property
    def covered_facts(self) -> List[str]:
        return self.covers_facts if self.covers_facts else [self.fact_id]

    def supports_requirement(self, req_fact_id: str, req_kind: str) -> bool:
        """Can this unit serve as evidence for a witness requirement?

        A raw unit satisfies any requirement on its fact. A predicate unit
        satisfies a requirement only if the requirement's kind is a task type
        in `supports` and NOT in `not_sufficient_for`.
        """
        if req_fact_id not in self.covered_facts:
            return False
        if self.kind == "raw":
            return True
        return req_kind in self.supports and req_kind not in self.not_sufficient_for


@dataclass
class AgentMemory:
    agent_id: str
    units: Dict[str, MemoryUnit] = field(default_factory=dict)
    token_budget: int = 0

    @property
    def used(self) -> int:
        return sum(u.token_cost for u in self.units.values())
