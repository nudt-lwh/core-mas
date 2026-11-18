"""E2: multi-agent LLM evaluation with the ECCR protocol.

Protocol per plan (Evidence-Constrained Capability Recovery):
  1. failure injection: victim agent's memory is inaccessible (its units are
     never shown in any prompt — permission isolation enforced by construction)
  2. each surviving agent answers ONE evidence-request call restricted to its
     own memory units (with source ids and rule versions)
  3. coordinator merges the evidence and answers the question, or replies
     INSUFFICIENT_EVIDENCE
  4. deterministic grading against the programmatic correct answer; a
     structural rule solver over the SAME accessible evidence gives the
     evidence-supported answer (solver/LLM gap = M13; unsupported LLM answers
     = M6 proxy)

LLMBackend is swappable: MockLLM for local unit tests, HTTPOpenAILLM for the
real model (cloud llama server). No LLM is invoked in this module's tests.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .memory_units import MemoryUnit
from .witness_graph import Task, register_req_kinds, requirements_satisfied

AGENT_SYSTEM = ("You are a memory agent in a multi-agent team. You can see ONLY "
                "your own memory records. Return the records relevant to the "
                "question, verbatim, including their [id] and source. If nothing "
                "is relevant, reply exactly: NO_RELEVANT_EVIDENCE")

COORD_SYSTEM = ("You are the team coordinator. Answer the question using ONLY "
                "the evidence below. Do not use outside knowledge. If any "
                "evidence mentions the entities asked about, you MUST answer. "
                "Reply exactly in the form: ANSWER: <answer> where <answer> "
                "is one of: yes, no, none, or a supplier id like S1. Only if "
                "NO evidence is relevant at all, reply: "
                "ANSWER: INSUFFICIENT_EVIDENCE")

COORD_SYSTEM_V2 = ("You are the team coordinator. Answer the question using ONLY "
                   "the verified evidence below. Do not use outside knowledge "
                   "and do not infer beyond the evidence. Reply exactly in the "
                   "form: ANSWER: <answer> where <answer> is one of: yes, no, "
                   "none, or a supplier id like S1. If the evidence is not "
                   "sufficient to justify an answer, reply: "
                   "ANSWER: INSUFFICIENT_EVIDENCE")


def record_line(u) -> str:
    """Canonical record line an agent may return (verification anchor)."""
    return f"[{u.unit_id}|src:{u.source_id}|rule:{u.rule_id or '-'}] {u.content}"


def get_records(response: str, units) -> list:
    """Accept only exact canonical records the responding agent actually holds.
    Fabricated ids, rewritten content, or records from other agents fail."""
    out = []
    for u in units:
        if any(s.strip().removeprefix("- ").strip() == record_line(u)
               for s in (response or "").splitlines()):
            out.append(u)
    return out


def full_gate(task, units) -> bool:
    """Pure boolean sufficiency check: does any witness bundle hold on the
    verified records, with predicate thresholds actually supporting the
    question? Never reads world facts or the gold answer."""
    def support(u, f, k):
        if f not in u.covered_facts:
            return False
        if u.kind == "raw":
            return True
        if k not in u.supports or k in u.not_sufficient_for:
            return False
        if k == "budget_check":
            q = re.search(r"(?:below |under budget )(\d+)0,000", task.question)
            p = re.search(r"price < (\d+)", u.content)
            return bool(q and p and int(p[1]) <= int(q[1]))
        if k == "deliver_by":
            q = re.search(r"within (\d+) days", task.question)
            p = re.search(r"within (\d+) days", u.content)
            return bool(q and p and int(p[1]) <= int(q[1]))
        return True
    return any(all(any(support(u, f, k) for u in units) for f, k in b)
               for b in task.witness_bundles)


def targeted_repair(task, current, accessible):
    """One deterministic exact-key fetch round over surviving agents' stored
    units (entity x attribute from the question), raw first, until the gate
    passes. Recovers stored-but-missed evidence only."""
    records = list(current)
    ids = {u.unit_id for u in records}
    sids = set(re.findall(r"\bS\d+\b", task.question))
    fields = {"budget_check": ["price"], "cheapest_pair": ["price"],
              "cheapest_compliant": ["price", "compliant"],
              "deliver_pair": ["delivery_days"],
              "budget_and_delivery": ["price", "delivery_days"]}[task.query_type]
    needed = {f"f_{attr}_{s}" for s in sids for attr in fields}
    candidates = {u.unit_id: u for us in accessible.values() for u in us
                  if u.fact_id in needed and u.unit_id not in ids}
    for u in sorted(candidates.values(),
                    key=lambda u: (u.kind != "raw", u.token_cost, u.unit_id)):
        if full_gate(task, records):
            break
        records.append(u)
    return records, full_gate(task, records)


def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)  # keep alnum only
    # map common phrasings to canonical tokens; note "none" must stay
    # distinct from "no" (exact token match, no prefix normalization)
    s = re.sub(r"^(yes).*", "yes", s)
    return s[:40]


def extract_answer(prediction: str) -> str:
    """Parse the ANSWER: marker first, fall back to raw heuristics."""
    raw = (prediction or "").strip()
    m = re.search(r"ANSWER\s*[:：]\s*([^\n,;.]{1,30})", raw, re.I)
    if m:
        return m.group(1).strip().lower()
    return raw.lower()


def grade(prediction: str, correct: str) -> bool:
    ans = extract_answer(prediction)
    p, c = normalize_answer(ans), normalize_answer(correct)
    if not p:
        return False
    if p == c:
        return True
    if c == "insufficientevidence":
        return p.startswith("insufficient")
    # supplier answers: exact supplier-id match only (no prefix / subset
    # semantics — "S1" must not match "S10" or "S10" match "S1")
    suppliers_p = set(re.findall(r"s\d+", ans))
    suppliers_c = set(re.findall(r"s\d+", correct.lower()))
    if suppliers_c and suppliers_p == suppliers_c:
        return True
    return False


class LLMBackend:
    def chat(self, system: str, user: str) -> str:
        raise NotImplementedError


class MockLLM(LLMBackend):
    """Deterministic mock: agents echo all their records; coordinator echoes
    the evidence back verbatim (flow tests only — not a solver)."""

    def chat(self, system: str, user: str) -> str:
        if "memory agent" in system:
            mem_block = user.split("Your memory:\n", 1)[-1].split("\n\nQuestion")[0]
            return mem_block if mem_block.strip() else "NO_RELEVANT_EVIDENCE"
        ev = user.split("Evidence:\n", 1)[-1]
        return ev.strip()


class HTTPOpenAILLM(LLMBackend):
    """Real model via an OpenAI-compatible endpoint (cloud llama server).

    Env overrides (for API models like DeepSeek):
      LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_MAX_TOKENS
    """

    def __init__(self, base_url: str = "http://127.0.0.1:6006/v1",
                 model: str = "qwen3-4b-local", max_tokens: int = 300):
        import os

        self.base_url = os.environ.get("LLM_BASE_URL", base_url)
        self.model = os.environ.get("LLM_MODEL", model)
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", str(max_tokens)))
        self._api_key = os.environ.get("LLM_API_KEY")

    def chat(self, system: str, user: str) -> str:
        import json

        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": 0, "max_tokens": self.max_tokens}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
        return (data["choices"][0]["message"].get("content") or "").strip()


@dataclass
class EvalAgents:
    world: object
    alloc: Dict[str, List[MemoryUnit]]
    llm: LLMBackend
    agents: List[str] = field(default_factory=lambda: ["finance", "compliance",
                                                       "logistics", "coordinator"])

    def accessible_units(self, victim: Optional[str]) -> Dict[str, List[MemoryUnit]]:
        return {a: self.alloc.get(a, []) for a in self.agents if a != victim}

    def structural_answer(self, task: Task,
                          units_by_agent: Dict[str, List[MemoryUnit]]) -> str:
        """Rule-solver answer from the SAME accessible evidence (no LLM)."""
        register_req_kinds([task])
        units = [u for us in units_by_agent.values() for u in us]
        satisfied = requirements_satisfied(units, None)
        if not task.covered_by(satisfied):
            return "INSUFFICIENT_EVIDENCE"
        # derive answer from the world facts referenced by any satisfied bundle
        for b in task.witness_bundles:
            if b.issubset(satisfied):
                return self._derive_answer(task, b, satisfied)
        return "INSUFFICIENT_EVIDENCE"

    def _derive_answer(self, task, bundle, satisfied) -> str:
        """Recompute the programmatic answer using only facts of a satisfied
        bundle (facts not in the bundle are treated as unknown)."""
        from .datagen import BUDGET_THRESHOLDS, DELIVERY_THRESHOLDS

        w = self.world
        q = task.query_type
        facts = {f for f, _ in bundle}
        vals = {f: w.facts[f]["value"] for f in facts}
        if q == "budget_check":
            (f1p,) = [f for f in facts if "price" in f]
            b = int(task.question.split("below ")[1].split("0,000")[0])
            return "yes" if vals[f1p] < b else "no"
        if q == "cheapest_pair":
            f1p, f2p = sorted(f for f in facts if "price" in f)
            s1, s2 = (w.facts[f]["supplier"] for f in (f1p, f2p))
            return s1 if vals[f1p] < vals[f2p] else s2
        if q == "cheapest_compliant":
            ps = sorted(f for f in facts if "price" in f)
            cs = sorted(f for f in facts if "compliant" in f)
            pairs = {w.facts[f]["supplier"]: (vals[f], None) for f in ps}
            for f in cs:
                pairs[w.facts[f]["supplier"]] = (
                    pairs[w.facts[f]["supplier"]][0], vals[f])
            ok = {s: (p, c) for s, (p, c) in pairs.items() if c is True}
            if not ok:
                return "none"
            best = min(ok, key=lambda s: ok[s][0])
            return best
        if q == "deliver_pair":
            ds = sorted(f for f in facts if "delivery" in f)
            d = int(task.question.split("within ")[1].split(" days")[0])
            return "yes" if all(vals[f] <= d for f in ds) else "no"
        # budget_and_delivery
        ps = sorted(f for f in facts if "price" in f)
        ds = sorted(f for f in facts if "delivery" in f)
        b = int(task.question.split("under budget ")[1].split("0,000")[0])
        d = int(task.question.split("within ")[1].split(" days")[0])
        ok = [w.facts[f]["supplier"] for f in ps
              if vals[f] < b and all(
                  vals[g] <= d for g in ds if w.facts[g]["supplier"] == w.facts[f]["supplier"])]
        return ok[0] if len(ok) == 1 else "none"

    def run_task(self, task: Task, victim: Optional[str] = None,
                 repair_flow: bool = False) -> dict:
        accessible = self.accessible_units(victim)
        evidence = []
        agent_responses = []
        for a in self.agents:
            if a not in accessible or not accessible[a]:
                continue
            mem_text = "\n".join(
                f"[{u.unit_id}|src:{u.source_id}|rule:{u.rule_id or '-'}] "
                f"{u.content}" for u in accessible[a])
            resp = self.llm.chat(AGENT_SYSTEM,
                                 f"Your memory:\n{mem_text}\n\nQuestion: {task.question}")
            if "NO_RELEVANT_EVIDENCE" in resp.upper():
                continue  # ECCR merge step: drop empty responses
            agent_responses.append((a, resp))
            evidence.append(f"Agent {a}:\n{resp}")
        ev_text = "\n\n".join(evidence) if evidence else "(no evidence available)"
        coord_system = COORD_SYSTEM_V2 if repair_flow else COORD_SYSTEM
        coord = self.llm.chat(coord_system,
                              f"Question: {task.question}\n\nEvidence:\n{ev_text}")
        structural = self.structural_answer(task, accessible)
        result = {
            "task_id": task.task_id,
            "victim": victim,
            "coord_answer": coord,
            "correct": task.correct_answer,
            "correct_llm": grade(coord, task.correct_answer),
            "structural_answer": structural,
            "correct_structural": grade(structural, task.correct_answer),
            "insufficient_llm": extract_answer(coord).startswith("insufficient"),
            "unsupported_llm": (not grade(coord, structural)
                                and not extract_answer(coord).startswith("insufficient")),
        }
        if repair_flow:
            # verify returned records -> gate -> one targeted repair round ->
            # answer from verified evidence or refuse
            verified = []
            for a, resp in agent_responses:
                verified.extend(get_records(resp, accessible.get(a, [])))
            sufficient = full_gate(task, verified)
            repaired, sufficient2 = (verified, True) if sufficient else \
                targeted_repair(task, verified, accessible)
            if sufficient2:
                ev2 = "\n\n".join(record_line(u) for u in repaired)
                coord2 = self.llm.chat(
                    coord_system,
                    f"Question: {task.question}\n\nEvidence:\n{ev2}")
            else:
                coord2 = "ANSWER: INSUFFICIENT_EVIDENCE"
            result.update({
                "repair_verified_n": len(verified),
                "repair_repaired_n": len(repaired) - len(verified),
                "repair_gate_pass": bool(sufficient2),
                "repair_answer": coord2,
                "correct_llm_repair": grade(coord2, task.correct_answer),
                "insufficient_llm_repair":
                    extract_answer(coord2).startswith("insufficient"),
            })
        return result
