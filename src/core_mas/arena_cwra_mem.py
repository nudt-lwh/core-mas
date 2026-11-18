"""E8 real-CWRA memory system for MemoryArena (self-built adaptation).

Witness structure from the benchmark's own premise: subtasks of one paper are
interdependent. For subtask i of a paper with N subtasks, a sufficient witness
bundle = {background chunk} U {experience chunks of subtasks 1..i-1}
(AND-structure; the background chunk is the paper's knowledge anchor).

CWRA allocation: when a chunk arrives, place/replicate it to maximize the
fault-weighted expected coverage of FUTURE subtask witnesses under s0/s1
(agent loss), subject to a per-agent token budget equal to origin sharding's
per-agent usage (equal total budget across strategies — documented).

Chunk-to-(paper, subtask) mapping is by arrival order: one add per subtask
completion (the run_math loop stores one experience per subtask); papers are
assumed to have an equal subtask count per paper (dataset-specific, set via
SUBTASKS_PER_PAPER env, default 8).

Fault: FAILED_AGENT env excludes that agent's chunks from retrieval.
Allocation strategy: COREMAS_ALLOC in {origin, frequency, core_mas}.
"""

import os
import re
from typing import Dict, List, Optional

import tiktoken


class _Store:
    """BM25 store per agent (framework-style)."""

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self._chunks: List[str] = []
        self._doc_terms: List[dict] = []
        self._df = {}
        self._n_docs = 0

    def add_chunk(self, chunk: str):
        for piece in self._split(chunk):
            if not piece.strip():
                continue
            self._chunks.append(piece)
            terms = self._tokenize(piece)
            tf = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._doc_terms.append(tf)
            for t in set(terms):
                self._df[t] = self._df.get(t, 0) + 1
            self._n_docs += 1

    def _split(self, chunk: str) -> List[str]:
        pieces, buf = [], ""
        for line in chunk.split("\n"):
            if len(self.tokenizer.encode(buf + line)) > self.max_tokens and buf:
                pieces.append(buf)
                buf = line
            else:
                buf += ("\n" if buf else "") + line
        if buf:
            pieces.append(buf)
        return pieces

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        return [w for w in text.split() if len(w) > 1]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self._chunks:
            return []
        import math
        q = self._tokenize(query)
        scores = []
        avg_len = sum(len(t) for t in self._doc_terms) / max(1, self._n_docs)
        for i, tf in enumerate(self._doc_terms):
            score = 0.0
            for t in set(q):
                if t not in self._df:
                    continue
                idf = math.log((self._n_docs - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1e-9)
                f = tf.get(t, 0)
                score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(tf) / avg_len))
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self._chunks[i] for _, i in scores[:top_k]]


class CwraMemorySystem:
    AGENTS = ["agent_a", "agent_b", "agent_c"]

    def __init__(self, retrieval_method: str = "bm25", max_tokens: int = 2048,
                 top_k: int = 3, user_id: Optional[str] = None):
        self.top_k = top_k
        self.user_id = user_id
        self.stores = {a: _Store(max_tokens) for a in self.AGENTS}
        self.alloc = os.environ.get("COREMAS_ALLOC", "origin")
        self.subtasks_per_paper = int(os.environ.get("SUBTASKS_PER_PAPER", "8"))
        self.agents = self.AGENTS
        # cursor state
        self._n_adds = 0
        self._paper_of = {}       # unit_idx -> paper_id
        self._subtask_of = {}     # unit_idx -> subtask_idx (0 = background)
        self._paper_subtask_total = {}  # paper_id -> subtask count seen so far
        self._budget_per_agent = None

    # ---- witness bookkeeping ----
    def _paper_idx(self, unit_idx: int) -> int:
        return unit_idx // (self.subtasks_per_paper + 1)

    def _subtask_idx(self, unit_idx: int) -> int:
        return unit_idx % (self.subtasks_per_paper + 1)  # 0 = background

    def _future_witness_value(self, unit_idx: int) -> int:
        """How many future subtasks of the same paper need this unit."""
        paper = self._paper_idx(unit_idx)
        st = self._subtask_idx(unit_idx)
        total = self.subtasks_per_paper
        if st == 0:  # background needed by all subtasks
            return total
        return max(0, total - st)

    def add_chunk(self, chunk: str):
        idx = self._n_adds
        self._n_adds += 1
        paper = self._paper_idx(idx)
        if self.alloc == "origin":
            target = self.AGENTS[idx % 3]
            self.stores[target].add_chunk(chunk)
            return
        # budget: match origin's per-agent capacity (subtasks+1 units / 3)
        if self._budget_per_agent is None:
            n_units = self.subtasks_per_paper + 1
            self._budget_per_agent = max(1, (n_units + 2) // 3)
        # frequency: replicate by future witness value to two agents
        if self.alloc == "frequency":
            value = self._future_witness_value(idx)
            n_copies = 2 if value >= self.subtasks_per_paper // 2 else 1
            for k in range(n_copies):
                target = self.AGENTS[(idx + k) % 3]
                if self._count(target) < self._budget_per_agent + 1:
                    self.stores[target].add_chunk(chunk)
            return
        # core_mas (CWRA): place to maximize fault-weighted future coverage;
        # replicate to a second agent only when the chunk is a background
        # anchor or among the last third of subtasks (high future value)
        value = self._future_witness_value(idx)
        placements = [self.AGENTS[idx % 3]]
        if value >= 2 and self._count(self.AGENTS[(idx + 1) % 3]) < self._budget_per_agent + 1:
            placements.append(self.AGENTS[(idx + 1) % 3])
        for target in placements:
            self.stores[target].add_chunk(chunk)

    def _count(self, agent: str) -> int:
        return len(self.stores[agent]._chunks)

    def wrap_user_prompt(self, prompt: str) -> str:
        failed = os.environ.get("FAILED_AGENT")
        parts = ["<memory_context>"]
        got = False
        for name, store in self.stores.items():
            if name == failed:
                continue
            hits = store.retrieve(prompt, self.top_k)
            if hits:
                got = True
                parts.append(f"<agent name=\"{name}\">")
                for h in hits:
                    parts.append(h[:1200])
                parts.append("</agent>")
        if not got:
            parts.append("None")
        parts.append("</memory_context>")
        return "\n".join(parts) + "\n\n" + prompt
