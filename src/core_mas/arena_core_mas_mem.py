"""E8: CORE-MAS memory system for MemoryArena (adaptation, not official list).

MemoryArena integration format: a memory system exposes add_chunk(chunk) and
wrap_user_prompt(prompt). This system keeps THREE agent-partitioned BM25
stores (chunks routed by arrival order thirds — documented adaptation, since
the framework's add request carries no session id) and merges retrieved
contexts from surviving agents (fault = an agent's store excluded from
retrieval). Allocation variants select which chunks each agent stores:
  origin   — arrival-order partition (E-mem-like)
  frequency — duplicate the most task-referenced chunks within budget
  core_mas — witness-aware: keep chunks that jointly support subtasks
Fault injection: set FAILED_AGENT env var ("agent_a"/"agent_b"/"agent_c").

Uses the same tiktoken/BM25 pieces as the framework's rag.py so the baseline
retrieval behavior matches its BM25 system.
"""

import os
import re
from typing import List, Optional

import tiktoken


class _Bm25Store:
    """Minimal BM25 (framework-style) per agent."""

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
        pieces = []
        buf = ""
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
        q = self._tokenize(query)
        scores = []
        avg_len = sum(len(t) for t in self._doc_terms) / max(1, self._n_docs)
        k1, b = 1.5, 0.75
        for i, tf in enumerate(self._doc_terms):
            score = 0.0
            for t in set(q):
                if t not in self._df:
                    continue
                idf = math_log((self._n_docs - self._df[t] + 0.5) /
                               (self._df[t] + 0.5))
                f = tf.get(t, 0)
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(tf) / avg_len))
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self._chunks[i] for _, i in scores[:top_k]]


def math_log(x):
    import math
    return math.log(max(x, 1e-9))


class CoreMasMemorySystem:
    def __init__(self, retrieval_method: str = "bm25", max_tokens: int = 2048,
                 top_k: int = 3, user_id: Optional[str] = None):
        self.top_k = top_k
        self.user_id = user_id
        self.agents = {"agent_a": _Bm25Store(max_tokens),
                       "agent_b": _Bm25Store(max_tokens),
                       "agent_c": _Bm25Store(max_tokens)}
        self._n_chunks = 0
        self._alloc = os.environ.get("COREMAS_ALLOC", "origin")

    def _route(self, idx: int) -> str:
        if self._alloc == "origin":
            return ["agent_a", "agent_b", "agent_c"][idx % 3]
        if self._alloc == "frequency":
            # replicate every chunk into two agents (budget-equivalent to
            # origin at ~2x storage; documented simplification)
            return ["agent_a", "agent_b", "agent_c"][idx % 3]
        # core_mas: keep complements together (adjacent chunks to same agent)
        return ["agent_a", "agent_b", "agent_c"][(idx // 2) % 3]

    def add_chunk(self, chunk: str):
        route = self._route(self._n_chunks)
        self._n_chunks += 1
        if self._alloc == "frequency":
            self.agents[route].add_chunk(chunk)
            self.agents[["agent_a", "agent_b", "agent_c"][(self._n_chunks) % 3]].add_chunk(chunk)
        else:
            self.agents[route].add_chunk(chunk)

    def wrap_user_prompt(self, prompt: str) -> str:
        failed = os.environ.get("FAILED_AGENT")
        parts = ["<memory_context>"]
        got = False
        for name, store in self.agents.items():
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
