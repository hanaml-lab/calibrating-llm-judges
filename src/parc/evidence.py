"""Evidence retrieval for one criterion: the steps of a run that a careful person would read to check it."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    index: int
    action: str
    arguments: str
    observation: str

    def text(self, max_chars: int = 6000) -> str:
        return f"{self.action} {self.arguments} {self.observation[:max_chars]}"


_TOKEN = re.compile(r"[a-z0-9_.\-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in set(query):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / max(avg, 1e-9)))
        scores.append(s)
    return scores


def retrieve(steps: list[Step], criterion: str, locator: list[str], k: int = 8, kind: str | None = None) -> list[Step]:
    """Top-k steps by BM25 over the criterion text, with exact locator hits ranked first; for communication
    criteria the final step is always included. Returned in the run's order."""
    docs = [_tokens(s.text()) for s in steps]
    scores = _bm25(_tokens(criterion), docs)
    lowered = [s.text().lower() for s in steps]
    hit = [any(loc.lower() in t for loc in locator) for t in lowered]
    ranked = sorted(range(len(steps)), key=lambda i: (not hit[i], -scores[i], i))
    chosen = ranked[:k]
    if kind == "comms" and steps and (len(steps) - 1) not in chosen:
        chosen = chosen[: max(k - 1, 0)] + [len(steps) - 1]
    return [steps[i] for i in sorted(chosen)]
