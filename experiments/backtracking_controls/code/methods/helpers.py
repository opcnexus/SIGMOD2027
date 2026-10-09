"""Shared method-level operators: majority vote, topic prior, subgraph recall (context reduction)."""
from collections import Counter

from datasets.base import ChainInstance


def majority(votes):
    """Majority vote (ties resolved by the earliest occurrence)."""
    if not votes:
        return -1
    return Counter(votes).most_common(1)[0][0]


def shown_theme(inst, i):
    return inst.items[i].meta.get("shown_theme", "")


def prior_same_theme(inst, i):
    """Heuristic prior: the most recent earlier node with the same topic; if none exists, declare a root (-1).

    Provides the content signal for experience/knowledge-base methods such as ExpeL, Ananke and EventWeave.
    """
    th = shown_theme(inst, i)
    for j in range(i - 1, -1, -1):
        if shown_theme(inst, j) == th:
            return j
    return -1


def recall_subgraph(inst, i, k=8):
    """Context reduction (Trace2ATT&CK style subgraph recall): keep the k most recent nodes, the same-topic nodes, and the root option."""
    cand = set(range(max(0, i - k), i))
    th = shown_theme(inst, i)
    for j in range(i):
        if shown_theme(inst, j) == th:
            cand.add(j)
    return [-1] + sorted(cand)


def cands_full(inst, i):
    """All candidates: [-1] (start of a new chain) plus every earlier node."""
    return [-1] + list(range(i))
