"""Controlled mock oracle -- for local pipeline validation and metric-mechanism validation only; not a research conclusion.

Mechanism: returns the gold parent with probability p (correct), otherwise a random wrong candidate.
This makes the per-hop accuracy p exactly controllable, so the granularity gap p^L can be reproduced quantitatively at chain length L.

WARNING: never treat numbers produced by the mock as research results; real conclusions require a re-run with the real LLM through api.py on the server.
"""
import random

from .base import BaseLLM


class MockOracleLLM(BaseLLM):
    name = "mock-oracle"

    def __init__(self, p=0.90, seed=0):
        self.p = p
        self.rng = random.Random(seed)
        self.n_calls = 0
        self.n_correct = 0

    def ask_parent(self, inst, i, cands):
        self.n_calls += 1
        gold = inst.gold_parent.get(i, -1)
        wrong = [c for c in cands if c != gold]
        if gold in cands and self.rng.random() < self.p:
            self.n_correct += 1
            return gold
        return self.rng.choice(wrong) if wrong else (cands[0] if cands else -1)

    def describe(self):
        acc = self.n_correct / self.n_calls if self.n_calls else 0.0
        return "mock-oracle(p=%.2f, calls=%d, realized_acc=%.3f)" % (
            self.p,
            self.n_calls,
            acc,
        )
