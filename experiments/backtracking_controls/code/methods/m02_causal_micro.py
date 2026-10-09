"""M02 Causal-Micro-Narratives (2024) -- causal micro-narrative graph plus cycle-consistency repair.

Style: 1-to-1 (per-node independent prediction) plus a lightweight global repair enforcing that the causal graph is acyclic:
      edges that close a cycle are re-queried once (the temporal-consistency constraint of causal narratives).
"""
from .base import Method, chain_from
from .helpers import cands_full


class M02CausalMicro(Method):
    name = "M02_causal_micro"
    style = "1to1"
    paper = "2024-arXiv2024-Causal-Micro-Narratives.pdf"
    dataset_hint = "D2_eventstoryline"

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            pred[i] = llm.ask_parent(inst, i, cands_full(inst, i))
        # causal-consistency repair: detect nodes on a cycle and re-query once (one extra vote)
        for i in range(1, inst.n):
            if self._in_cycle(pred, i):
                cand = cands_full(inst, i)
                retry = llm.ask_parent(inst, i, cand)
                pred[i] = retry
        return pred

    @staticmethod
    def _in_cycle(pred, i):
        seen, cur = set(), i
        while cur != -1 and cur in pred:
            if cur in seen:
                return True
            seen.add(cur)
            cur = pred[cur]
        return False
