"""M09 Trace2ATT&CK (2026) -- a four-stage pipeline: graph construction -> subgraph recall -> staged LLM reasoning -> verification.

Style: chain. The core is "context reduction (subgraph recall)" plus "staged reasoning by majority of three".
      (In this implementation the verifier is realised as multi-vote consistency; a server version may replace it with an explicit second verification pass.)
"""
from .base import Method
from .helpers import majority, recall_subgraph


class M09Trace2ATTACK(Method):
    name = "M09_trace2attck"
    style = "chain"
    paper = "2026-arXiv2026-Trace2ATT&CK.pdf"
    dataset_hint = "D3_provenance"

    TOPK = 8

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            # stage 2: subgraph recall (context reduction) -- reasoning happens only inside the recalled subgraph
            cand = recall_subgraph(inst, i, k=self.TOPK)
            # stages 3 and 4: staged reasoning plus verification (three votes)
            v1 = llm.ask_parent(inst, i, cand)
            v2 = llm.ask_parent(inst, i, cand)
            v3 = llm.ask_parent(inst, i, cand)
            pred[i] = majority([v1, v2, v3])
        return pred
