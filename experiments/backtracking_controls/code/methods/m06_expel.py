"""M06 ExpeL (2023, EMNLP) -- agents learn from and reuse experience.

Style: 1-to-1. Experience is formalised as a content prior (the most recent same-topic node) and combined with two LLM judgements into a majority of three.
"""
from .base import Method
from .helpers import cands_full, majority, prior_same_theme


class M06ExpeL(Method):
    name = "M06_expel"
    style = "1to1"
    paper = "2023-EMNLP2023-ExpeL.pdf"
    dataset_hint = "D5_agent_traj"

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            cand = cands_full(inst, i)
            exp = prior_same_theme(inst, i)  # reuse historical experience (content prior)
            v1 = llm.ask_parent(inst, i, cand)
            v2 = llm.ask_parent(inst, i, cand)
            pred[i] = majority([exp, v1, v2])
        return pred
