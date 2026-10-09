"""M10 Cascade-Inference-Problem (2024) -- debiased recovery of cascade networks.

Style: chain. First produce a debiased estimate using a structural prior (nearest neighbour / same topic),
      then combine it with two LLM judgements into a majority of three, and finally impose a tree/forest consistency constraint (acyclicity).

This matches the original paper's core conclusion: distortions in the reconstruction assumptions of cascade inference systematically skew analysis --
this implementation weakens such distortions with a structural prior plus a multi-vote ensemble.
"""
from .base import Method
from .helpers import cands_full, majority, prior_same_theme


class M10CascadeInference(Method):
    name = "M10_cascade_inference"
    style = "chain"
    paper = "2024-arXiv2024-Cascade-Inference-Problem.pdf"
    dataset_hint = "D4_cascade"

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            cand = cands_full(inst, i)
            # the structural prior uses recency (the nearest node) rather than topic matching, so it stays distinct from the content priors of ExpeL/Ananke
            struct = i - 1
            v1 = llm.ask_parent(inst, i, cand)
            v2 = llm.ask_parent(inst, i, cand)
            pred[i] = majority([struct, v1, v2])
        # forest consistency constraint: break residual cycles
        for i in range(1, inst.n):
            seen, cur = set(), i
            while cur != -1 and cur in pred:
                if cur in seen:
                    pred[i] = -1
                    break
                seen.add(cur)
                cur = pred[cur]
        return pred
