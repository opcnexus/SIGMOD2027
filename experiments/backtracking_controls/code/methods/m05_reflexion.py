"""M05 Reflexion (2023, NeurIPS) -- verbal-reflection self-backtracking.

Style: 1-to-1 (per-node independent), but each decision uses a reflection-style multi-round vote:
      first judgement -> reflective second judgement -> a third vote if they disagree -> majority of three (effective accuracy 3p^2-2p^3).
      Note: decisions remain independent per hop, so chaining still decays as p_eff^L.
"""
from .base import Method
from .helpers import cands_full, majority


class M05Reflexion(Method):
    name = "M05_reflexion"
    style = "1to1"
    paper = "2023-NeurIPS2023-Reflexion.pdf"
    dataset_hint = "D5_agent_traj"

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            cand = cands_full(inst, i)
            v1 = llm.ask_parent(inst, i, cand)
            v2 = llm.ask_parent(inst, i, cand)  # reflective second judgement
            if v1 == v2:
                pred[i] = v1
                continue
            v3 = llm.ask_parent(inst, i, cand)  # third vote breaks the tie
            pred[i] = majority([v1, v2, v3])
        return pred
