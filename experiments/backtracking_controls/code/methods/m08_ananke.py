"""M08 Ananke (2025) -- knowledge-base (KB) augmented LLM attack investigation.

Style: chain. Build a domain knowledge base (topic -> most-recent-node registry) and use it to propose structured candidates,
      then combine them with two LLM judgements into a majority of three (an ensemble of KB constraint and LLM reasoning).

WARNING when reproducing: the original paper was criticised for a "knowledge base is circular with respect to the evaluation set" risk;
   when transferring the method to your system, the knowledge-base corpus must be strictly separated from the evaluation set.
"""
from .base import Method
from .helpers import cands_full, majority, shown_theme


class M08Ananke(Method):
    name = "M08_ananke"
    style = "chain"
    paper = "2025-arXiv2025-Ananke.pdf"
    dataset_hint = "D3_provenance"

    def predict(self, inst, llm, rng):
        # knowledge base: topic -> most recently seen node (updated as the scan proceeds)
        kb = {}
        pred = {}
        for i in range(1, inst.n):
            th = shown_theme(inst, i)
            kb_candidate = kb.get(th, -1)
            # KB hit: majority of three, combining the KB constraint with two LLM judgements
            # KB miss (brand-new topic): declare a new chain root directly without calling the LLM (this is the KB acting as a constraint)
            if kb_candidate == -1:
                pred[i] = -1
                kb[th] = i
                continue
            cand = cands_full(inst, i)
            v1 = llm.ask_parent(inst, i, cand)
            v2 = llm.ask_parent(inst, i, cand)
            pred[i] = majority([kb_candidate, v1, v2])
            kb[th] = i  # update the knowledge base
        return pred
