"""M01 Event-Extraction-LLM (2025) -- event extraction followed by independent per-node linking.

Style: 1-to-1 (each hop calls the LLM independently, no global consistency) -> chaining accumulates error as p^L
"""
from .base import Method
from .helpers import cands_full


class M01EventExtraction(Method):
    name = "M01_event_extraction"
    style = "1to1"
    paper = "2025-arXiv2025-Event-Extraction-LLM.pdf"
    dataset_hint = "D2_eventstoryline"

    def predict(self, inst, llm, rng):
        pred = {}
        for i in range(1, inst.n):
            pred[i] = llm.ask_parent(inst, i, cands_full(inst, i))
        return pred
