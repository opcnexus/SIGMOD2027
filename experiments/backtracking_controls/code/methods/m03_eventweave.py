"""M03 EventWeave (2025) -- a two-stage core/supporting-event framework.

Style: chain (semi-global). First decide the core events (chain heads) globally, then link node by node:
      chain heads are identified with a topic-novelty heuristic (the first occurrence of a topic marks a new chain core),
      non-core nodes use the LLM to pick their direct upstream from the candidate set.
"""
from .base import Method
from .helpers import cands_full, shown_theme


class M03EventWeave(Method):
    name = "M03_eventweave"
    style = "chain"
    paper = "2025-arXiv2025-EventWeave.pdf"
    dataset_hint = "D2_eventstoryline"

    def predict(self, inst, llm, rng):
        # stage 1 (global): identify core / chain-head nodes -- the first occurrence of a topic is a chain head
        seen_themes = set()
        is_core = {}
        for i in range(inst.n):
            th = shown_theme(inst, i)
            is_core[i] = th not in seen_themes
            seen_themes.add(th)
        # stage 2: non-core nodes use the LLM to pick the direct upstream; core nodes become roots
        pred = {}
        for i in range(1, inst.n):
            if is_core[i]:
                pred[i] = -1
            else:
                pred[i] = llm.ask_parent(inst, i, cands_full(inst, i))
        return pred
