# -*- coding: utf-8 -*-
"""
FFScorer: adapts the trained FF model (277-dim features) into a scorer usable by BacktrackEngine.

Protocol (aligned with baselines.Baseline):
- predict_parents(log, i) -> List[int]  # [i] means root
- score(log, i, j) -> float             # softmax probability
- .tfidf                                # used by BacktrackEngine.locate for text lookup

Usage:
    from ff_engine_adapter import load_ff_scorer
    scorer = load_ff_scorer()            # loads results/ff_model_glove.pt
    eng = BacktrackEngine(model=scorer, logs=logs)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ChatLog, load_split
from ff_reproduction import FFNet, LogPrep, load_glove, DATA_ROOT
from baselines import TFIDF

import torch

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "ff_model_glove.pt")


class FFScorer:
    name = "FF-277d"

    def __init__(self, model: FFNet, emb: dict, tfidf: TFIDF):
        self.model = model
        self.emb = emb
        self.tfidf = tfidf
        self._prep_cache = {}  # log.name -> LogPrep

    # ---- internals ----

    def _prep(self, log: ChatLog) -> LogPrep:
        p = self._prep_cache.get(log.name)
        if p is None:
            p = LogPrep(log, emb=self.emb)
            self._prep_cache[log.name] = p
        return p

    def _probs(self, log: ChatLog, i: int):
        """Returns (cands, softmax probability array)"""
        p = self._prep(log)
        cands = p.candidates(i)
        if not cands:
            return [], np.array([])
        feats = np.array([p.features(i, c) for c in cands], dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(feats))
            probs = torch.softmax(logits, dim=0).numpy()
        return cands, probs

    # ---- protocol interface ----

    def predict_parents(self, log: ChatLog, i: int):
        cands, probs = self._probs(log, i)
        if not cands:
            return [i]
        top = int(probs.argmax())
        return [cands[top]]

    def score(self, log: ChatLog, i: int, j: int) -> float:
        cands, probs = self._probs(log, i)
        if j not in cands:
            return 0.0
        return float(probs[cands.index(j)])


def load_ff_scorer(logs=None, ckpt: str = CKPT):
    """Loads the trained FF+GloVe model; logs is used to build the tfidf required by locate."""
    device = "cpu"
    model = FFNet(in_dim=277)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))

    if logs is None:
        logs = load_split("test", DATA_ROOT) + load_split("dev", DATA_ROOT)
    tf = TFIDF()
    tf.fit([m.text for l in logs for m in l.messages])

    return FFScorer(model, emb, tf)


if __name__ == "__main__":
    # self-test: score the first dev message under the official protocol
    scorer = load_ff_scorer()
    logs = load_split("dev", DATA_ROOT)
    log = logs[0]
    i = 1006
    parents = scorer.predict_parents(log, i)
    print("query [{}] {}: {}".format(i, log.messages[i].speaker, log.messages[i].text[:50]))
    print("FF predicted parent:", parents, "| gold:", log.parents.get(i))
    print("score(1006->1005) =", round(scorer.score(log, 1006, 1005), 4))
