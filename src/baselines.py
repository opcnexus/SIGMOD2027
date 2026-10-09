#!/usr/bin/env python3
"""Reproduction baselines: conversation disentanglement (reply-to link prediction)

Task definition (aligned with Kummerfeld et al. ACL 2019 + Zhu et al. ALTA 2021):
  Given a message stream and a target message i (Utterance of Interest), predict
  inside the window [i-100, i-1] which historical message(s) i replies to; a
  self-link means the root of a new conversation.

Baselines:
  B1 last-message     : always reply to the previous message (weak baseline)
  B2 same-speaker     : reply to the most recent message from the same speaker
  B3 mention          : reply to the most recent message of the user @/mentioned
                        in the text
  B4 tfidf-cosine     : reply to the highest TF-IDF cosine with i, with distance
                        decay
  B5 combined-linear  : a logical combination of handcrafted features (a simplified
                        version of the official 77 features)
  N1 neural-ptr-net   : lightweight pointer network (TF-IDF features + two-layer MLP
                        + softmax over the window)

Evaluation (official graph-eval.py protocol): link-level P/R/F1, test starts at
line>=1000.
"""
from __future__ import annotations

import math
import os
import re
import sys
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ChatLog, load_split, gold_graph_pairs

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_START = 1000
MAX_DIST = 101  # official default window: 101 messages including the current one

USER_PAT = re.compile(r"[A-Za-z0-9_\|\-\{\}\[\]\^`]+")


# ---------------- Features ----------------

def norm_speaker(sp: str) -> str:
    return sp.lower().strip()


def mentioned_users(text: str) -> set:
    """In IRC a mention is usually a leading 'name:' or 'name ,', or a name
    occurring inside the text"""
    out = set()
    m = re.match(r"^([A-Za-z0-9_\|\-\{\}\[\]\^`]+)\s*[:,]", text)
    if m:
        out.add(m.group(1).lower())
    return out


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


class TFIDF:
    def __init__(self):
        self.df = Counter()
        self.n_docs = 0
        self.idf = {}

    def fit(self, texts: List[str]):
        for t in texts:
            for w in set(t.lower().split()):
                self.df[w] += 1
        self.n_docs = len(texts)
        for w, c in self.df.items():
            self.idf[w] = math.log((1 + self.n_docs) / (1 + c)) + 1.0

    def vec(self, text: str) -> Dict[str, float]:
        tf = Counter(text.lower().split())
        return {w: c * self.idf.get(w, math.log(1 + self.n_docs) + 1.0) for w, c in tf.items()}

    @staticmethod
    def cosine(va: Dict[str, float], vb: Dict[str, float]) -> float:
        common = set(va) & set(vb)
        num = sum(va[w] * vb[w] for w in common)
        da = math.sqrt(sum(v * v for v in va.values()))
        db = math.sqrt(sum(v * v for v in vb.values()))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)


# ---------------- Baseline predictors ----------------

class Baseline:
    name = "base"

    def predict_parents(self, log: ChatLog, i: int) -> List[int]:
        """Return the list of ids of the messages that message i is predicted to
        reply to (may be empty = no prediction, or [i] = root)"""
        raise NotImplementedError


class LastMessage(Baseline):
    name = "B1-last"

    def predict_parents(self, log, i):
        if i - 1 >= 0:
            return [i - 1]
        return [i]


class SameSpeaker(Baseline):
    name = "B2-speaker"

    def predict_parents(self, log, i):
        sp = norm_speaker(log.messages[i].speaker)
        if not sp:
            return [i - 1] if i > 0 else [i]
        for j in range(i - 1, max(-1, i - MAX_DIST), -1):
            if norm_speaker(log.messages[j].speaker) == sp:
                return [j]
        return [i]


class Mention(Baseline):
    name = "B3-mention"

    def predict_parents(self, log, i):
        mens = mentioned_users(log.messages[i].text)
        if not mens:
            return [i - 1] if i > 0 else [i]
        for j in range(i - 1, max(-1, i - MAX_DIST), -1):
            if norm_speaker(log.messages[j].speaker) in mens:
                return [j]
        return [i]


class TfidfBaseline(Baseline):
    name = "B4-tfidf"

    def __init__(self, tfidf: TFIDF):
        self.tfidf = tfidf

    def predict_parents(self, log, i):
        vi = self.tfidf.vec(log.messages[i].text)
        best, bs = None, -1.0
        for j in range(i - 1, max(-1, i - MAX_DIST), -1):
            vj = self.tfidf.vec(log.messages[j].text)
            s = TFIDF.cosine(vi, vj) / (1 + 0.02 * (i - j))  # distance decay
            if s > bs:
                bs, best = s, j
        if best is None or bs < 0.02:
            return [i]  # nothing similar -> root
        return [best]


class CombinedLinear(Baseline):
    """Simplified handcrafted-feature scoring (in the spirit of the official
    77 features)"""
    name = "B5-combined"

    def __init__(self, tfidf: TFIDF, w: Optional[Dict[str, float]] = None):
        self.tfidf = tfidf
        # feature weights (set by hand, coarsely tuned on dev)
        self.w = w or {
            "dist_log": -0.8,
            "same_speaker": 1.2,
            "mention": 2.5,
            "overlap": 1.5,
            "cosine": 2.0,
            "other_is_question_and_i_short": 0.6,
            "prev1": 0.9,
            "root_bias": 0.35,
        }

    def score(self, log, i, j) -> float:
        mi, mj = log.messages[i], log.messages[j]
        dist = abs(i - j)
        s = self.w["dist_log"] * math.log(1 + max(dist, 0))
        s += self.w["same_speaker"] * (norm_speaker(mi.speaker) == norm_speaker(mj.speaker) and mi.speaker != "")
        s += self.w["mention"] * (norm_speaker(mj.speaker) in mentioned_users(mi.text))
        s += self.w["overlap"] * token_overlap(mi.text, mj.text)
        s += self.w["cosine"] * TFIDF.cosine(self.tfidf.vec(mi.text), self.tfidf.vec(mj.text))
        if mj.text.rstrip().endswith("?") and len(mi.text.split()) <= 8:
            s += self.w["other_is_question_and_i_short"]
        if j == i - 1:
            s += self.w["prev1"]
        return s

    def predict_parents(self, log, i):
        scores = []
        for j in range(max(0, i - MAX_DIST + 1), i):
            scores.append((self.score(log, i, j), j))
        scores.sort(reverse=True)
        root_s = self.w["root_bias"]
        if not scores or scores[0][0] < root_s:
            return [i]
        top = [scores[0][1]]
        # multi-parent link: also add a candidate whose score is high and close to
        # the top one (mimics the official multi-link behaviour)
        if len(scores) > 1 and scores[1][0] > scores[0][0] - 0.05 and scores[1][0] > 0.5:
            top.append(scores[1][1])
        return top


# ---------------- Evaluation (official protocol) ----------------

def evaluate_split(logs: List[ChatLog], predictor: Baseline, test_start: int = TEST_START) -> Dict[str, float]:
    """link-level P/R/F1: gold edges (src,tgt) with src>=test_start; self-links are
    counted"""
    total_gold = total_auto = matched = 0
    for log in logs:
        gold = set()
        for src, tgts in log.parents.items():
            if src < test_start:
                continue
            for t in tgts:
                gold.add((src, t))
        auto = set()
        for i in range(test_start, len(log.messages)):
            for t in predictor.predict_parents(log, i):
                auto.add((i, t))
        total_gold += len(gold)
        total_auto += len(auto)
        matched += len(gold & auto)
    p = 100 * matched / total_auto if total_auto else 0.0
    r = 100 * matched / total_gold if total_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"P": round(p, 1), "R": round(r, 1), "F1": round(f, 1),
            "gold": total_gold, "auto": total_auto, "matched": matched}


# ---------------- main ----------------

def build_tfidf(logs: List[ChatLog]) -> TFIDF:
    tf = TFIDF()
    tf.fit([m.text for log in logs for m in log.messages])
    return tf


def main():
    random.seed(10)
    print("loading data ...")
    train_logs = load_split("train", DATA_ROOT)
    dev_logs = load_split("dev", DATA_ROOT)
    test_logs = load_split("test", DATA_ROOT)
    print("train/dev/test files:", len(train_logs), len(dev_logs), len(test_logs))

    print("building tfidf on train ...")
    tfidf = build_tfidf(train_logs)

    baselines = [
        LastMessage(),
        SameSpeaker(),
        Mention(),
        TfidfBaseline(tfidf),
        CombinedLinear(tfidf),
    ]

    results = {}
    for b in baselines:
        r_dev = evaluate_split(dev_logs, b)
        print("{:14s} dev  {}".format(b.name, r_dev))
        results[b.name] = {"dev": r_dev}

    print("\n== test ==")
    best_name = None
    for b in baselines:
        r_test = evaluate_split(test_logs, b)
        print("{:14s} test {}".format(b.name, r_test))
        results[b.name]["test"] = r_test

    out = "./results/baselines.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
