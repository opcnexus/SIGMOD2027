# -*- coding: utf-8 -*-
"""GTS-Backtrack v1 (experimental implementation): Graph-semantic Topic-completeness Backtracking

Positioning in the literature (see reports/experiment_report_gts_backtrack.md):
- Prior art exists in the graph/structural-method family: Kummerfeld et al. ACL 2019 (FF, P19-1374),
  Ma et al. ACL 2022 (Struct/r-GCN, 2022.acl-long.23), Li et al. 2023 (heterogeneous graph EGCN +
  easy-first decoding, arXiv:2306.03975).
- However, there is no prior work that "explicitly trains a topic-completeness model and uses a convergence criterion to control when backtracking stops".
- This implementation = modifications on top of that family: a shared encoder (FF 277-dim features + pretrained weights)
  + a new CompleteHead (whether a candidate message, merged into the current topic thread, is "converged/complete")
  + two-stage backtracking decoding (Root gate -> window backtracking -> topic check -> completeness gate -> stop).

Evaluation: the official link-level protocol (dev/test from line>=1000, ordered-pair P/R/F1).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import (DATA_ROOT, CKPT_DIR, TEST_START, FFNet, LogPrep,
                             load_glove, get_time_diff, gold_edge_sets,
                             link_prf, short_name)

RESULTS_DIR = os.path.dirname(CKPT_DIR)  # <workspace>/results
FF_CKPT = os.path.join(CKPT_DIR, "ff_model_glove.pt")


# ---------------- utility: union-find (with component aggregate statistics) ----------------

class ComponentTracker:
    """Incremental union-find maintaining per-component aggregate features (size / centroid / speakers / earliest member / max link probability)"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.earliest = list(range(n))          # earliest (smallest) message id in the component
        self.max_prob = [0.0] * n               # max parent-child link probability in the component
        self.centroid = [None] * n              # mean message vector in the component (100,)
        self.spk = [{-1: 1}] * n                # placeholder; really assigned in add_node

    def add_node(self, i: int, vec: np.ndarray, speaker):
        self.centroid[i] = vec.copy()
        self.spk[i] = {speaker: 1}
        self.earliest[i] = i
        self.size[i] = 1
        self.max_prob[i] = 0.0

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int, prob: float):
        """a=child, b=parent, prob is that link's probability"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.max_prob[ra] = max(self.max_prob[ra], prob)
            return
        # merge by size: rb is merged into ra, or vice versa
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.earliest[ra] = min(self.earliest[ra], self.earliest[rb])
        self.max_prob[ra] = max(self.max_prob[ra], self.max_prob[rb], prob)
        c = self.centroid[ra] + self.centroid[rb] if self.centroid[ra] is not None else self.centroid[rb]
        self.centroid[ra] = c
        for k, v in self.spk[rb].items():
            self.spk[ra][k] = self.spk[ra].get(k, 0) + v
        self.spk[rb] = {}

    def stats(self, i: int):
        r = self.find(i)
        return r, self.size[r], self.earliest[r], self.max_prob[r], self.centroid[r], self.spk[r]


def gold_components(log):
    """Undirected connected components of the full gold graph (union-find). Self-link = root, no edge added.
    Returns comp: dict msg->root"""
    n = len(log.messages)
    tr = ComponentTracker(n)
    for i, m in enumerate(log.messages):
        tr.add_node(i, None, m.speaker)
    for src, tgts in log.parents.items():
        for t in tgts:
            if t != src and 0 <= t < n:
                tr.union(src, t, 1.0)
    return {i: tr.find(i) for i in range(n)}


def past_components(log, upto: int):
    """Builds the union-find using only gold edges with source < upto (to mimic the decoder-time state)"""
    n = len(log.messages)
    tr = ComponentTracker(n)
    for i, m in enumerate(log.messages):
        tr.add_node(i, None, m.speaker)
    for src, tgts in log.parents.items():
        if src >= upto:
            continue
        for t in tgts:
            if t != src and 0 <= t < n:
                tr.union(src, t, 1.0)
    return tr


# ---------------- CompleteHead ----------------

class CompleteHead(nn.Module):
    """Topic-completeness head: takes the aggregate features of (candidate message, target thread) -> P(same thread, keep backtracking)"""

    N_FEATS = 12

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CompleteHead.N_FEATS, hidden), nn.Softsign(),
            nn.Linear(hidden, hidden // 2), nn.Softsign(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def thread_agg_features(tr: ComponentTracker, j: int, i: int, p_link: float,
                        prep: LogPrep, best_prob: float, rank: int,
                        win: int, p_root_i: float):
    """Builds the 12-dim input features of CompleteHead (the thread of j = the component containing j in tr)"""
    root, size, earliest, max_prob, centroid, spk = tr.stats(j)
    qv = prep.utt_vec(i)
    if centroid is not None and np.linalg.norm(centroid) > 0:
        denom = np.linalg.norm(centroid) * np.linalg.norm(qv) + 1e-9
        cohesion = float(np.dot(qv, centroid) / denom)
    else:
        cohesion = 0.0
    spk_i = prep.log.messages[i].speaker
    spk_total = sum(spk.values())
    same_ratio = spk.get(spk_i, 0) / max(1, spk_total)
    span = min(100, i - earliest) / 100.0
    dist = min(100, i - j) / 100.0
    td = get_time_diff(prep.info, i, j)
    td = min(120, td) / 60.0
    same_spk = 1.0 if prep.log.messages[j].speaker == spk_i else 0.0
    return [
        p_link,
        p_link / max(best_prob, 1e-9),
        rank / max(1, win),
        dist,
        same_spk,
        td,
        p_root_i,
        math.log1p(size),
        max_prob,
        cohesion,
        same_ratio,
        span,
    ]


# ---------------- batched FF probability computation ----------------

def ff_probs(prep: LogPrep, model: FFNet, i: int):
    cands = prep.candidates(i)
    feats = np.array([prep.features(i, c) for c in cands], dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats))
        probs = torch.softmax(logits, dim=0).numpy()
    return cands, probs


# ---------------- CompleteHead training data (streaming, teacher forcing) ----------------

def stream_complete_instances(train_logs, ff_model, emb, max_files=None):
    """Streams (feat12, label) training samples:
    - for every annotated query i, scan its window candidates j
    - label = 1 if j and i lie in the same connected component of the gold graph (same topic -> keep backtracking), else 0
    - thread features use gold edges with source<i (teacher forcing, mimicking the decoder-time point)
    - negative downsampling: at most 8 per query (to balance positives and negatives)"""
    n_files = len(train_logs) if max_files is None else min(max_files, len(train_logs))
    rng = np.random.RandomState(7)
    for li in range(n_files):
        log = train_logs[li]
        prep = LogPrep(log, emb=emb, cache_feats=False)
        comp_full = gold_components(log)
        n = len(log.messages)
        tr = ComponentTracker(n)
        for k, m in enumerate(log.messages):
            tr.add_node(k, None, m.speaker)
        edge_ptr = 0
        # gold edges sorted by source
        edges = sorted(((s, t) for s, ts in log.parents.items() for t in ts
                        if t != s), key=lambda x: x[0])
        queries = sorted(set(k for k in log.parents if 1 <= k < n))
        for qi in queries:
            # add the edges with source < qi
            while edge_ptr < len(edges) and edges[edge_ptr][0] < qi:
                s, t = edges[edge_ptr]
                if t < n:
                    tr.union(s, t, 1.0)
                edge_ptr += 1
            cands, probs = ff_probs(prep, ff_model, qi)
            order = np.argsort(-probs)
            rank_of = {cands[r]: r for r in order}
            best = float(probs[order[0]])
            p_root = float(probs[cands.index(qi)])
            comp_i = comp_full.get(qi)
            pos, neg = [], []
            for k_, j in enumerate(cands):
                if j == qi:
                    continue
                f12 = thread_agg_features(tr, j, qi, float(probs[k_]), prep,
                                          best, rank_of[j], len(cands), p_root)
                if comp_full.get(j) is not None and comp_i is not None and comp_full[j] == comp_i:
                    pos.append(f12)
                else:
                    neg.append(f12)
            if pos:
                for f12 in pos:
                    yield (f12, 1.0)
            if neg:
                idx = rng.choice(len(neg), size=min(8, len(neg)), replace=False)
                for k_ in idx:
                    yield (neg[k_], 0.0)
        # teardown: release at end of file
        del prep, tr


def train_complete_head(train_logs, ff_model, emb, epochs=2, max_files=None,
                        lr=1e-3, report_every=20000):
    head = CompleteHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    step, t0 = 0, time.time()
    n_pos = n_tot = 0
    for ep in range(epochs):
        # re-stream the whole split every epoch (features computed on the fly)
        ep_pos = ep_tot = 0
        buf_x, buf_y = [], []
        for f12, y in stream_complete_instances(train_logs, ff_model, emb, max_files):
            buf_x.append(f12)
            buf_y.append(y)
            ep_tot += 1
            ep_pos += y
            if len(buf_x) >= 4096:
                step = _head_step(head, opt, bce, buf_x, buf_y, step, report_every, t0)
                buf_x, buf_y = [], []
        if buf_x:
            step = _head_step(head, opt, bce, buf_x, buf_y, step, report_every, t0)
        n_pos, n_tot = ep_pos, ep_tot
        print("  complete-head ep{}: {} samples, pos={:.1%} ({:.0f}s)".format(
            ep, ep_tot, ep_pos / max(1, ep_tot), time.time() - t0), flush=True)
    return head


def _head_step(head, opt, bce, buf_x, buf_y, step, report_every, t0):
    x = torch.tensor(np.array(buf_x, dtype=np.float32))
    y = torch.tensor(np.array(buf_y, dtype=np.float32))
    logits = head(x)
    loss = bce(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += len(buf_y)
    if step % report_every < len(buf_y):
        print("    head step{} loss={:.4f} ({:.0f}s)".format(
            step, loss.item(), time.time() - t0), flush=True)
    return step


# ---------------- two-stage backtracking decoding ----------------

def backtrack_decode(log, prep, ff_model, head, tau_topic=0.05, tau_complete=0.5,
                     tau_root=None, head_thresh_hold=None):
    """GTS-Backtrack inference:
    1) Root gate (optional): if the best FF probability < tau_root -> predict root
    2) Backtracking scan j=i-1..window start:
       - topic check: P(i->j) >= tau_topic -> same-topic candidate
       - completeness check: CompleteHead(i, thread(j)) >= tau_complete -> converged, stop
       - otherwise keep backtracking (a same-topic candidate that has not converged joins A; a different-topic one is simply skipped and the scan continues)
    3) parent = the candidate in A with the highest FF probability; if A is empty -> root
    Returns (edges, tracker): edges is the predicted edge set for this file, tracker is reused incrementally by later messages
    """
    n = len(log.messages)
    tr = ComponentTracker(n)
    for k, m in enumerate(log.messages):
        tr.add_node(k, prep.utt_vec(k), m.speaker)
    edges = set()
    for i in range(TEST_START, n):
        cands, probs = ff_probs(prep, ff_model, i)
        order = np.argsort(-probs)
        best = float(probs[order[0]])
        p_root = float(probs[cands.index(i)])
        # edge probability table (for component maintenance)
        prob_of = {cands[k_]: float(probs[k_]) for k_ in range(len(cands))}

        # merge the already-decoded edges of message i-1 into the components (the edges with source=i-1 are known once it has been decoded)
        # -- implementation: when decoding i, first merge the edges with source == i-1 (if i-1 >= TEST_START)
        j_src = i - 1
        if j_src >= TEST_START:
            pass  # the edge was already merged when decoding j_src (see the union call below)

        parent = None
        if tau_root is not None and best < tau_root:
            parent = i  # root
        else:
            A = []
            stopped = False
            for k_ in range(len(cands)):
                j = cands[k_]
                if j == i:
                    continue
                pj = float(probs[k_])
                if pj >= tau_topic:  # stage 1: topic check
                    A.append(j)
                    f12 = np.array([thread_agg_features(
                        tr, j, i, pj, prep, best,
                        int(np.where(order == k_)[0][0]), len(cands), p_root)],
                        dtype=np.float32)
                    with torch.no_grad():
                        pc = float(torch.sigmoid(head(torch.from_numpy(f12).unsqueeze(0))).item())
                    if pc >= tau_complete:  # stage 2: completeness converged -> stop
                        stopped = True
                        break
                # different topic -> keep backtracking (by design: do not stop)
            if A:
                parent = max(A, key=lambda j: prob_of[j])
            else:
                parent = i
        if parent != i:
            edges.add((i, parent))
            tr.union(i, parent, prob_of.get(parent, 0.0))
    return edges, tr


def backtrack_decode_log(log, prep, ff_model, head, **kw):
    """Incrementally decodes a whole file in message order (edges merged into the components on the fly)"""
    n = len(log.messages)
    tr = ComponentTracker(n)
    for k, m in enumerate(log.messages):
        tr.add_node(k, prep.utt_vec(k), m.speaker)
    edges = set()
    for i in range(TEST_START, n):
        cands, probs = ff_probs(prep, ff_model, i)
        order = np.argsort(-probs)
        best = float(probs[order[0]])
        p_root = float(probs[cands.index(i)])
        prob_of = {cands[k_]: float(probs[k_]) for k_ in range(len(cands))}

        parent = None
        if kw.get("tau_root") is not None and best < kw["tau_root"]:
            parent = i
        else:
            A = []
            for k_ in range(len(cands)):
                j = cands[k_]
                if j == i:
                    continue
                pj = float(probs[k_])
                if pj >= kw.get("tau_topic", 0.05):
                    A.append(j)
                    rank = int(np.where(order == k_)[0][0])
                    f12 = np.array([thread_agg_features(
                        tr, j, i, pj, prep, best, rank, len(cands), p_root)],
                        dtype=np.float32)
                    with torch.no_grad():
                        pc = float(torch.sigmoid(
                            head(torch.from_numpy(f12).unsqueeze(0))).item())
                    if pc >= kw.get("tau_complete", 0.5):
                        break
            if A:
                parent = max(A, key=lambda j: prob_of[j])
            else:
                parent = i
        if parent != i:
            edges.add((i, parent))
            tr.union(i, parent, prob_of.get(parent, 0.0))
    return edges


# ---------------- evaluation ----------------

def eval_split(logs, ff_model, head, emb, **kw):
    gold = gold_edge_sets(logs)
    auto = {}
    for log in logs:
        prep = LogPrep(log, emb=emb, cache_feats=True)
        edges = backtrack_decode_log(log, prep, ff_model, head, **kw)
        auto[short_name(log.name)] = edges
    return link_prf(gold, auto), auto


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "grid"], default="train")
    ap.add_argument("--max-files", type=int, default=None,
                    help="upper bound on the number of train files used to train CompleteHead")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tau-topic", type=float, default=0.05)
    ap.add_argument("--tau-complete", type=float, default=0.5)
    ap.add_argument("--tau-root", type=float, default=None)
    args = ap.parse_args()

    print("loading data ...")
    dev_logs = load_split("dev", DATA_ROOT)
    test_logs = load_split("test", DATA_ROOT)
    glove_path = os.path.join(DATA_ROOT, "glove-ubuntu.txt")
    emb = load_glove(glove_path)
    print("glove vocab:", len(emb))

    print("loading pretrained FF-277 ...")
    ff_model = FFNet(in_dim=277)
    ff_model.load_state_dict(torch.load(FF_CKPT, map_location="cpu"))
    ff_model.eval()

    head_ckpt = os.path.join(CKPT_DIR, "gts_complete_head.pt")

    if args.mode in ("train", "grid"):
        train_logs = load_split("train", DATA_ROOT)
        print("training CompleteHead (max_files={}, epochs={}) ...".format(
            args.max_files, args.epochs))
        head = train_complete_head(train_logs, ff_model, emb,
                                   epochs=args.epochs, max_files=args.max_files)
        torch.save(head.state_dict(), head_ckpt)
        print("head saved:", head_ckpt)

        if args.mode == "grid":
            best = None
            for tt in [0.02, 0.05, 0.10]:
                for tc in [0.3, 0.5, 0.7]:
                    for tr_ in [None, 0.3]:
                        kw = dict(tau_topic=tt, tau_complete=tc, tau_root=tr_)
                        m, _ = eval_split(dev_logs, ff_model, head, emb, **kw)
                        print("  tau_topic={} tau_complete={} tau_root={} -> dev F1={}".format(
                            tt, tc, tr_, m["F1"]), flush=True)
                        if best is None or m["F1"] > best[0]:
                            best = (m["F1"], dict(kw))
            print("best dev:", best)
            kw = best[1]
        else:
            kw = dict(tau_topic=args.tau_topic, tau_complete=args.tau_complete,
                      tau_root=args.tau_root)
    else:
        head = CompleteHead()
        head.load_state_dict(torch.load(head_ckpt, map_location="cpu"))
        head.eval()
        kw = dict(tau_topic=args.tau_topic, tau_complete=args.tau_complete,
                  tau_root=args.tau_root)

    print("evaluating with", kw)
    dev_m, _ = eval_split(dev_logs, ff_model, head, emb, **kw)
    test_m, test_auto = eval_split(test_logs, ff_model, head, emb, **kw)
    print("dev :", dev_m)
    print("test:", test_m)

    out = {"params": kw, "dev": dev_m, "test": test_m}
    with open(os.path.join(RESULTS_DIR, "gts_backtrack_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "gts_predictions_test.json"), "w") as f:
        json.dump({k: [list(e) for e in sorted(v)] for k, v in test_auto.items()}, f)
    print("results saved")


if __name__ == "__main__":
    main()
