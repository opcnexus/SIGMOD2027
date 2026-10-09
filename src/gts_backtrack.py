# -*- coding: utf-8 -*-
"""GTS-Backtrack v1 (experimental implementation): Graph-semantic
Topic-completeness Backtracking

Literature positioning (see reports/experiment_report_gts_backtrack.md):
- The graph / structural method family has precedents: Kummerfeld et al.
  ACL 2019 (FF, P19-1374), Ma et al. ACL 2022 (Struct/r-GCN,
  2022.acl-long.23), Li et al. 2023 (heterogeneous-graph EGCN + easy-first
  decoding, arXiv:2306.03975).
- But there is no precedent for "explicitly training a topic-completeness
  model and controlling backtrack termination with a convergence criterion".
- This implementation = a modification on top of that family: a shared encoder
  (FF 277-dim features + pretrained weights)
  + a new CompleteHead (whether a candidate message is "converged / complete"
    once merged into the current topic thread)
  + two-stage backtracking decoding (Root gate -> window backtrack -> topic
    check -> completeness gate -> stop).

Evaluation: the official link-level protocol (dev/test from line>=1000, ordered
pairs P/R/F1).
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


# ---------------- Utility: union-find (with component aggregate stats) ------

class ComponentTracker:
    """Incremental union-find maintaining aggregate features per component
    (size / centroid / speakers / earliest member / max link probability)"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.earliest = list(range(n))          # earliest (smallest) message index in the component
        self.max_prob = [0.0] * n               # max parent-child link probability in the component
        self.centroid = [None] * n              # mean message vector of the component (100,)
        self.spk = [dict() for _ in range(n)]   # independent dict per node

    def add_node(self, i: int, vec: np.ndarray, speaker):
        self.centroid[i] = vec.copy() if vec is not None else None
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
        """a=child, b=parent; prob is the probability of that link"""
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
        ca, cb = self.centroid[ra], self.centroid[rb]
        if ca is not None and cb is not None:
            self.centroid[ra] = ca + cb
        else:
            self.centroid[ra] = ca if ca is not None else cb
        for k, v in self.spk[rb].items():
            self.spk[ra][k] = self.spk[ra].get(k, 0) + v
        self.spk[rb] = {}

    def stats(self, i: int):
        r = self.find(i)
        return r, self.size[r], self.earliest[r], self.max_prob[r], self.centroid[r], self.spk[r]


def gold_components(log):
    """Undirected connected components of the full gold graph (union-find).
    A self-link means root, so no edge is added.
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
    """Build the union-find from gold edges with source < upto only
    (simulates the state at decoding time)"""
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
    """Topic-completeness head: takes aggregated features of
    (candidate message, target thread) -> P(same thread, keep backtracking)"""

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


class RerankHead(nn.Module):
    """Pairwise reranking head: input [f(j), f(top1), f(j)-f(top1)] (36 dims)
    -> P(j is more likely than the FF top-1 to be the true parent of i)"""

    N_FEATS = CompleteHead.N_FEATS * 3

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(RerankHead.N_FEATS, hidden), nn.Softsign(),
            nn.Linear(hidden, hidden // 2), nn.Softsign(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CtxHead(nn.Module):
    """Pure context scoring head (no p_link/rank; forces the model to learn a
    signal complementary to FF)
    9-dim input -> P(j is the gold parent of i)"""

    N_FEATS = 9

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CtxHead.N_FEATS, hidden), nn.Softsign(),
            nn.Linear(hidden, hidden // 2), nn.Softsign(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def ctx_features(tr: ComponentTracker, j: int, i: int, prep: LogPrep):
    """Context features without p_link (9 dims):
    distance / same speaker / time gap / thread size / thread max link
    probability / centroid cohesion / same-speaker ratio / span /
    direct semantic cosine"""
    root, size, earliest, max_prob, centroid, spk = tr.stats(j)
    qv_i = prep.utt_vec(i)
    qv_j = prep.utt_vec(j)
    denom = np.linalg.norm(qv_i) * np.linalg.norm(qv_j) + 1e-9
    cos_direct = float(np.dot(qv_i, qv_j) / denom)
    if centroid is not None and np.linalg.norm(centroid) > 0:
        denom2 = np.linalg.norm(centroid) * np.linalg.norm(qv_i) + 1e-9
        cohesion = float(np.dot(qv_i, centroid) / denom2)
    else:
        cohesion = 0.0
    spk_i = prep.log.messages[i].speaker
    spk_total = sum(spk.values())
    same_ratio = spk.get(spk_i, 0) / max(1, spk_total)
    return [
        min(100, i - j) / 100.0,
        1.0 if prep.log.messages[j].speaker == spk_i else 0.0,
        min(120, get_time_diff(prep.info, i, j)) / 60.0,
        math.log1p(size),
        max_prob,
        cohesion,
        same_ratio,
        min(100, i - earliest) / 100.0,
        cos_direct,
    ]


def stream_ctx_instances(train_logs, ff_model, emb, max_files=None, topk=10):
    """Stream CtxHead training instances: keep every top-K candidate
    (label = whether it is a gold parent)"""
    n_files = len(train_logs) if max_files is None else min(max_files, len(train_logs))
    for li in range(n_files):
        log = train_logs[li]
        prep = LogPrep(log, emb=emb, cache_feats=False)
        n = len(log.messages)
        tr = ComponentTracker(n)
        for k, m in enumerate(log.messages):
            tr.add_node(k, None, m.speaker)
        edge_ptr = 0
        edges = sorted(((s, t) for s, ts in log.parents.items() for t in ts
                        if t != s), key=lambda x: x[0])
        queries = sorted(set(k for k in log.parents if 1 <= k < n))
        for qi in queries:
            while edge_ptr < len(edges) and edges[edge_ptr][0] < qi:
                s, t = edges[edge_ptr]
                if t < n:
                    tr.union(s, t, 1.0)
                edge_ptr += 1
            gold_ps = set(t for t in log.parents.get(qi, []) if t != qi)
            if not gold_ps:
                continue
            cands, probs = ff_probs(prep, ff_model, qi)
            order = np.argsort(-probs)
            for r_ in range(min(topk, len(order))):
                k_ = int(order[r_])
                j = cands[k_]
                if j == qi:
                    continue
                f9 = ctx_features(tr, j, qi, prep)
                yield (f9, 1.0 if j in gold_ps else 0.0)
        del prep, tr


def train_ctx_head(train_logs, ff_model, emb, epochs=4, max_files=None,
                   lr=1e-3, report_every=20000):
    head = CtxHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    # positive rate inside top-10 is about 12%; pos_weight=4 compensates
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
    t0 = time.time()
    for ep in range(epochs):
        ep_pos = ep_tot = 0
        buf_x, buf_y = [], []
        for f9, y in stream_ctx_instances(train_logs, ff_model, emb, max_files):
            buf_x.append(f9)
            buf_y.append(y)
            ep_tot += 1
            ep_pos += y
            if len(buf_x) >= 4096:
                _head_step(head, opt, bce, buf_x, buf_y, 0, report_every, t0)
                buf_x, buf_y = [], []
        if buf_x:
            _head_step(head, opt, bce, buf_x, buf_y, 0, report_every, t0)
        print("  ctx-head ep{}: {} cands, pos={:.1%} ({:.0f}s)".format(
            ep, ep_tot, ep_pos / max(1, ep_tot), time.time() - t0), flush=True)
    return head


def thread_agg_features(tr: ComponentTracker, j: int, i: int, p_link: float,
                        prep: LogPrep, best_prob: float, rank: int,
                        win: int, p_root_i: float):
    """Build the 12-dim input features of CompleteHead
    (the thread of j = the component of tr that contains j)"""
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


# ---------------- Batch FF probability computation ----------------

def ff_probs(prep: LogPrep, model: FFNet, i: int):
    cands = prep.candidates(i)
    feats = np.array([prep.features(i, c) for c in cands], dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats))
        probs = torch.softmax(logits, dim=0).numpy()
    return cands, probs


# ---------------- CompleteHead training data (streaming, teacher forcing) ---

def stream_complete_instances(train_logs, ff_model, emb, max_files=None):
    """Stream (feat12, label) training instances:
    - for every labelled query i, scan its window candidates j
    - label = 1 if j is one of the gold parents of i (a correct parent -> stop
      backtracking and pick it), otherwise 0
      (note: the label is "direct parent", not "same-thread member" -- the latter
      has too high a base rate to be rerankable)
    - thread features use gold edges with source<i (teacher forcing, simulating
      the state at decoding time)
    - negative subsampling: at most 2 per query (compensated by BCE pos_weight)"""
    n_files = len(train_logs) if max_files is None else min(max_files, len(train_logs))
    rng = np.random.RandomState(7)
    for li in range(n_files):
        log = train_logs[li]
        prep = LogPrep(log, emb=emb, cache_feats=False)
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
            gold_ps = set(t for t in log.parents.get(qi, []) if t != qi)
            pos, neg = [], []
            for k_, j in enumerate(cands):
                if j == qi:
                    continue
                f12 = thread_agg_features(tr, j, qi, float(probs[k_]), prep,
                                          best, rank_of[j], len(cands), p_root)
                if j in gold_ps:
                    pos.append(f12)
                else:
                    neg.append(f12)
            if pos:
                for f12 in pos:
                    yield (f12, 1.0)
            if neg:
                idx = rng.choice(len(neg), size=min(2, len(neg)), replace=False)
                for k_ in idx:
                    yield (neg[k_], 0.0)
        # cleanup: release at end of file
        del prep, tr


def train_complete_head(train_logs, ff_model, emb, epochs=2, max_files=None,
                        lr=1e-3, report_every=20000):
    head = CompleteHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    # 2 negatives per query + about 1 positive per query -> pos_weight~2 balance
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    step, t0 = 0, time.time()
    n_pos = n_tot = 0
    for ep in range(epochs):
        # re-stream one full pass per epoch (features computed on the fly)
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


def stream_rerank_instances(train_logs, ff_model, emb, max_files=None, topk=10):
    """Stream pairwise reranking training instances:
    - for every query i that has a non-self-link gold parent: pair the top-1 t*
      with each candidate j inside the top-K
    - f_j = thread_agg_features(j), f_t = thread_agg_features(t*)
    - label: 1 if t* is not a gold parent of i while j is; otherwise 0
    (teacher forcing: thread features use gold edges with source<i)"""
    n_files = len(train_logs) if max_files is None else min(max_files, len(train_logs))
    rng = np.random.RandomState(11)
    for li in range(n_files):
        log = train_logs[li]
        prep = LogPrep(log, emb=emb, cache_feats=False)
        n = len(log.messages)
        tr = ComponentTracker(n)
        for k, m in enumerate(log.messages):
            tr.add_node(k, None, m.speaker)
        edge_ptr = 0
        edges = sorted(((s, t) for s, ts in log.parents.items() for t in ts
                        if t != s), key=lambda x: x[0])
        queries = sorted(set(k for k in log.parents if 1 <= k < n))
        for qi in queries:
            while edge_ptr < len(edges) and edges[edge_ptr][0] < qi:
                s, t = edges[edge_ptr]
                if t < n:
                    tr.union(s, t, 1.0)
                edge_ptr += 1
            gold_ps = set(t for t in log.parents.get(qi, []) if t != qi)
            if not gold_ps:
                continue
            cands, probs = ff_probs(prep, ff_model, qi)
            order = np.argsort(-probs)
            best = float(probs[order[0]])
            p_root = float(probs[cands.index(qi)])
            top1 = cands[int(order[0])]
            if top1 == qi:
                continue  # queries that FF judges as root are not link-reranked
            f_t = thread_agg_features(tr, top1, qi, float(probs[order[0]]), prep,
                                      best, 0, len(cands), p_root)
            pool = []  # (j, f_j, is_gold)
            for r_ in range(1, min(topk, len(order))):
                k_ = int(order[r_])
                j = cands[k_]
                if j == qi:
                    continue
                f_j = thread_agg_features(tr, j, qi, float(probs[k_]), prep,
                                          best, r_, len(cands), p_root)
                pool.append((j, f_j, j in gold_ps))
            pos = [(j, f_j) for j, f_j, g in pool if g and top1 not in gold_ps]
            neg = [(j, f_j) for j, f_j, g in pool if not g]
            for j, f_j in pos:
                yield (np.concatenate([f_j, f_t, np.array(f_j) - np.array(f_t)]), 1.0)
            if neg:
                idx = rng.choice(len(neg), size=min(2, len(neg)), replace=False)
                for k_ in idx:
                    j, f_j = neg[k_]
                    yield (np.concatenate([f_j, f_t, np.array(f_j) - np.array(f_t)]), 0.0)
        del prep, tr


def train_rerank_head(train_logs, ff_model, emb, epochs=2, max_files=None,
                      lr=1e-3, report_every=20000):
    head = RerankHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(epochs):
        ep_pos = ep_tot = 0
        buf_x, buf_y = [], []
        for x36, y in stream_rerank_instances(train_logs, ff_model, emb, max_files):
            buf_x.append(x36)
            buf_y.append(y)
            ep_tot += 1
            ep_pos += y
            if len(buf_x) >= 4096:
                _head_step(head, opt, bce, buf_x, buf_y, 0, report_every, t0)
                buf_x, buf_y = [], []
        if buf_x:
            _head_step(head, opt, bce, buf_x, buf_y, 0, report_every, t0)
        print("  rerank-head ep{}: {} pairs, pos={:.1%} ({:.0f}s)".format(
            ep, ep_tot, ep_pos / max(1, ep_tot), time.time() - t0), flush=True)
    return head


def pair_features(f_j, f_t):
    return np.concatenate([f_j, f_t, np.array(f_j) - np.array(f_t)])


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

# ---------------- Two-stage backtracking decoding ----------------

def precompute_probs(log, prep: LogPrep, ff_model: FFNet):
    """Precompute FF probabilities for all evaluation messages in the file
    (the 277-dim feature computation dominates the cost, so do it only once)"""
    cache = {}
    for i in range(TEST_START, len(log.messages)):
        cache[i] = ff_probs(prep, ff_model, i)
    return cache


def backtrack_decode_log(log, prep, ff_model, head, prob_cache=None, **kw):
    """GTS-Backtrack v1.1 incremental decoding (easy-first):
    1) Root gate (optional): FF best probability < tau_root -> predict root
    2) Scan candidates j in descending probability order (easy-first):
       - topic check: P(i->j) < tau_topic -> treated as a different topic, skip
         (keep backtracking to further candidates)
       - completeness check: CompleteHead(i, thread(j)) >= tau_complete ->
         converged, stop backtracking
       - otherwise keep backtracking (move to the next candidate)
    3) The candidate where we stop becomes the parent; if nothing converges ->
       degrade to the FF top-1 (argmax)
    Each decoded edge is merged into the components immediately (online thread
    context, no future information leakage)"""
    n = len(log.messages)
    tr = ComponentTracker(n)
    for k, m in enumerate(log.messages):
        tr.add_node(k, prep.utt_vec(k), m.speaker)
    edges = set()
    tau_topic = kw.get("tau_topic", 0.02)
    tau_complete = kw.get("tau_complete", 0.5)
    tau_root = kw.get("tau_root")
    max_scan = kw.get("max_scan", 30)
    head2 = kw.get("head2")
    rerank_tau = kw.get("rerank_tau")
    rerank_topk = kw.get("rerank_topk", 10)
    head3 = kw.get("head3")
    fusion_lambda = kw.get("fusion_lambda")
    for i in range(TEST_START, n):
        if prob_cache is not None:
            cands, probs = prob_cache[i]
        else:
            cands, probs = ff_probs(prep, ff_model, i)
        order = np.argsort(-probs)
        best = float(probs[order[0]])
        p_root = float(probs[cands.index(i)])
        prob_of = {cands[k_]: float(probs[k_]) for k_ in range(len(cands))}

        parent = None
        if tau_root is not None and best < tau_root:
            parent = i
        elif cands[int(order[0])] == i:
            # FF top-1 is a self-link (root): keep FF's root decision, the head
            # only reranks links
            parent = i
        elif head3 is not None and fusion_lambda is not None:
            # context fusion ranking: score(j) = lambda * log p_link(j)
            #                            + logit(pc_ctx(j))
            # lambda -> infinity degenerates to FF277; a small lambda lets the
            # context head dominate
            win = len(cands)
            best_score, best_j = None, None
            for r_ in range(min(rerank_topk, len(order))):
                k_ = int(order[r_])
                j = cands[k_]
                if j == i:
                    continue
                pj = float(probs[k_])
                if pj < tau_topic:
                    break
                f9 = np.array(ctx_features(tr, j, i, prep), dtype=np.float32)
                with torch.no_grad():
                    z = float(head3(torch.from_numpy(f9).unsqueeze(0)).item())
                score = fusion_lambda * math.log(max(pj, 1e-9)) + z
                if best_score is None or score > best_score:
                    best_score, best_j = score, j
            parent = best_j if best_j is not None else cands[int(order[0])]
        elif head2 is not None and rerank_tau is not None:
            # pairwise reranking: compute P(j beats top-1) for the top-K candidates,
            # switch parent when it exceeds the threshold
            t1 = cands[int(order[0])]
            win = len(cands)
            f_t = np.array(thread_agg_features(
                tr, t1, i, float(probs[order[0]]), prep, best, 0, win, p_root),
                dtype=np.float32)
            best_j, best_p = None, rerank_tau
            for r_ in range(1, min(rerank_topk, len(order))):
                k_ = int(order[r_])
                j = cands[k_]
                if j == i:
                    continue
                pj = float(probs[k_])
                if pj < tau_topic:
                    break
                f_j = np.array(thread_agg_features(
                    tr, j, i, pj, prep, best, r_, win, p_root),
                    dtype=np.float32)
                x = torch.from_numpy(pair_features(f_j, f_t)).unsqueeze(0)
                with torch.no_grad():
                    ps = float(torch.sigmoid(head2(x)).item())
                if ps > best_p:
                    best_p, best_j = ps, j
            parent = best_j if best_j is not None else t1
        else:
            win = len(cands)
            for r_, k_ in enumerate(order[:max_scan]):
                j = cands[k_]
                if j == i:
                    continue
                pj = float(probs[k_])
                if pj < tau_topic:  # stage 1: topic check (different topic -> keep backtracking)
                    break
                # stage 2: completeness check
                f12 = np.array([thread_agg_features(
                    tr, j, i, pj, prep, best, r_, win, p_root)],
                    dtype=np.float32)
                with torch.no_grad():
                    pc = float(torch.sigmoid(
                        head(torch.from_numpy(f12).unsqueeze(0))).item())
                if pc >= tau_complete:  # converged -> stop backtracking
                    parent = j
                    break
            if parent is None:
                parent = cands[int(order[0])]  # fallback: FF top-1
        # official protocol: root predictions must also emit the self-link (i,i)
        # (the gold data contains self-links, and official FF emits them too)
        edges.add((i, parent))
        if parent != i:
            tr.union(i, parent, prob_of.get(parent, 0.0))
    return edges


# ---------------- Evaluation ----------------

def precompute_split(logs, ff_model, emb):
    """Precompute FF probabilities and LogPrep for a whole split
    (reused across multiple decoding runs)"""
    out = []
    for log in logs:
        prep = LogPrep(log, emb=emb, cache_feats=False)
        cache = precompute_probs(log, prep, ff_model)
        out.append((log, prep, cache))
        print("  probs done: {} ({} queries)".format(
            short_name(log.name), len(cache)), flush=True)
    return out


def eval_precomputed(precomp, head, **kw):
    gold = gold_edge_sets([pc[0] for pc in precomp])
    auto = {}
    for log, prep, cache in precomp:
        auto[short_name(log.name)] = backtrack_decode_log(
            log, prep, None, head, prob_cache=cache, **kw)
    return link_prf(gold, auto), auto


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "grid"], default="train")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap on the number of train files used to train CompleteHead")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tau-topic", type=float, default=0.05)
    ap.add_argument("--tau-complete", type=float, default=0.5)
    ap.add_argument("--tau-root", type=float, default=None)
    ap.add_argument("--rerank-tau", type=float, default=0.5)
    ap.add_argument("--rerank-topk", type=int, default=10)
    ap.add_argument("--fusion-lambda", type=float, default=2.0)
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
    ctx_ckpt = os.path.join(CKPT_DIR, "gts_ctx_head.pt")
    rerank_ckpt = os.path.join(CKPT_DIR, "gts_rerank_head.pt")

    if args.mode in ("train", "grid"):
        train_logs = load_split("train", DATA_ROOT)
        print("training CompleteHead (max_files={}, epochs={}) ...".format(
            args.max_files, args.epochs))
        head = train_complete_head(train_logs, ff_model, emb,
                                   epochs=args.epochs, max_files=args.max_files)
        torch.save(head.state_dict(), head_ckpt)
        print("head saved:", head_ckpt)

        print("training CtxHead (no p_link, complementary signal) ...")
        head3 = train_ctx_head(train_logs, ff_model, emb,
                               epochs=4, max_files=args.max_files)
        torch.save(head3.state_dict(), ctx_ckpt)
        print("ctx head saved:", ctx_ckpt)

        print("precomputing dev probs ...")
        dev_pre = precompute_split(dev_logs, ff_model, emb)

        if args.mode == "grid":
            best = None
            # lambda=inf ablation: pure FF277 (no head attached)
            m0, _ = eval_precomputed(dev_pre, head, tau_topic=0.01,
                                     tau_root=None, fusion_lambda=None,
                                     head3=None)
            print("  lambda=inf (FF277 ablation) -> dev F1={}".format(m0["F1"]), flush=True)
            for lam in [0.5, 1.0, 2.0, 4.0]:
                kw = dict(head3=head3, fusion_lambda=lam, rerank_topk=10,
                          tau_topic=0.01, tau_root=None)
                m, _ = eval_precomputed(dev_pre, head, **kw)
                print("  fusion_lambda={} -> dev F1={}".format(lam, m["F1"]), flush=True)
                if best is None or m["F1"] > best[0]:
                    best = (m["F1"], dict(kw))
            if best is None or m0["F1"] >= best[0]:
                print("FF277 ablation wins on dev; using pure FF277 params")
                kw = dict(head3=None, fusion_lambda=None, tau_topic=0.01, tau_root=None)
            else:
                print("best dev:", best[0],
                      {k: v for k, v in best[1].items() if k != "head3"})
                kw = best[1]
        else:
            kw = dict(head3=None, fusion_lambda=None,
                      tau_topic=args.tau_topic, tau_complete=args.tau_complete,
                      tau_root=args.tau_root)
    else:
        head = CompleteHead()
        head.load_state_dict(torch.load(head_ckpt, map_location="cpu"))
        head.eval()
        head3 = CtxHead()
        head3.load_state_dict(torch.load(ctx_ckpt, map_location="cpu"))
        head3.eval()
        kw = dict(head3=head3, fusion_lambda=args.fusion_lambda,
                  rerank_topk=args.rerank_topk,
                  tau_topic=args.tau_topic, tau_root=args.tau_root)
        dev_pre = None

    print("evaluating with", kw)
    if dev_pre is None:
        dev_pre = precompute_split(dev_logs, ff_model, emb)
    dev_m, _ = eval_precomputed(dev_pre, head, **kw)
    print("dev :", dev_m, flush=True)
    test_pre = precompute_split(test_logs, ff_model, emb)
    test_m, test_auto = eval_precomputed(test_pre, head, **kw)
    print("test:", test_m)

    out = {"params": {k: v for k, v in kw.items() if k not in ("head2", "head3")},
           "dev": dev_m, "test": test_m}
    with open(os.path.join(RESULTS_DIR, "gts_backtrack_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "gts_predictions_test.json"), "w") as f:
        json.dump({k: [list(e) for e in sorted(v)] for k, v in test_auto.items()}, f)
    print("results saved")


if __name__ == "__main__":
    main()
