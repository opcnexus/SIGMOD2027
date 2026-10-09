# -*- coding: utf-8 -*-
"""Diagnostic 2: rescuable rate of CompleteHead (correct-parent label) when used as a reranker
- rescuable: top1 is wrong && the gold parent is in the top-10 && pc(gold) > pc(top1)
- AUC of pc (is-gold-parent label, restricted to the top-10 candidates)"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import DATA_ROOT, FFNet, load_glove, LogPrep
from gts_backtrack import (ComponentTracker, thread_agg_features,
                           CompleteHead, ff_probs, CKPT_DIR)

dev_logs = load_split("dev", DATA_ROOT)
emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))
ff = FFNet(in_dim=277)
ff.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ff_model_glove.pt"), map_location="cpu"))
ff.eval()
h1 = CompleteHead()
h1.load_state_dict(torch.load(os.path.join(CKPT_DIR, "gts_complete_head.pt"), map_location="cpu"))
h1.eval()

ys, ps = [], []
n_wrong, n_gold_in_topk, n_rescue = 0, 0, 0
for log in dev_logs:
    prep = LogPrep(log, emb=emb, cache_feats=False)
    n = len(log.messages)
    tr = ComponentTracker(n)
    for k, m in enumerate(log.messages):
        tr.add_node(k, None, m.speaker)
    edge_ptr = 0
    edges = sorted(((s, t) for s, ts in log.parents.items() for t in ts if t != s),
                   key=lambda x: x[0])
    for qi in sorted(set(k for k in log.parents if 1 <= k < n)):
        while edge_ptr < len(edges) and edges[edge_ptr][0] < qi:
            s, t = edges[edge_ptr]
            tr.union(s, t, 1.0)
            edge_ptr += 1
        gold_ps = set(t for t in log.parents.get(qi, []) if t != qi)
        if not gold_ps:
            continue
        cands, probs = ff_probs(prep, ff, qi)
        order = np.argsort(-probs)
        top1 = cands[int(order[0])]
        if top1 == qi:
            continue
        best = float(probs[order[0]])
        p_root = float(probs[cands.index(qi)])
        # top1 error accounting
        top1_wrong = top1 not in gold_ps
        if top1_wrong:
            n_wrong += 1
        pcs = {}
        for r_ in range(0, min(10, len(order))):
            k_ = int(order[r_])
            j = cands[k_]
            if j == qi:
                continue
            f12 = np.array(thread_agg_features(tr, j, qi, float(probs[k_]),
                                               prep, best, r_, len(cands), p_root),
                           dtype=np.float32)
            with torch.no_grad():
                pcs[j] = float(torch.sigmoid(h1(torch.from_numpy(f12).unsqueeze(0))).item())
            ys.append(1 if j in gold_ps else 0)
            ps.append(pcs[j])
        if top1_wrong and any(g in pcs for g in gold_ps):
            n_gold_in_topk += 1
            gp = max((g for g in gold_ps if g in pcs), key=lambda g: pcs[g])
            if pcs[gp] > pcs[top1]:
                n_rescue += 1

ys, ps = np.array(ys), np.array(ps)
pos, neg = ps[ys == 1], ps[ys == 0]
print("candidates:", len(ys), "pos:", int(ys.sum()))
print("pc | gold-parent:", np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 3))
print("pc | non-gold   :", np.round(np.quantile(neg, [0.1, 0.5, 0.9]), 3))
lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
sc = np.concatenate([pos, neg])
order = np.argsort(sc)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(sc) + 1)
auc = (ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg) + 1e-9)
print("AUC(pc):", round(float(auc), 3))
print("wrong top1:", n_wrong, "| gold in top10:", n_gold_in_topk,
      "| rescuable(pc gold>pc top1):", n_rescue)
