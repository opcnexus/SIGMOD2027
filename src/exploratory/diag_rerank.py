# -*- coding: utf-8 -*-
"""Diagnose the discriminative power of RerankHead: P_swap distribution grouped by label on dev + AUC"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import DATA_ROOT, FFNet, load_glove, LogPrep
from gts_backtrack import (ComponentTracker, thread_agg_features, pair_features,
                           RerankHead, ff_probs, CKPT_DIR)

dev_logs = load_split("dev", DATA_ROOT)
emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))
ff = FFNet(in_dim=277)
ff.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ff_model_glove.pt"), map_location="cpu"))
ff.eval()
h2 = RerankHead()
h2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "gts_rerank_head.pt"), map_location="cpu"))
h2.eval()

ys, ps = [], []
for log in dev_logs[:3]:
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
        f_t = np.array(thread_agg_features(tr, top1, qi, float(probs[order[0]]),
                                           prep, best, 0, len(cands), p_root),
                       dtype=np.float32)
        for r_ in range(1, min(10, len(order))):
            k_ = int(order[r_])
            j = cands[k_]
            if j == qi:
                continue
            f_j = np.array(thread_agg_features(tr, j, qi, float(probs[k_]),
                                               prep, best, r_, len(cands), p_root),
                           dtype=np.float32)
            x = torch.from_numpy(pair_features(f_j, f_t).astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                p = float(torch.sigmoid(h2(x)).item())
            ys.append(1 if (j in gold_ps and top1 not in gold_ps) else 0)
            ps.append(p)

ys, ps = np.array(ys), np.array(ps)
pos, neg = ps[ys == 1], ps[ys == 0]
print("pairs:", len(ys), "pos:", int(ys.sum()))
print("P_swap | gold-better:", np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 3) if len(pos) else "none")
print("P_swap | otherwise  :", np.round(np.quantile(neg, [0.1, 0.5, 0.9]), 3))
lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
sc = np.concatenate([pos, neg])
order = np.argsort(sc)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(sc) + 1)
auc = (ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg) + 1e-9)
print("AUC:", round(float(auc), 3))
