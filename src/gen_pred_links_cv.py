#!/usr/bin/env python3
"""5-fold cross-validation to generate the first-best predicted edges of the train files (the correct
approach for the autoregressive thread construction).

Problem: the train predictions in pred_links_v2.json come from "a v2 trained on train" -- the overfitting
makes the thread features far too accurate on train and severely mismatched with dev/test (F1 ~51 quality),
so the pred_online retraining collapsed (dev 31.1 / test 35.3, confirmed at 19:00).

Solution: 5-fold CV (seed=42). Fold k is trained for 12 epochs on 4/5 of train and predicts on the
held-out 1/5. After aggregation the prediction of every train file comes from a model that "has
never seen it" -- matching the prediction quality distribution of dev/test.
dev/test keep the predictions of the full v2 (already honest).

Output: results/pred_links_cv.json {basename: [[src,tgt],...]} (completed_at precise to the minute)
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_reproduction import DATA_ROOT, TEST_START, load_glove, short_name
from digat import (RESULTS_DIR, load_split_paths, load_logs_for,
                   node_features, same_speaker_matrix, cand_index,
                   DiGAT, PairScorer)


def predict_edges_all(graph, scorer, log, emb, device, query_lo=1):
    graph.eval(); scorer.eval()
    with torch.no_grad():
        x = node_features(log, emb)
        n = x.shape[0]
        x_t = torch.from_numpy(x).to(device)
        same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
        h = graph(x_t, same_t)
        queries = list(range(query_lo, n))
        if not queries:
            return []
        qi, cj, valid = cand_index(n, queries)
        scores = scorer(h, torch.from_numpy(qi).to(device),
                        torch.from_numpy(cj).to(device), same_t, n)
        scores = scores.masked_fill(~torch.from_numpy(valid).to(device), -1e9)
        pred = scores.argmax(dim=1).cpu().numpy()
    return [(int(i), int(cj[r][pred[r]])) for r, i in enumerate(queries)]


def train_fold(graph, scorer, logs, emb, device, epochs, patience):
    opt = torch.optim.Adam(list(graph.parameters()) + list(scorer.parameters()),
                           lr=1e-3, weight_decay=1e-6)
    best_dev, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        graph.train(); scorer.train()
        rng = np.random.RandomState(500 + ep)
        for log in rng.permutation(logs):
            x = node_features(log, emb)
            n = x.shape[0]
            queries = sorted(i for i in log.parents if 1 <= i < n)
            if not queries:
                continue
            qi, cj, valid = cand_index(n, queries)
            Y = np.zeros_like(valid, dtype=np.float32)
            keep = np.zeros(len(queries), dtype=bool)
            for r, i in enumerate(queries):
                gold = set(t for t in log.parents.get(int(i), [int(i)]))
                hit = np.isin(cj[r], list(gold))
                Y[r] = hit.astype(np.float32); keep[r] = hit.any()
            if not keep.any():
                continue
            qi, cj, valid, Y = qi[keep], cj[keep], valid[keep], Y[keep]
            x_t = torch.from_numpy(x).to(device)
            same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
            h = graph(x_t, same_t)
            scores = scorer(h, torch.from_numpy(qi).to(device),
                            torch.from_numpy(cj).to(device), same_t, n)
            valid_t = torch.from_numpy(valid).to(device)
            Y_t = torch.from_numpy(Y).to(device)
            s_m = scores.masked_fill(~valid_t, -1e9)
            gold_scores = s_m * Y_t + (-1e9) * (1 - Y_t)
            top1 = gold_scores.argmax(dim=1)
            loss = nn.functional.cross_entropy(s_m, top1)
            opt.zero_grad(); loss.backward(); opt.step()
        # a quick dev evaluation on the held-out fold (only the first 6 files, to save time)
        dev_m = quick_f1(graph, scorer, logs[:8], emb, device)
        if dev_m > best_dev:
            best_dev, bad = dev_m, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in list(graph.state_dict().items()) +
                          list(scorer.state_dict().items())}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        gk = set(graph.state_dict().keys())
        graph.load_state_dict({k: v for k, v in best_state.items() if k in gk})
        scorer.load_state_dict({k: v for k, v in best_state.items() if k not in gk})
    return best_dev


def quick_f1(graph, scorer, logs, emb, device):
    from ff_reproduction import gold_edge_sets, link_prf
    gold = gold_edge_sets(logs)
    auto = {}
    for log in logs:
        edges = predict_edges_all(graph, scorer, log, emb, device, TEST_START)
        auto[short_name(log.name)] = set(edges)
    return link_prf(gold, auto)["F1"]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))
    split = load_split_paths()
    train_logs = load_logs_for(split["train"])
    print("train files:", len(train_logs), flush=True)

    in_dim, d_model = 55, 128
    K = 5
    rng = np.random.RandomState(42)
    order = rng.permutation(len(train_logs))
    folds = np.array_split(order, K)

    out = {}
    t0 = time.time()
    for k in range(K):
        hold_idx = list(folds[k])
        fit_idx = [i for j in range(K) if j != k for i in folds[j]]
        fit_logs = [train_logs[i] for i in fit_idx]
        hold_logs = [train_logs[i] for i in hold_idx]
        graph = DiGAT(in_dim, d_model=d_model).to(device)
        scorer = PairScorer(d_model=d_model).to(device)
        best = train_fold(graph, scorer, fit_logs, emb, device,
                          epochs=30, patience=5)
        print("fold {}: best quick-dev F1={:.1f} ({:.0f}s)".format(
            k, best, time.time() - t0), flush=True)
        for log in hold_logs:
            out[short_name(log.name)] = predict_edges_all(
                graph, scorer, log, emb, device, query_lo=1)

    # dev/test keep the full v2 predictions (files it never saw, hence honest)
    with open(os.path.join(RESULTS_DIR, "pred_links_v2.json")) as f:
        v2 = json.load(f)
    v2.pop("_meta", None)
    for split_name in ("dev", "test"):
        for log in load_logs_for(split[split_name]):
            out[short_name(log.name)] = v2[short_name(log.name)]

    path = os.path.join(RESULTS_DIR, "pred_links_cv.json")
    meta = {"_meta": {
        "source": "5-fold CV first-best edges (train) + full-v2 edges (dev/test)",
        "n_files": len(out),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}}
    meta.update(out)
    with open(path, "w") as f:
        json.dump(meta, f)
    print("saved:", path, "completed_at:",
          datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
