"""GTS-DiGAT v3: solution-space-driven structural adaptation (multi-parent
multi-label + thread-level hierarchical attention).

Evidence from the solution-space analysis (results/data_space_analysis.json,
2026-09-25 16:43):
- Multi-parent messages genuinely exist: 2.67% of labelled messages (3.52% of
  non-root ones) have >=2 parents
  -> the single-label pointer softmax assumption is incomplete, so we switch to
     multi-label BCE + first-best CE.
- For single-parent cases 84% of the distances fall in 1-10 and only 1.7% exceed
  100 -> a candidate window of 101 covers 99.3% of gold parents.
- Literature basis (sources verified):
  - Zhu et al., AAAI 2020 (arXiv:1911.10666) Masked Hierarchical Transformer：
    predicts the parent from ancestor history (full ancestor aggregation) instead of
    an isolated pair; on Ubuntu IRC 97.40% are single-parent,
    which cross-validates against the 97.33% found in this analysis.
  - Tai et al., ACL 2015 (arXiv:1506.01726) TreeLSTM child-sum variant:
    a node representation is aggregated from several children -- this corresponds to
    the hierarchical aggregation of thread messages in the reply-to tree.
  - Velickovic et al., ICLR 2018 (arXiv:1710.10903) GAT: attention-based edge weights.

Architecture (two levels of attention):
  L1 message level: DiGAT time-forward DAG masked attention (kept from v2)
  L2 thread level: the reply-to thread representation of candidate j = child-sum
    mean pooling (gold threads under teacher forcing; causally safe because node indices
    inside a component are necessarily < the current query i)
  Scoring: score(i,j) = MLP([h_i, h_j, h_i*h_j, |h_i-h_j|, thread(j), scalars])

Pipeline (as required by the user: preprocessing -> featurization -> structure
optimization -> fine-tuning, keeping the intermediate data):
  reads the caches registered in data/artifacts/manifest.json (split_8020 / bert_feats /
  baselines),
  and writes every new result into results/ with a completed_at stamp (minute
  precision).
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ChatLog  # noqa: F401
from ff_reproduction import (DATA_ROOT, TEST_START, load_glove,
                             gold_edge_sets, link_prf, short_name)
from digat import (RESULTS_DIR, load_split_paths, load_logs_for,
                   node_features, same_speaker_matrix, cand_index,
                   DiGAT, GRAPH_WINDOW, CAND_WINDOW)

BERT_FEATS = "./data/bert_feats.npz"


def now_min():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------- Thread state (incremental union-find over the reply-to tree,
# ---------------- with child-sum pooling) ----------------

class ThreadState:
    """Incrementally maintain the connected components of the gold/predicted
    reply-to tree; the thread vector = mean of h inside the component.

    Causality: only edges with source < the current query are merged in, so node
    indices inside a component are necessarily < the query.
    """

    def __init__(self, h: np.ndarray):
        self.h = h                                  # (n, d) model output
        self.parent = list(range(len(h)))
        self.sum = h.copy()                         # root node -> sum of h over the component
        self.cnt = np.ones(len(h), dtype=np.int64)

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def add_edge(self, s, t):
        ra, rb = self.find(s), self.find(t)
        if ra == rb:
            return
        self.parent[rb] = ra
        self.sum[ra] = self.sum[ra] + self.sum[rb]
        self.cnt[ra] += self.cnt[rb]

    def thread_vec(self, j):
        """Returns (thread mean vector d,) (thread size,) (whether in a component,)"""
        r = self.find(j)
        if self.cnt[r] == 1 and r == j:
            return self.h[j], 1, 0                  # isolated node: use itself
        return self.sum[r] / self.cnt[r], int(self.cnt[r]), 1


def gold_edges_sorted(log: ChatLog):
    return sorted((s, t) for s, ts in log.parents.items() for t in ts if t != s)


# ---------------- Scorer v3 ----------------

class PairScorerV3(nn.Module):
    """Appends thread-level representations to the v2 pair features
    (hierarchical attention L2)."""

    N_PAIR = 5   # log distance, 1/distance, same speaker, query position, cand position
    N_THREAD = 3  # log thread size, whether in a component, log max link distance in thread

    def __init__(self, d_model=128, hidden=256, dropout=0.2):
        super().__init__()
        in_dim = d_model * 5 + PairScorerV3.N_PAIR + PairScorerV3.N_THREAD
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h, qi, cj, same_spk, n, th_vec, th_feat):
        """th_vec: (Q, W, d) candidate thread means; th_feat: (Q, W, 3) thread scalars"""
        hi = h[qi]                                        # (Q, d)
        hj = h[cj]                                        # (Q, W, d)
        W = cj.shape[1]
        hi_ = hi.unsqueeze(1).expand(-1, W, -1)
        scal = torch.stack([
            torch.log1p((qi.view(-1, 1) - cj).float().clamp(min=0)),
            1.0 / (qi.view(-1, 1) - cj).float().clamp(min=1),
            same_spk[qi.unsqueeze(1).expand(-1, W), cj].float(),
            (qi.float() / max(1, n - 1)).view(-1, 1).expand(-1, W),
            cj.float() / max(1, n - 1),
        ], dim=-1)                                        # (Q, W, 5)
        feats = torch.cat([hi_, hj, hi_ * hj, (hi_ - hj).abs(),
                           th_vec, th_feat, scal], dim=-1)
        return self.net(feats).squeeze(-1)                # (Q, W)


def build_thread_features(h, log, queries, use_thread=True,
                          thread_edges=None, thread_parents=None,
                          thread_drop=0.0):
    """Build thread-level features for every candidate of every query.

    thread_edges: edge list [(src,tgt)] used to build the threads -- a source that is
      consistent between training and inference (v2/CV predicted edges). When None it
      falls back to the gold edges (only for controlled reproduction; note the inference
      leakage).
    thread_parents: {tgt: [src,...]} precomputed parent table (used for the nearest
      distance scalar inside a thread).
    thread_drop: probability of randomly zeroing the thread features of valid columns
      during training (mitigates over-reliance on thread features).
    Returns th_vec (Q, W, d), th_feat (Q, W, 3)
    """
    n = h.shape[0]
    d = h.shape[1]
    qi, cj, _ = cand_index(n, queries)
    Q, W = qi.shape[0], cj.shape[1]
    th_vec = np.zeros((Q, W, d), dtype=np.float32)
    th_feat = np.zeros((Q, W, 3), dtype=np.float32)
    if not use_thread:
        return th_vec, th_feat
    ts = ThreadState(h)
    edges = (sorted((int(s), int(t)) for s, t in thread_edges)
             if thread_edges is not None else gold_edges_sorted(log))
    ptr = 0
    # parent list of each node (used for the max link distance scalar inside a thread)
    if thread_parents is None:
        parents_of = {}
        for s, tlist in log.parents.items():
            for t in tlist:
                if t != s:
                    parents_of.setdefault(t, []).append(s)
    else:
        parents_of = thread_parents
    for r, i in enumerate(qi):
        while ptr < len(edges) and edges[ptr][0] < i:
            s, t = edges[ptr]
            ts.add_edge(s, t)
            ptr += 1
        for w in range(W):
            j = int(cj[r, w])
            if j >= i:                     # invalid column (clip out of range), zero features
                th_feat[r, w, 2] = np.log1p(100)
                continue
            if thread_drop > 0 and np.random.rand() < thread_drop:
                continue                   # dropout: zero the features of a valid column
                                            #   (all zeros = missing signal)
            vec, cnt, in_comp = ts.thread_vec(j)
            th_vec[r, w] = vec
            th_feat[r, w, 0] = np.log1p(cnt)
            th_feat[r, w, 1] = in_comp
            # distance from i to the nearest member of the thread of candidate j
            # (hierarchical structure signal)
            ps = parents_of.get(j, [j])
            ds = [i - p_ for p_ in ps if p_ < i]
            th_feat[r, w, 2] = np.log1p(min(ds) if ds else 100)
    return th_vec, th_feat


# ---------------- Training / evaluation ----------------

def labels_matrix(log, queries):
    """(Q, W) multi-hot labels: 1 if j is in gold parents(i) (root self-links included)"""
    n = max(queries) + 1 if queries else 1
    qi, cj, valid = cand_index(n, queries)
    Y = np.zeros_like(valid, dtype=np.float32)
    dropped = []
    for r, i in enumerate(qi):
        gold = set(t for t in log.parents.get(int(i), [int(i)]))
        hit = np.isin(cj[r], list(gold))
        Y[r] = hit.astype(np.float32)
        if not hit.any():
            dropped.append(int(i))
    return Y, valid, dropped


def train_file_forward_v3(model, scorer, x_t, same_t, queries, log, device,
                          use_thread=True, thread_edges=None,
                          thread_parents=None, thread_drop=0.0):
    h = model(x_t, same_t)
    h_np = h.detach().cpu().numpy()
    n = x_t.shape[0]
    if not queries:
        return None, None, None, None
    qi, cj, valid = cand_index(n, queries)
    Y, valid, dropped = labels_matrix(log, queries)
    keep = ~np.array([q in set(dropped) for q in queries])
    if not keep.any():
        return None, None, None, None
    qi, cj, valid, Y = qi[keep], cj[keep], valid[keep], Y[keep]
    queries_k = [q for q, k in zip(queries, keep) if k]

    th_vec_np, th_feat_np = build_thread_features(
        h_np, log, queries_k, use_thread=use_thread,
        thread_edges=thread_edges, thread_parents=thread_parents,
        thread_drop=thread_drop)
    qi_t = torch.from_numpy(qi).to(device)
    cj_t = torch.from_numpy(cj).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    th_vec = torch.from_numpy(th_vec_np).to(device)
    th_feat = torch.from_numpy(th_feat_np).to(device)

    scores = scorer(h, qi_t, cj_t, same_t, n, th_vec, th_feat)
    return scores, Y_t, valid_t, (cj_t, valid_t)


def loss_v3(scores, Y, valid):
    """Multi-label BCE + first-best CE (first-best = the highest-scoring gold
    position at the moment)"""
    s_m = scores.masked_fill(~valid, -1e9)
    gold_scores = s_m * Y + (-1e9) * (1 - Y)
    top1 = gold_scores.argmax(dim=1)                     # (Q,)
    ce = nn.functional.cross_entropy(s_m, top1)
    bce = nn.functional.binary_cross_entropy_with_logits(scores, Y, reduction="none")
    bce = (bce * valid.float()).sum() / valid.float().sum().clamp(min=1)
    return ce + bce


def predict_edges_v3(model, scorer, log, emb, device, tau_link=0.5,
                     use_thread=True, bert_cache=None, thread_edges=None,
                     thread_parents=None, extra_x=None, mp_cap=3):
    """Thresholded multi-parent decoding: every candidate with sigmoid >= tau is
    predicted as a parent (capped at mp_cap, keeping the top ones by probability);
    an empty set falls back to argmax"""
    model.eval(); scorer.eval()
    with torch.no_grad():
        x = node_features(log, emb)
        if extra_x is not None:
            x = np.concatenate([x, extra_x], axis=1)
        if bert_cache is not None:
            x = np.concatenate([x, bert_cache[short_name(log.name)]], axis=1)
        n = x.shape[0]
        x_t = torch.from_numpy(x).to(device)
        same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
        h = model(x_t, same_t)
        h_np = h.cpu().numpy()
        queries = list(range(TEST_START, n))
        if not queries:
            return set()
        qi, cj, valid = cand_index(n, queries)
        th_vec_np, th_feat_np = build_thread_features(
            h_np, log, queries, use_thread=use_thread,
            thread_edges=thread_edges, thread_parents=thread_parents)
        qi_t = torch.from_numpy(qi).to(device)
        cj_t = torch.from_numpy(cj).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        th_vec = torch.from_numpy(th_vec_np).to(device)
        th_feat = torch.from_numpy(th_feat_np).to(device)
        scores = scorer(h, qi_t, cj_t, same_t, n, th_vec, th_feat)
        scores = scores.masked_fill(~valid_t, -1e9)
        probs = torch.sigmoid(scores).cpu().numpy()
    edges = set()
    for r, i in enumerate(queries):
        above = np.where(probs[r] >= tau_link)[0]
        if len(above) > mp_cap:  # multi-parent cap: keep the top mp_cap by probability
            above = above[np.argsort(-probs[r][above])][:mp_cap]
        if len(above) == 0:
            above = [int(np.argmax(probs[r]))]
        for w in above:
            edges.add((i, int(cj[r][w])))
    return edges


def evaluate_v3(model, scorer, logs, emb, device, tau_link, use_thread,
                bert_cache=None, pred_links=None, time_cache=None, mp_cap=3):
    gold = gold_edge_sets(logs)
    auto = {}
    for log in logs:
        te, tp = None, None
        if pred_links is not None:
            te = [tuple(e) for e in pred_links[short_name(log.name)]]
            tp = {}
            for s, t in te:
                if t != s:
                    tp.setdefault(t, []).append(s)
        ex = time_cache[short_name(log.name)] if time_cache is not None else None
        auto[short_name(log.name)] = predict_edges_v3(
            model, scorer, log, emb, device, tau_link, use_thread, bert_cache,
            thread_edges=te, thread_parents=tp, extra_x=ex, mp_cap=mp_cap)
    return link_prf(gold, auto), auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--tau-link", type=float, default=0.5)
    ap.add_argument("--no-thread", action="store_true", help="Ablation: remove the thread-level features")
    ap.add_argument("--no-mp", action="store_true", help="Ablation: single-label CE (v2 behaviour)")
    ap.add_argument("--thread-source", choices=["pred", "gold"], default="pred",
                    help="Thread edge source: pred = v2 predicted edges (honest, default); gold = gold edges (controlled comparison only, inference leakage)")
    ap.add_argument("--pred-cache",
                    default=os.path.join(RESULTS_DIR, "pred_links_v2.json"))
    ap.add_argument("--use-time", action="store_true",
                    help="Use time features (log gap + activity flag, data/time_feats.npz)")
    ap.add_argument("--mp-cap", type=int, default=3,
                    help="Cap on the number of predicted parents (top-k by prob)")
    ap.add_argument("--thread-drop", type=float, default=0.0,
                    help="Thread-feature dropout probability during training")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training and load the stored weights for a tau grid re-evaluation")
    ap.add_argument("--load-tag", default=None, help="Tag loaded in eval-only mode")
    ap.add_argument("--tag", default="digat_v3")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    split = load_split_paths()
    train_logs = load_logs_for(split["train"])
    dev_logs = load_logs_for(split["dev"])
    test_logs = load_logs_for(split["test"])
    print("split: train={} dev={} test={}".format(
        len(train_logs), len(dev_logs), len(test_logs)), flush=True)

    print("loading GloVe (cached artifact) ...", flush=True)
    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))

    bert_cache = None
    if os.path.exists(BERT_FEATS):
        z = np.load(BERT_FEATS)
        bert_cache = {k.split(".")[0]: z[k].astype(np.float32) for k in z.files}
        print("bert feats loaded (cached):", len(bert_cache), "files", flush=True)

    pred_links = None
    if args.thread_source == "pred":
        with open(args.pred_cache) as f:
            raw = json.load(f)
        raw.pop("_meta", None)
        pred_links = raw
        print("pred thread links loaded:", len(pred_links), "files",
              "(source: v2 first-best)", flush=True)

    time_cache = None
    if args.use_time:
        z = np.load("./data/time_feats.npz")
        time_cache = {k: z[k].astype(np.float32) for k in z.files}
        print("time feats loaded:", len(time_cache), "files", flush=True)

    def extra_of(log):
        return time_cache[short_name(log.name)] if time_cache is not None else None

    in_dim = 55 + 768 + (2 if args.use_time else 0)

    if args.eval_only:
        lt = args.load_tag or args.tag
        ck = torch.load(os.path.join(RESULTS_DIR, "digat_v3_{}.pt".format(lt)),
                        map_location="cpu")
        assert ck["in_dim"] == in_dim, \
            "in_dim mismatch: ckpt={} vs args={}".format(ck["in_dim"], in_dim)
        graph = DiGAT(in_dim, d_model=ck["d_model"]).to(device)
        graph.load_state_dict(ck["graph"])
        scorer = PairScorerV3(d_model=ck["d_model"]).to(device)
        scorer.load_state_dict(ck["scorer"])
        print("loaded checkpoint tag={}".format(lt), flush=True)
    else:
        graph = DiGAT(in_dim, d_model=args.d_model).to(device)
        scorer = PairScorerV3(d_model=args.d_model).to(device)
    opt = torch.optim.Adam(list(graph.parameters()) + list(scorer.parameters()),
                           lr=args.lr, weight_decay=1e-6)
    use_thread = not args.no_thread

    def thread_of(log):
        if pred_links is None:
            return None, None
        te = [tuple(e) for e in pred_links[short_name(log.name)]]
        tp = {}
        for s, t in te:
            if t != s:
                tp.setdefault(t, []).append(s)
        return te, tp

    best_dev, best_state, bad = -1.0, None, 0
    t0 = time.time()
    ep = -1
    if not args.eval_only:
        for ep in range(args.epochs):
            graph.train(); scorer.train()
            rng = np.random.RandomState(2000 + ep)
            order = rng.permutation(len(train_logs))
            tot_loss, n_b = 0.0, 0
            for bi in order:
                log = train_logs[bi]
                x = node_features(log, emb)
                ex = extra_of(log)
                if ex is not None:
                    x = np.concatenate([x, ex], axis=1)
                x = np.concatenate([x, bert_cache[short_name(log.name)]], axis=1)
                queries = sorted(i for i in log.parents if 1 <= i < x.shape[0])
                if not queries:
                    continue
                te, tp = thread_of(log)
                x_t = torch.from_numpy(x).to(device)
                same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
                scores, Y, valid, _ = train_file_forward_v3(
                    graph, scorer, x_t, same_t, queries, log, device,
                    use_thread=use_thread, thread_edges=te, thread_parents=tp,
                    thread_drop=args.thread_drop)
                if scores is None:
                    continue
                if args.no_mp:
                    # single-label CE (v2 behaviour): top1 = highest-scoring gold
                    s_m = scores.masked_fill(~valid, -1e9)
                    gold_scores = s_m * Y + (-1e9) * (1 - Y)
                    top1 = gold_scores.argmax(dim=1)
                    loss = nn.functional.cross_entropy(s_m, top1)
                else:
                    loss = loss_v3(scores, Y, valid)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += loss.item(); n_b += 1
            m_dev, _ = evaluate_v3(graph, scorer, dev_logs, emb, device,
                                   args.tau_link, use_thread, bert_cache,
                                   pred_links=pred_links,
                                   time_cache=time_cache, mp_cap=args.mp_cap)
            # the time features for dev evaluation are concatenated per file inside
            # evaluate_v3
            print("ep{:02d} loss={:.4f} dev F1={:.1f} ({:.0f}s)".format(
                ep, tot_loss / max(1, n_b), m_dev["F1"], time.time() - t0),
                flush=True)
            if m_dev["F1"] > best_dev:
                best_dev = m_dev["F1"]
                bad = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in list(graph.state_dict().items()) +
                              list(scorer.state_dict().items())}
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop at ep", ep, flush=True)
                    break

    if best_state:
        g_keys = set(graph.state_dict().keys())
        graph.load_state_dict({k: v for k, v in best_state.items() if k in g_keys})
        scorer.load_state_dict({k: v for k, v in best_state.items()
                                if k not in g_keys})

    # fine tau grid (selected on dev, then fixed for test)
    print("tau grid on dev ...", flush=True)
    best_tau, best_f1 = args.tau_link, -1
    grid = ([round(0.30 + 0.05 * k, 2) for k in range(11)]
            if not args.no_mp else [args.tau_link])
    for tau in grid:
        m, _ = evaluate_v3(graph, scorer, dev_logs, emb, device, tau,
                           use_thread, bert_cache, pred_links=pred_links,
                           time_cache=time_cache, mp_cap=args.mp_cap)
        print("  tau={} -> dev F1={}".format(tau, m["F1"]), flush=True)
        if m["F1"] > best_f1:
            best_f1, best_tau = m["F1"], tau

    m_test, auto = evaluate_v3(graph, scorer, test_logs, emb, device,
                               best_tau, use_thread, bert_cache,
                               pred_links=pred_links, time_cache=time_cache,
                               mp_cap=args.mp_cap)
    print("BEST dev F1={:.1f} (tau={})".format(best_f1, best_tau))
    print("test:", m_test)

    tag = args.tag
    out = {
        "model": "digat_v3", "tag": tag,
        "use_thread": use_thread, "multi_parent_head": not args.no_mp,
        "thread_source": args.thread_source, "use_time": args.use_time,
        "mp_cap": args.mp_cap, "thread_drop": args.thread_drop,
        "tau_link": best_tau, "best_dev_F1": best_f1, "test": m_test,
        "epochs_run": ep + 1, "train_seconds": round(time.time() - t0),
        "completed_at": now_min(),
        "protocol": "80/20 pooled split seed=42 (NOT comparable to official)",
        "leakage_note": ("thread edges from v2 predictions (honest)"
                         if args.thread_source == "pred"
                         else "WARNING: gold thread edges at inference = label leakage"),
    }
    with open(os.path.join(RESULTS_DIR, "digat_v3_{}.json".format(tag)), "w") as f:
        json.dump(out, f, indent=2)
    torch.save({"graph": graph.state_dict(), "scorer": scorer.state_dict(),
                "in_dim": in_dim, "d_model": args.d_model},
               os.path.join(RESULTS_DIR, "digat_v3_{}.pt".format(tag)))
    print("saved:", tag, "completed_at:", out["completed_at"])


if __name__ == "__main__":
    main()
