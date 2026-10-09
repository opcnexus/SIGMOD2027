#!/usr/bin/env python3
"""Full benchmark: 6 datasets x 5 algorithms x 12 metrics (2026-09-26).

Datasets (unified conv format, all with parent edge labels; gold = human annotation,
platform silver):
  irc(gold) molweni(gold) discord(silver reply) slack(silver mention proxy)
  hn(silver thread) reddit(silver thread) -- HN/Reddit are the additionally
  crawled text-only versions

Algorithms:
  prev1     the previous message (heuristic baseline)
  saprev    the most recent message by the same author, falling back to prev1
            (heuristic)
  simmax    the largest char-3gram cosine inside the window (heuristic)
  PairMLP   6 edge features + vector dot product -> MLP (learned, lightweight)
  MHA-Net   char-3gram 64d + causal multi-head attention + BCE multi-label
            (learned, the main model)

Metrics (per dataset x algorithm, test split):
  link P/R/F1 (edge-level micro inside the window)
  root P/R/F1 (predicting the empty set for messages without a parent)
  cat F1: one-to-one / many-to-one / one-to-many / many-to-many
  unidentified% (gold edges missed) hallucination% (predicted edges wrong)
  multi-parent recall (recall over multi-parent edges)
  fragmentation / affinity (thread-level, simplified Kummerfeld definition)
  density ratio (density of the predicted graph / density of the gold graph)
  coverage% (share of multi-label predictions, i.e. not the argmax fallback)

Protocol notes:
  - candidates = forward window W=100; gold edges outside the window are not
    counted as FN (consistent with transfer_experiment)
  - a conv is truncated to its most recent 1200 messages (large HN/Reddit/Slack
    conversations); messages whose parent falls out of range after truncation are
    treated as having no parent
  - 80/20 conversation-level split (seed 42); IRC uses the official 117/21/35
Record: results/full_benchmark.json (completed_at at minute precision)
"""
import os, sys, json, time, glob, random
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import transfer_experiment as TE
import torch
import torch.nn as nn

RESULTS = os.path.join(ROOT, "results")
MAX_MSGS = 1200
W = TE.W
EPOCHS = 12
PATIENCE = 3

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def now_min():
    return time.strftime("%Y-%m-%d %H:%M")

# ---------------- Data ----------------
def truncate_conv(conv, max_msgs=MAX_MSGS):
    msgs = conv["msgs"]
    if len(msgs) <= max_msgs:
        return conv
    keep = msgs[-max_msgs:]
    ids = {m["id"] for m in keep}
    for m in keep:
        m["parents"] = [p for p in m["parents"] if p in ids]
    return {"conv_id": conv["conv_id"], "msgs": keep}

def load_slack(limit=150):
    convs = []
    for line in open(os.path.join(ROOT, "data", "candidates", "slack_silver_threads.jsonl")):
        d = json.loads(line)
        if sum(1 for m in d["msgs"] if m["parents"]) < 5:
            continue
        convs.append(truncate_conv(d))
        if len(convs) >= limit * 2:
            break
    return TE.split_convs(convs[:limit])

def load_hn(limit=100):
    convs = []
    for line in open(os.path.join(ROOT, "data", "candidates", "hn_threads_text.jsonl")):
        d = json.loads(line)
        msgs = [{"id": m["id"], "author": m.get("author"),
                 "parents": [m["parent"]] if m.get("parent") else [],
                 "ts": m.get("ts"), "text": m.get("text") or ""}
                for m in d["msgs"]]
        # sort by time (the HN tree traversal order is not chronological)
        msgs.sort(key=lambda m: m["ts"] or 0)
        if sum(1 for m in msgs if m["parents"]) < 10:
            continue
        convs.append(truncate_conv({"conv_id": d["conv_id"], "msgs": msgs}))
        if len(convs) >= limit * 2:
            break
    return TE.split_convs(convs[:limit])

def load_reddit(limit=200):
    convs = []
    for line in open(os.path.join(ROOT, "data", "candidates", "reddit_threads_text.jsonl")):
        d = json.loads(line)
        msgs = [{"id": m["id"], "author": m.get("author"),
                 "parents": [m["parent"]] if m.get("parent") else [],
                 "ts": m.get("ts"), "text": m.get("text") or ""}
                for m in d["msgs"]]
        msgs.sort(key=lambda m: m["ts"] or 0)
        if sum(1 for m in msgs if m["parents"]) < 5:
            continue
        convs.append(truncate_conv({"conv_id": d["conv_id"], "msgs": msgs}))
        if len(convs) >= limit * 2:
            break
    return TE.split_convs(convs[:limit])

# ---------------- Candidates and gold (position space) ----------------
def gold_positions(msgs):
    """id -> pos mapping plus, for each message, the set of gold parent positions
    (only the edges inside the window are kept and handled at evaluation time)"""
    id2pos = {m["id"]: i for i, m in enumerate(msgs)}
    gold = []
    for i, m in enumerate(msgs):
        gold.append({id2pos[p] for p in m["parents"] if p in id2pos})
    return id2pos, gold

def window_cands(i):
    return list(range(max(0, i - W), i))

# ---------------- Metrics ----------------
def cat_of(n_par, n_chi):
    mp = n_par > 1
    mc = n_chi > 1
    if not mp and not mc: return "1-1"
    if mp and not mc: return "many-1"
    if not mp and mc: return "1-many"
    return "many-many"

def compute_metrics(convs, make_predictor):
    """make_predictor(msgs, vecs) -> fn(i, cands) -> set of parent positions.
    vecs is the list of vectors precomputed for this conversation (avoids repeated
    O(n^2) encoding).
    Returns the dict of all metrics."""
    tp = fp = fn = 0
    root_tp = root_fp = root_fn = 0
    cat = defaultdict(lambda: [0, 0, 0])  # cat -> [tp, fp, fn]
    n_gold_edges = n_gold_mp_edges = n_mp_tp = 0
    n_multi_pred = n_scored = 0
    gold_denom = pred_denom = 0
    gold_comps_all, pred_comps_all = [], []
    for conv in convs:
        msgs = conv["msgs"]
        n = len(msgs)
        id2pos, gold = gold_positions(msgs)
        gold_children = defaultdict(int)
        for i, gs in enumerate(gold):
            for p in gs:
                gold_children[p] += 1
        vecs = None
        pred_fn, pred_children = None, None
        preds = {}
        for i in range(n):
            cands = window_cands(i)
            if cands and pred_fn is None:
                vecs = [msg_vec_global(m) for m in msgs]
                pred_fn = make_predictor(msgs, vecs)
                pred_children = defaultdict(int)
            if not cands:
                preds[i] = set()
                continue
            P = pred_fn(i, cands)
            preds[i] = P
            for p in P:
                pred_children[p] += 1
        # edge-level statistics
        for i in range(n):
            cset = set(window_cands(i))
            G = gold[i] & cset
            P = preds[i]
            n_scored += 1
            if P:
                n_multi_pred += 1
            gold_denom += len(G)
            pred_denom += len(P)
            tp += len(G & P); fn += len(G - P)
            fp_edges = P - G
            fp += len(fp_edges)
            if not G and not P: root_tp += 1
            elif not G and P:   root_fp += 1
            elif G and not P:   root_fn += 1
            for p in (G & P):
                c = cat_of(len(G), gold_children[p])
                cat[c][0] += 1
            for p in G - P:
                c = cat_of(len(G), gold_children[p])
                cat[c][2] += 1
            for p in fp_edges:
                c = cat_of(len(P), pred_children[p])
                cat[c][1] += 1
            if len(gold[i]) > 1:
                n_gold_mp_edges += len(G)
                n_mp_tp += len(G & P)
        def comps(edges):
            par = list(range(n))
            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]; x = par[x]
                return x
            for (a, b) in edges:
                ra, rb = find(a), find(b)
                if ra != rb: par[ra] = rb
            groups = defaultdict(set)
            for x in range(n):
                groups[find(x)].add(x)
            return list(groups.values())
        g_edges = [(i, p) for i in range(n) for p in gold[i] & set(window_cands(i))]
        p_edges = [(i, p) for i in range(n) for p in preds[i]]
        gold_comps_all.append(comps(g_edges))
        pred_comps_all.append(comps(p_edges))
    # aggregate
    def prf(t, f, n):
        p = t / max(1, t + f); r = t / max(1, t + n)
        return {"P": round(100 * p, 1), "R": round(100 * r, 1),
                "F1": round(100 * 2 * p * r / max(1e-9, p + r), 1)}
    link = prf(tp, fp, fn)
    root = prf(root_tp, root_fp, root_fn)
    cats = {c: prf(v[0], v[1], v[2]) for c, v in sorted(cat.items())}
    # fragmentation / affinity
    frags, affs = [], []
    for gcs, pcs in zip(gold_comps_all, pred_comps_all):
        for g in gcs:
            if len(g) < 2:
                continue
            gs = set(g)
            touches = [len(gs & p) for p in pcs if gs & p]
            frags.append(max(1, len(touches)))
            if touches:
                affs.append(max(touches) / len(gs))
    unid = round(100 * fn / max(1, tp + fn), 1)
    hallu = round(100 * fp / max(1, tp + fp), 1)
    return {
        "link": link, "root": root, "cats": cats,
        "unidentified_pct": unid, "hallucination_pct": hallu,
        "multi_parent_recall": round(100 * n_mp_tp / max(1, n_gold_mp_edges), 1)
                               if n_gold_mp_edges else None,
        "fragmentation": round(float(np.mean(frags)), 2) if frags else None,
        "affinity": round(float(np.mean(affs)), 3) if affs else None,
        "density_gold": round(gold_denom / max(1, n_scored), 3),
        "density_pred": round(pred_denom / max(1, n_scored), 3),
        "coverage_pct": round(100 * n_multi_pred / max(1, n_scored), 1),
        "n_scored_msgs": n_scored, "tp": tp, "fp": fp, "fn": fn,
    }

# ---------------- Algorithms ----------------
MSG_VEC_CACHE = {}

def cached_vec(msg_vec_of, text, _c=MSG_VEC_CACHE):
    v = _c.get(text)
    if v is None:
        v = msg_vec_of(text)
        if len(_c) < 2_000_000:
            _c[text] = v
    return v

def make_heuristics():
    def prev1(msgs, vecs):
        return lambda i, cands: {cands[-1]} if cands else set()
    def saprev(msgs, vecs):
        def f(i, cands):
            a = msgs[i]["author"]
            for c in reversed(cands):
                if a and msgs[c]["author"] == a:
                    return {c}
            return {cands[-1]} if cands else set()
        return f
    def simmax(msgs, vecs):
        vs = np.stack(vecs)
        def f(i, cands):
            if not cands:
                return set()
            s = vs[cands] @ vs[i]
            return {cands[int(np.argmax(s))]}
        return f
    return {"prev1": prev1, "saprev": saprev, "simmax": simmax}

class PairMLP(nn.Module):
    """7-dim input (6 edge features + vector dot product) -> 32 -> 1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, feats):
        return self.net(feats).squeeze(-1)

def pairmlp_feats(msgs, i, c, vi, vc):
    f = TE.pair_features(msgs, i, c, None)
    return f + [float(np.dot(vi, vc))]

def train_pairmlp(train_convs, dev_convs, tag):
    model = PairMLP().to(TE.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_f1, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(train_convs)
        tot = nb = 0
        for conv in train_convs:
            msgs = conv["msgs"]
            vecs = [msg_vec_global(m) for m in msgs]
            for i, m in enumerate(msgs):
                if not m["parents"]:
                    continue
                cands = window_cands(i)
                if not cands:
                    continue
                vi = vecs[i]
                X = torch.tensor([pairmlp_feats(msgs, i, c, vi, vecs[c]) for c in cands],
                                 dtype=torch.float32, device=TE.DEVICE)
                y = torch.tensor([1.0 if msgs[c]["id"] in set(m["parents"]) else 0.0
                                  for c in cands], dtype=torch.float32, device=TE.DEVICE)
                logits = model(X)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for conv in dev_convs:
                msgs = conv["msgs"]
                vecs = [msg_vec_global(m) for m in msgs]
                for i, m in enumerate(msgs):
                    cands = window_cands(i)
                    if not cands or not m["parents"]:
                        continue
                    vi = vecs[i]
                    X = torch.tensor([pairmlp_feats(msgs, i, c, vi, vecs[c]) for c in cands],
                                     dtype=torch.float32, device=TE.DEVICE)
                    s = torch.sigmoid(model(X)).cpu().numpy()
                    pred = {cands[j] for j, v in enumerate(s) if v >= 0.5}
                    if not pred:
                        pred = {cands[int(np.argmax(s))]}
                    gs = {c for c in cands if msgs[c]["id"] in set(m["parents"])}
                    tp += len(pred & gs); fp += len(pred - gs); fn += len(gs - pred)
        f1 = 200 * tp / max(1, 2 * tp + fp + fn)
        log(f"  {tag} ep{ep:02d} loss={tot/max(1,nb):.4f} dev F1={f1:.1f}")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"  {tag} early stop ep{ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

def mhanet_make(model):
    """Precompute the sequence per conversation and return a closure compatible
    with make_predictor"""
    def make(msgs, vecs):
        seq = torch.tensor(TE.node_seq(msgs, vecs), dtype=torch.float32,
                           device=TE.DEVICE).unsqueeze(0)
        model.eval()
        def f(i, cands):
            with torch.no_grad():
                pair_idx = torch.tensor([[i, c] for c in cands], dtype=torch.long,
                                        device=TE.DEVICE)
                pf = torch.tensor([TE.pair_features(msgs, i, c, None) for c in cands],
                                  dtype=torch.float32, device=TE.DEVICE)
                logits = model(seq, pair_idx, pf).cpu().numpy()
            pred = {cands[j] for j, v in enumerate(logits) if v >= 0.5}
            if not pred:
                pred = {cands[int(np.argmax(logits))]}
            return pred
        return f
    return make

def pairmlp_make(model):
    def make(msgs, vecs):
        model.eval()
        def f(i, cands):
            with torch.no_grad():
                vi = vecs[i]
                X = torch.tensor([pairmlp_feats(msgs, i, c, vi, vecs[c]) for c in cands],
                                 dtype=torch.float32, device=TE.DEVICE)
                s = torch.sigmoid(model(X)).cpu().numpy()
            pred = {cands[j] for j, v in enumerate(s) if v >= 0.5}
            if not pred:
                pred = {cands[int(np.argmax(s))]}
            return pred
        return f
    return make

# ---------------- Main flow ----------------
def main():
    import random
    t0 = time.time()
    out = {"completed_at": now_min(), "device": TE.DEVICE,
           "config": {"W": W, "max_msgs": MAX_MSGS, "epochs": EPOCHS,
                      "patience": PATIENCE, "seed": TE.SEED,
                      "protocol": "window-W candidates; edges outside window not counted; conv truncated to last 1200 msgs"},
           "algorithms": ["prev1", "saprev", "simmax", "PairMLP", "MHA-Net"],
           "metrics_desc": {
               "link": "edge-level micro P/R/F1 (window)",
               "root": "no-parent message detection",
               "cats": "edge category F1 by gold structure: 1-1/many-1/1-many/many-many",
               "unidentified_pct": "gold edges missed",
               "hallucination_pct": "predicted edges wrong",
               "multi_parent_recall": "recall on edges of multi-parent children",
               "fragmentation": "mean #pred components per gold thread (simplified Kummerfeld)",
               "affinity": "mean best pred-component overlap per gold thread",
               "density_gold/pred": "mean parents per scored message",
               "coverage_pct": "% messages with multi-label prediction (non-fallback)"},
           "datasets": {}, "results": {}}

    log("loading datasets ...")
    ds = {}
    ds["irc"] = TE.load_irc()
    ds["molweni"] = TE.load_molweni()
    ds["discord"] = TE.split_convs(TE.load_discord())
    ds["slack"] = load_slack()
    log("loading hn_text / reddit_text ...")
    ds["hn"] = load_hn()
    ds["reddit"] = load_reddit()

    global msg_vec_global
    hasher = TE.Hasher(); hasher.fit(None)
    msg_vec_global = lambda m: cached_vec(hasher.msg_vec, m["text"])

    for name, splits in ds.items():
        if not splits.get("test"):
            log(f"[skip] {name}: no test split")
            continue
        log(f"=== dataset {name} ===")
        out["datasets"][name] = {}
        for ph, convs in splits.items():
            n_msg = sum(len(c["msgs"]) for c in convs)
            n_edge = sum(len(m["parents"]) for c in convs for m in c["msgs"])
            out["datasets"][name][ph] = {"convs": len(convs), "msgs": n_msg, "gold_edges": n_edge}
            log(f"  {ph}: convs={len(convs)} msgs={n_msg} gold={n_edge}")

        test = splits["test"]; dev = splits["dev"]; train = splits["train"]
        res = {}

        # heuristics
        heur = make_heuristics()
        for aname, maker in heur.items():
            m = compute_metrics(test, maker)
            res[aname] = m
            log(f"  {aname}: link F1={m['link']['F1']} root F1={m['root']['F1']}")

        # PairMLP
        pmlp = train_pairmlp(train, dev, f"{name}-PairMLP")
        res["PairMLP"] = compute_metrics(test, pairmlp_make(pmlp))
        log(f"  PairMLP: link F1={res['PairMLP']['link']['F1']}")
        del pmlp
        if TE.DEVICE == "mps": torch.mps.empty_cache()

        # MHA-Net (reuses the training loop of transfer_experiment)
        mha = TE.train_model(train, dev, msg_vec_global, f"{name}-MHA")
        res["MHA-Net"] = compute_metrics(test, mhanet_make(mha))
        log(f"  MHA-Net: link F1={res['MHA-Net']['link']['F1']}")
        del mha
        if TE.DEVICE == "mps": torch.mps.empty_cache()

        out["results"][name] = res
        out["completed_at"] = now_min()
        out["runtime_seconds"] = round(time.time() - t0)
        json.dump(out, open(os.path.join(RESULTS, "full_benchmark.json"), "w"),
                  ensure_ascii=False, indent=1)

    out["runtime_seconds"] = round(time.time() - t0)
    out["completed_at"] = now_min()
    json.dump(out, open(os.path.join(RESULTS, "full_benchmark.json"), "w"),
              ensure_ascii=False, indent=1)
    log(f"saved -> results/full_benchmark.json")
    log(f"ALL_DONE in {out['runtime_seconds']}s")

if __name__ == "__main__":
    main()
