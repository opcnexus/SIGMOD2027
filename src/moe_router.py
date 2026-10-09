#!/usr/bin/env python3
"""P1-7: the RepSpace-Route routing prototype (a feasibility check of the diagnose -> improve loop).

Idea: the RepSpace criterion shows that the regions where each probe is strong are complementary ->
train a lightweight router on conversation-level profiling features,
learning on dev which probe is best for a given conversation, then route predictions per conversation
on test and aggregate link-F1.
Controls: each single probe F1 + the routed F1 + oracle routing (upper bound).
Probe family: prev1 / saprev / simmax / PairMLP(seed42) (MHA differs from PairMLP by <1.3 F1, so it is omitted).
Routing features (conversation-level profiling, language independent): n_msgs / n_participants /
  root_ratio /
  mean_parent_dist / mean_gap_s / has_ts_ratio / n_gold_edges_window.

Memory safety: load and release one dataset at a time; run under nice.
Output: results/moe_router.json
"""
import os, sys, json, time
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import transfer_experiment as TE
import full_benchmark as FB

W = FB.W
RESULTS = os.path.join(ROOT, "results")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def conv_route_feats(msgs):
    n = len(msgs)
    id2pos = {m["id"]: i for i, m in enumerate(msgs)}
    parts = len({m["author"] for m in msgs if m["author"]})
    root_ratio = sum(1 for m in msgs if not m["parents"]) / max(1, n)
    dist = [i - id2pos[p] for i, m in enumerate(msgs) for p in m["parents"] if p in id2pos]
    ts = [m.get("ts") or 0 for m in msgs]
    gaps = [ts[i] - ts[i - 1] for i in range(1, n) if ts[i] and ts[i - 1]]
    has_ts = sum(1 for t in ts if t) / max(1, n)
    gold_w = sum(1 for i, m in enumerate(msgs)
                 for p in m["parents"] if p in id2pos and i - id2pos[p] <= W)
    return [n, parts, root_ratio,
            float(np.mean(dist)) if dist else 0.0,
            float(np.mean(gaps)) if gaps else 0.0,
            has_ts, gold_w]

def perconv_counts(convs, pred_fn, msg_vec):
    """tp/fp/fn of a single probe on a single conversation (within the window). vecs must be really
    computed (simmax/PairMLP depend on them)."""
    out = []
    for conv in convs:
        msgs = conv["msgs"]
        n = len(msgs)
        id2pos = {m["id"]: i for i, m in enumerate(msgs)}
        tp = fp = fn = 0
        fn_f = None
        for i, m in enumerate(msgs):
            cands = list(range(max(0, i - W), i))
            if not cands:
                continue
            if fn_f is None:
                vecs = [msg_vec(m2) for m2 in msgs]
                fn_f = pred_fn(msgs, vecs)
            gold = {id2pos[p] for p in m["parents"] if p in id2pos} & set(cands)
            pred = fn_f(i, cands)
            tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        out.append((tp, fp, fn))
    return out

def agg(counts):
    tp = sum(c[0] for c in counts); fp = sum(c[1] for c in counts); fn = sum(c[2] for c in counts)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"F1": round(100 * 2 * p * r / max(1e-9, p + r), 1),
            "P": round(100 * p, 1), "R": round(100 * r, 1)}

def main():
    t0 = time.time()
    out = {"completed_at": time.strftime("%Y-%m-%d %H:%M"),
           "probes": ["prev1", "saprev", "simmax", "PairMLP"], "datasets": {}}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # inject FB's global vector function (train_pairmlp depends on the module-level msg_vec_global internally)
    hasher = TE.Hasher(); hasher.fit(None)
    FB.msg_vec_global = lambda m: hasher.msg_vec(m["text"])

    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}

    def msg_vec(m):
        return hasher.msg_vec(m["text"])

    for name, loader in loaders.items():
        # resume support: skip datasets already written to disk
        try:
            prev = json.load(open(os.path.join(RESULTS, "moe_router.json")))
            if name in prev.get("datasets", {}):
                out["datasets"].update({name: prev["datasets"][name]})
                log(f"[skip] {name}: already in checkpoint")
                continue
        except Exception:
            pass
        log(f"=== dataset {name} ===")
        splits = loader()
        train, dev, test = splits["train"], splits["dev"], splits["test"]
        del splits
        # probes
        heur = FB.make_heuristics()
        probes = {"prev1": heur["prev1"], "saprev": heur["saprev"], "simmax": heur["simmax"]}
        log("  training PairMLP (seed 42) ...")
        model = FB.train_pairmlp(train, dev, f"{name}-PairMLP-route")
        probes["PairMLP"] = FB.pairmlp_make(model)

        names = list(probes)
        dev_counts = {p: perconv_counts(dev, probes[p], msg_vec) for p in names}
        test_counts = {p: perconv_counts(test, probes[p], msg_vec) for p in names}

        # routing label: the best probe per conversation on dev (by the tp-fp difference; all-zero
        # conversations are labelled prev1)
        Xd, yd = [], []
        for ci in range(len(dev)):
            msgs = dev[ci]["msgs"]
            scores = [dev_counts[p][ci][0] - dev_counts[p][ci][1] for p in names]
            if all(dev_counts[p][ci][0] == 0 for p in names):
                yd.append(0)
            else:
                yd.append(int(np.argmax(scores)))
            Xd.append(conv_route_feats(msgs))
        # degenerate handling: when the dev labels collapse to a single class (one probe is best
        # everywhere), routing degenerates to a constant choice
        import collections as _col
        cnt = _col.Counter(yd)
        Xte = [conv_route_feats(c["msgs"]) for c in test]
        if len(cnt) < 2:
            const = cnt.most_common(1)[0][0]
            route = np.full(len(test), const)
            router_info = "degenerate→" + names[const]
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(Xd, yd)
            route = clf.predict(Xte)
            router_info = "logreg"
        route = clf.predict(Xte)
        tp = fp = fn = 0
        oracle_tp = oracle_fp = oracle_fn = 0
        for ci in range(len(test)):
            p = names[route[ci]]
            tp += test_counts[p][ci][0]; fp += test_counts[p][ci][1]; fn += test_counts[p][ci][2]
            best = max(names, key=lambda q: test_counts[q][ci][0] - test_counts[q][ci][1])
            oracle_tp += test_counts[best][ci][0]; oracle_fp += test_counts[best][ci][1]; oracle_fn += test_counts[best][ci][2]
        def f1(tp, fp, fn):
            p_ = tp / max(1, tp + fp); r_ = tp / max(1, tp + fn)
            return round(100 * 2 * p_ * r_ / max(1e-9, p_ + r_), 1)
        single = {p: agg(test_counts[p]) for p in names}
        out["datasets"][name] = {
            "single": {p: v["F1"] for p, v in single.items()},
            "routed_F1": f1(tp, fp, fn),
            "oracle_F1": f1(oracle_tp, oracle_fp, oracle_fn),
            "route_distribution": {names[k]: int((route == k).sum()) for k in range(len(names))},
            "router": router_info,
        }
        log(f"  routed={out['datasets'][name]['routed_F1']} "
            f"oracle={out['datasets'][name]['oracle_F1']} single={out['datasets'][name]['single']}")
        # incremental write: a crash on one dataset does not lose the completed part
        out["completed_at"] = time.strftime("%Y-%m-%d %H:%M")
        with open(os.path.join(RESULTS, "moe_router.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        del train, dev, test, model, probes, dev_counts, test_counts
        import gc; gc.collect()
        if TE.DEVICE == "mps":
            import torch; torch.mps.empty_cache()

    # summary
    for name, d in out["datasets"].items():
        d["gain_vs_best_single"] = round(d["routed_F1"] - max(d["single"].values()), 1)
    with open(os.path.join(RESULTS, "moe_router.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"saved -> results/moe_router.json | done in {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
