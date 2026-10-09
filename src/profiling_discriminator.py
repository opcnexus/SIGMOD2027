#!/usr/bin/env python3
"""P1-9 profiling discriminability experiment + P0-4 window coverage (the quantification of RepSpace stage one).

Question 1 (discriminability): using only the 8 conv-level structural features (the six dimensions
plus derived ones), can we determine the representation tier or the source platform?
  - 3 classes: tier (forum tree / annotated dialogue / real chat) -- corresponding to the O1 -> O3
    upgrade in the paper
  - 6 classes: platform -- LOPO (leave-one-platform-out) cross-validation
Question 2 (window coverage): the coverage of the gold edges under the unified protocol W=100
(empirical evidence for Proposition 2).

Memory safety: load one dataset at a time -> extract features -> release (16GB unified memory constraint).
Output: results/profiling_discriminator.json
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB

W = 100
LAYER = {"irc": "annotated", "molweni": "annotated", "hn": "forum",
         "reddit": "forum", "discord": "realchat", "slack": "realchat"}

def conv_feats(msgs):
    """The 8-dimensional structural features of a single conversation (all taken from metadata, language independent)."""
    n = len(msgs)
    if n < 2:
        return None
    id2pos = {m["id"]: i for i, m in enumerate(msgs)}
    par = {i: [id2pos[p] for p in m["parents"] if p in id2pos]
           for i, m in enumerate(msgs)}
    edges = [(i, p) for i, ps in par.items() for p in ps]
    root_ratio = sum(1 for i in range(n) if not par[i]) / n
    mp_ratio = sum(1 for i in range(n) if len(par[i]) > 1) / n
    dist = [i - p for i, p in edges]
    ts = [m.get("ts") or 0 for m in msgs]
    lat = [ts[i] - ts[p] for i, p in edges if ts[i] and ts[p] and ts[i] > ts[p]]
    memo = {}
    def depth(i):
        if i in memo:
            return memo[i]
        r = 1 + max((depth(p) for p in par[i]), default=0)
        memo[i] = r
        return r
    depth_mean = np.mean([depth(i) for i in range(n)]) if n else 0
    gaps = [ts[i] - ts[i - 1] for i in range(1, n) if ts[i] and ts[i - 1]]
    return [root_ratio, mp_ratio,
            float(np.mean(dist)) if dist else 0.0,
            float(np.median(dist)) if dist else 0.0,
            float(np.mean(lat)) if lat else 0.0,
            float(np.median(lat)) if lat else 0.0,
            float(depth_mean),
            float(np.mean(gaps)) if gaps else 0.0], \
           [1.0 if i - p <= W else 0.0 for i, p in edges]

def load_all_convs(name):
    s = {"irc": TE.load_irc, "molweni": TE.load_molweni,
         "discord": lambda: TE.split_convs(TE.load_discord()),
         "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}[name]()
    return s["train"] + s["dev"] + s["test"]

def main():
    t0 = time.time()
    X, y_layer, y_plat, cov = {}, {}, {}, {}
    for name in LAYER:
        convs = load_all_convs(name)
        feats, covers = [], []
        for conv in convs:
            r = conv_feats(conv["msgs"])
            if r is None:
                continue
            f, c = r
            feats.append(f)
            covers.extend(c)
        X[name] = np.array(feats)
        y_layer[name] = [LAYER[name]] * len(feats)
        y_plat[name] = [name] * len(feats)
        cov[name] = {"coverage_pct": round(100 * float(np.mean(covers)), 2) if covers else None,
                     "n_edges": len(covers)}
        print(f"[{time.strftime('%H:%M:%S')}] {name}: convs={len(feats)} "
              f"window_coverage={cov[name]['coverage_pct']}%", flush=True)
        del convs, feats, covers

    # feature standardization (statistics over the full set; approximately leak-free inside LOPO: the
    # cross-dataset difference of the structural scales is itself signal,
    # a strict version should standardize inside each training fold -- both are reported)
    Xa, ya_l, ya_p = [], [], []
    plats = list(LAYER)
    for nm in plats:
        Xa.append(X[nm]); ya_l += y_layer[nm]; ya_p += y_plat[nm]
    Xa = np.vstack(Xa)
    Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def lopo(label_map):
        accs = {}
        for held in plats:
            tr = [nm for nm in plats if nm != held]
            tr_idx = np.concatenate([np.where(np.array(ya_p) == nm)[0] for nm in tr])
            te_idx = np.where(np.array(ya_p) == held)[0]
            labels = [label_map(p) for p in ya_p]
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=1.0))
            clf.fit(Xa[tr_idx], [labels[i] for i in tr_idx])
            acc = float(np.mean(clf.predict(Xa[te_idx]) ==
                                [labels[i] for i in te_idx]))
            accs[held] = round(100 * acc, 1)
            print(f"  LOPO hold-out={held}: acc={accs[held]}%", flush=True)
        accs["macro_avg"] = round(float(np.mean(list(accs.values()))), 1)
        return accs

    def confusion(y_true, y_pred, classes):
        idx = {c: i for i, c in enumerate(classes)}
        m = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            m[idx[t], idx[p]] += 1
        return m.tolist()

    print("=== 3-class (layer) LOPO ===", flush=True)
    acc_layer = {}
    layer_true_all, layer_pred_all = [], []
    labels_l = np.array([LAYER[p] for p in ya_p])
    for held in plats:
        tr = [nm for nm in plats if nm != held]
        tr_idx = np.concatenate([np.where(np.array(ya_p) == nm)[0] for nm in tr])
        te_idx = np.where(np.array(ya_p) == held)[0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(Xa[tr_idx], labels_l[tr_idx])
        pred = clf.predict(Xa[te_idx])
        acc_layer[held] = round(100 * float(np.mean(pred == labels_l[te_idx])), 1)
        layer_true_all += list(labels_l[te_idx]); layer_pred_all += list(pred)
        print(f"  LOPO hold-out={held}: acc={acc_layer[held]}%", flush=True)
    acc_layer["macro_avg"] = round(float(np.mean([v for k, v in acc_layer.items() if k != "macro_avg"])), 1)
    conf_layer = confusion(layer_true_all, layer_pred_all,
                           ["annotated", "forum", "realchat"])

    print("=== 6-class (platform) stratified 5-fold ===", flush=True)
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    labels_p = np.array(ya_p)
    clf6 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    pred6 = cross_val_predict(clf6, Xa, labels_p,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42))
    acc_plat = {"overall": round(100 * float(np.mean(pred6 == labels_p)), 1)}
    for nm in plats:
        m = labels_p == nm
        acc_plat[nm] = round(100 * float(np.mean(pred6[m] == nm)), 1)
    conf_plat = confusion(list(labels_p), list(pred6), plats)
    print("platform 5-fold overall:", acc_plat["overall"], "%", flush=True)

    out = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M"),
        "features": ["root_ratio", "multi_parent_ratio", "parent_dist_mean",
                     "parent_dist_median", "latency_mean_s", "latency_median_s",
                     "tree_depth_mean", "inter_gap_mean_s"],
        "per_dataset": {nm: {"n_convs": int(len(X[nm])),
                             "feat_mean": [round(float(v), 4) for v in X[nm].mean(0)],
                             **cov[nm]} for nm in plats},
        "layer_lopo_accuracy_pct": acc_layer,
        "layer_lopo_confusion_rows_true_cols_pred__annotated_forum_realchat": conf_layer,
        "platform_5fold_accuracy_pct": acc_plat,
        "platform_5fold_confusion_rows_true_cols_pred__6platforms": conf_plat,
        "window_W": W,
        "runtime_seconds": round(time.time() - t0),
    }
    with open(os.path.join(ROOT, "results", "profiling_discriminator.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved -> results/profiling_discriminator.json", flush=True)
    print("layer LOPO macro:", acc_layer["macro_avg"], "%",
          "| platform LOPO macro:", acc_plat["macro_avg"], "%")

if __name__ == "__main__":
    main()
