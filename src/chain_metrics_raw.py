#!/usr/bin/env python3
"""Per-row raw data export of the chain-level metrics (Excel/CSV) -- for row-by-row manual inspection.

Each row = the raw chain-level reconstruction result of one (dataset, algorithm, conversation)
(not a statistical aggregate):
  dataset, algorithm, conv_id, n_msgs, n_gold_edges, n_pred_edges,
  n_gold_paths, n_pred_paths, path_intersect, path_P, path_R, path_F1,
  n_gold_components, n_pred_components, component_exact_match (0/1),
  ancestor_jaccard_mean, n_anc_msgs

Output:
  results/chain_metrics_raw.xlsx  (sheets: per_conv rows + a summary comparison)
  results/chain_metrics_raw.csv   (same per_conv layout, convenient for scripts)
  results/chain_metrics_raw.jsonl  (written incrementally, resumable after a crash)
Reuses every evaluation function of chain_metrics (single seed 42; same convention as chain_metrics.json).
Incremental: resume support keyed by (dataset, algorithm).
"""
import os, sys, json, time
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM

W = FB.W
RAW_XLSX = os.path.join(ROOT, "results", "chain_metrics_raw.xlsx")
RAW_CSV = os.path.join(ROOT, "results", "chain_metrics_raw.csv")
RAW_JSONL = os.path.join(ROOT, "results", "chain_metrics_raw.jsonl")
ALL_PROBES = ["prev1", "saprev", "simmax", "PairMLP", "MHA-Net"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def per_conv_row(dataset, alg, conv, gold_par, pred_par):
    msgs = conv["msgs"]
    n = len(msgs)
    gc, gr, _ = CM.build_forest(gold_par)
    pc, pr_, _ = CM.build_forest(pred_par)
    gpaths, _ = CM.enumerate_paths(gc, gr)
    ppaths, _ = CM.enumerate_paths(pc, pr_)
    gs, ps = set(gpaths), set(ppaths)
    inter = len(gs & ps)
    pp = inter / len(ps) if ps else None
    prr = inter / len(gs) if gs else None
    f1 = 2 * pp * prr / max(1e-9, pp + prr) if (pp is not None and prr is not None) else None
    gcomp, pcomp = CM.components(gold_par), CM.components(pred_par)
    # component exact match: whether each predicted component is exactly equal to some gold component
    # set (counted over threads with more than one message)
    exact = 0
    total = 0
    for c in pcomp:
        if len(c) > 1:
            total += 1
            if frozenset(c) in gcomp:
                exact += 1
    gold_multi = sum(1 for c in gcomp if len(c) > 1)
    ancs = []
    for i in range(n):
        if gold_par[i]:
            ga = CM.ancestor_closure(gold_par, i)
            pa = CM.ancestor_closure(pred_par, i)
            if ga:
                ancs.append(len(ga & pa) / len(ga | pa))
    return {
        "dataset": dataset, "algorithm": alg,
        "conv_id": conv.get("conv_id", ""),
        "n_msgs": n,
        "n_gold_edges": sum(len(p) for p in gold_par),
        "n_pred_edges": sum(len(p) for p in pred_par),
        "n_gold_paths": len(gs), "n_pred_paths": len(ps),
        "path_intersect": inter,
        "path_P": round(pp, 4) if pp is not None else None,
        "path_R": round(prr, 4) if prr is not None else None,
        "path_F1": round(f1, 4) if f1 is not None else None,
        "n_gold_components_multi": gold_multi,
        "n_pred_components_multi": total,
        "pred_components_exact_in_gold": exact,
        "component_exact_match_rate": round(exact / total, 4) if total else None,
        "ancestor_jaccard_mean": round(float(np.mean(ancs)), 4) if ancs else None,
        "n_anc_msgs": len(ancs),
    }

def main():
    t0 = time.time()
    hasher = TE.Hasher(); hasher.fit(None)
    FB.msg_vec_global = lambda m: hasher.msg_vec(m["text"])
    msg_vec = lambda m: hasher.msg_vec(m["text"])

    # resume point: read the (dataset, alg) set from the existing jsonl
    done = set()
    if os.path.exists(RAW_JSONL):
        for line in open(RAW_JSONL):
            try:
                r = json.loads(line)
                done.add((r["dataset"], r["algorithm"]))
            except Exception:
                pass
    f_jsonl = open(RAW_JSONL, "a")
    rows = []
    if os.path.exists(RAW_CSV):
        import csv
        rows = list(csv.DictReader(open(RAW_CSV)))
    else:
        rows = []

    heur = FB.make_heuristics()
    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}

    for name, loader in loaders.items():
        for alg in ALL_PROBES:
            if (name, alg) in done:
                log(f"[skip] {name}/{alg}")
                continue
            log(f"=== {name}/{alg} ===")
            splits = loader()
            if alg in ("PairMLP", "MHA-Net"):
                if alg == "PairMLP":
                    model = FB.train_pairmlp(splits["train"], splits["dev"], f"{name}-PairMLP-raw")
                    maker = FB.pairmlp_make(model)
                else:
                    model = TE.train_model(splits["train"], splits["dev"], msg_vec, f"{name}-MHA-raw")
                    maker = FB.mhanet_make(model)
            else:
                maker = heur[alg]
                model = None
            for conv in splits["test"]:
                msgs = conv["msgs"]
                id2pos = {m["id"]: i for i, m in enumerate(msgs)}
                gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
                r = CM.eval_probe_on_conv(msgs, gold_par, maker, msg_vec)
                # rebuild pred_par from the eval result (already computed inside eval; recomputed here for simplicity)
                pred_par = [[] for _ in range(len(msgs))]
                fn_f = None
                for i in range(len(msgs)):
                    cands = list(range(max(0, i - W), i))
                    if not cands:
                        continue
                    if fn_f is None:
                        vecs = [msg_vec(m) for m in msgs]
                        fn_f = maker(msgs, vecs)
                    for p in fn_f(i, cands):
                        pred_par[i].append(p)
                row = per_conv_row(name, alg, conv, gold_par, pred_par)
                rows.append(row)
                f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            # incremental CSV write
            import csv as _csv
            if rows:
                keys = list(rows[0].keys())
                with open(RAW_CSV, "w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
            log(f"  done {name}/{alg} rows={len(rows)}")
            del splits
            if model is not None:
                del model
            import gc; gc.collect()
            if TE.DEVICE == "mps":
                import torch; torch.mps.empty_cache()
    f_jsonl.close()

    # Excel output
    import pandas as pd
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="per_conv", index=False)
        # summary comparison sheet (edge-level F1 vs the chain-level mean)
        g = df.groupby(["dataset", "algorithm"]).agg(
            n_convs=("conv_id", "count"),
            path_F1_mean=("path_F1", "mean"),
            comp_match_mean=("component_exact_match_rate", "mean"),
            ancJ_mean=("ancestor_jaccard_mean", "mean")).reset_index().round(4)
        g.to_excel(w, sheet_name="summary_by_dataset_alg", index=False)
    log(f"saved {RAW_XLSX} rows={len(rows)} | done in {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
