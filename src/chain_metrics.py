#!/usr/bin/env python3
"""Chain-level reconstruction metrics v2: recursively backtrack from 1-1 edge
predictions to build an event tree/graph and compare it against the gold.

Three metrics:
  M1 path F1 (event chain level): exact root->leaf path matching P/R/F1
  M2 thread full reconstruction rate (event tree level): the share of gold weakly
     connected components reproduced message by message exactly
  M3 ancestor closure backtracking accuracy (recursive backtracking reconstruction
     metric): for every message that has a gold parent, the Jaccard between the
     predicted and gold ancestor closures -- a standalone metric requested by the
     user, emitted into a separate results file
     results/backtrack_reconstruction.json + reports/backtrack_reconstruction.md

v2 fixes / enhancements (relative to the v1 version that crashed on its first run):
  - fixed misuse of the enumerate_paths return value ((paths, truncated) was
    treated as a set element -> unhashable)
  - --probe can now be given several times (by default all 5 algorithms,
    including MHA-Net)
  - merge-style resumable runs: read the existing chain_metrics.json and skip
    the (dataset, probe) pairs that are already done
  - flush to disk after every completed (dataset, probe); at the end, emit the
    standalone backtracking metrics files automatically
"""
import os, sys, json, time, argparse
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB

W = FB.W
RESULTS = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(RESULTS, "chain_metrics.json")
BT_PATH = os.path.join(RESULTS, "backtrack_reconstruction.json")
BT_MD = os.path.join(ROOT, "reports", "backtrack_reconstruction.md")
from chain_metrics_core import (  # metrics core (zero dependencies, the single implementation)
    PATH_CAP, METRIC_VERSION, build_forest, enumerate_paths, components,
    ancestor_closure, conv_chain_metrics,
)
ALL_PROBES = ["prev1", "saprev", "simmax", "PairMLP", "MHA-Net"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def eval_probe_on_conv(msgs, gold_par, pred_fn, msg_vec):
    """Single conversation x single probe -> path set, component set and list of
    ancestor Jaccard values."""
    n = len(msgs)
    pred_par = [[] for _ in range(n)]
    fn_f = None
    for i in range(n):
        cands = list(range(max(0, i - W), i))
        if not cands:
            continue
        if fn_f is None:
            vecs = [msg_vec(m) for m in msgs]   # simmax/PairMLP/MHA need the real vectors
            fn_f = pred_fn(msgs, vecs)
        for p in fn_f(i, cands):
            pred_par[i].append(p)
    gc, gr, _ = build_forest(gold_par)
    pc, pr_, _ = build_forest(pred_par)
    gpaths, gtr = enumerate_paths(gc, gr)
    ppaths, ptr = enumerate_paths(pc, pr_)
    gcomp, pcomp = components(gold_par), components(pred_par)
    anc = []
    for i in range(n):
        if gold_par[i]:
            ga = ancestor_closure(gold_par, i)
            pa = ancestor_closure(pred_par, i)
            if ga:
                anc.append(len(ga & pa) / len(ga | pa))
    return {"gpaths": gpaths, "ppaths": ppaths, "gtrunc": gtr, "ptrunc": ptr,
            "gcomp": gcomp, "pcomp": pcomp, "anc": anc}

def dataset_probe_eval(name, splits, pname, maker, msg_vec):
    test = splits["test"]
    gpaths_all, ppaths_all = set(), set()
    gcomp_all, pcomp_all = set(), set()
    anc_all, trunc = [], 0
    for conv in test:
        msgs = conv["msgs"]
        id2pos = {m["id"]: i for i, m in enumerate(msgs)}
        gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
        r = eval_probe_on_conv(msgs, gold_par, maker, msg_vec)
        gpaths_all |= set(r["gpaths"]); ppaths_all |= set(r["ppaths"])
        gcomp_all |= r["gcomp"]; pcomp_all |= r["pcomp"]
        anc_all += r["anc"]
        trunc += int(r["gtrunc"] or r["ptrunc"])
    inter = len(gpaths_all & ppaths_all)
    pp = inter / len(ppaths_all) if ppaths_all else None
    prr = inter / len(gpaths_all) if gpaths_all else None
    pf1 = (2 * pp * prr / max(1e-9, pp + prr)) if (pp is not None and prr is not None) else None
    return {
        "path_P": round(100 * pp, 2) if pp is not None else None,
        "path_R": round(100 * prr, 2) if prr is not None else None,
        "path_F1": round(100 * pf1, 2) if pf1 is not None else None,
        "n_gold_paths": len(gpaths_all), "n_pred_paths": len(ppaths_all),
        "component_recall": round(100 * len(gcomp_all & pcomp_all) / len(gcomp_all), 2) if gcomp_all else None,
        "component_precision": round(100 * len(gcomp_all & pcomp_all) / len(pcomp_all), 2) if pcomp_all else None,
        "n_gold_components": len(gcomp_all), "n_pred_components": len(pcomp_all),
        "ancestor_jaccard": round(100 * float(np.mean(anc_all)), 2) if anc_all else None,
        "n_anc_msgs": len(anc_all),
        "path_truncated_convs": trunc,
    }

def write_backtrack_standalone(out):
    """Standalone results file: the recursive backtracking reconstruction metric
    (M3) per dataset x algorithm + a Markdown table."""
    bt = {"metric": "backtrack_reconstruction_accuracy",
          "definition": "For every message with a gold parent, recursively backtrack from the predicted edges to the root and take the Jaccard between the predicted and gold ancestor closures x100 (macro-averaged over test conversations)",
          "completed_at": time.strftime("%Y-%m-%d %H:%M"),
          "datasets": {ds: {a: v.get("ancestor_jaccard") for a, v in probes.items()}
                       for ds, probes in out.get("datasets", {}).items()}}
    json.dump(bt, open(BT_PATH, "w"), ensure_ascii=False, indent=1)
    ds_list = sorted(bt["datasets"].keys())
    algs = ALL_PROBES
    lines = ["# Recursive Backtracking Reconstruction Metric (Backtrack Reconstruction Accuracy)", "",
             "> Definition: 1-1 edge predictions -> recursively backtrack to the root -> |predicted ancestor closure INTERSECT gold ancestor closure| / |union| x100 (macro-average).",
             "> Generated by: `src/chain_metrics.py` | Data: `results/backtrack_reconstruction.json` | Protocol: test split, gold = all parent edges of the truncated conv (a chain may span the window over multiple hops).", ""]
    hdr = "| Dataset | " + " | ".join(algs) + " |"
    lines += [hdr, "|" + "---|" * (len(algs) + 1)]
    for ds in ds_list:
        row = [ds] + [str(bt["datasets"][ds].get(a)) if bt["datasets"][ds].get(a) is not None else "—" for a in algs]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "For reference: the edge-level link F1 of the same probes is in `results/full_benchmark_v2/summary.json` (Table 3) -- the gap between the edge level and the chain / backtracking level quantifies the cascading failure where '1-1 links are accurate but whole chains are mismatched'."]
    open(BT_MD, "w").write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="append", default=None,
                    help="Algorithms to evaluate (may be repeated): prev1 saprev simmax PairMLP MHA-Net; all by default")
    args = ap.parse_args()
    want = args.probe if args.probe else list(ALL_PROBES)
    for p in want:
        assert p in ALL_PROBES, p

    # merge-style loading
    if os.path.exists(OUT_PATH):
        out = json.load(open(OUT_PATH))
        out.setdefault("datasets", {})
    else:
        out = {"completed_at": None, "metrics": [
            "M1 path_F1: event chain level, exact root->leaf path matching",
            "M2 component_recall: event tree level, message-by-message exact thread reconstruction rate",
            "M3 ancestor_jaccard: recursive backtracking reconstruction metric (standalone file backtrack_reconstruction.json)"],
            "protocol": "gold = all parent edges of the truncated conv (a chain may span the window over multiple hops); pred = decisions in a window of W=100; probes use a single seed 42",
            "datasets": {}}

    heur = FB.make_heuristics()
    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}

    hasher = TE.Hasher(); hasher.fit(None)
    FB.msg_vec_global = lambda m: hasher.msg_vec(m["text"])

    t0 = time.time()
    for name, loader in loaders.items():
        todo = [p for p in want if p not in out["datasets"].get(name, {})]
        if not todo:
            log(f"[skip] {name}: all requested probes done")
            continue
        log(f"=== dataset {name} === (todo probes: {todo})")
        splits = loader()
        for pname in todo:
            if pname == "PairMLP":
                log(f"  training PairMLP ...")
                model = FB.train_pairmlp(splits["train"], splits["dev"], f"{name}-PairMLP-chain")
                maker = FB.pairmlp_make(model)
            elif pname == "MHA-Net":
                log(f"  training MHA-Net ...")
                model = TE.train_model(splits["train"], splits["dev"],
                                       lambda m: hasher.msg_vec(m["text"]),
                                       f"{name}-MHA-chain")
                maker = FB.mhanet_make(model)
            else:
                maker = heur[pname]
            res = dataset_probe_eval(name, splits, pname, maker,
                                     lambda m: hasher.msg_vec(m["text"]))
            out["datasets"].setdefault(name, {})[pname] = res
            out["completed_at"] = time.strftime("%Y-%m-%d %H:%M")
            json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
            write_backtrack_standalone(out)
            log(f"  [ckpt] {name}/{pname}: pathF1={res['path_F1']} "
                f"compR={res['component_recall']} ancJ={res['ancestor_jaccard']}")
            if pname in ("PairMLP", "MHA-Net"):
                del model
                import gc; gc.collect()
                if TE.DEVICE == "mps":
                    import torch; torch.mps.empty_cache()
        del splits
        import gc; gc.collect()
    write_backtrack_standalone(out)
    log(f"ALL_DONE in {round(time.time()-t0)}s -> {OUT_PATH}")
    log(f"standalone -> {BT_PATH} + {BT_MD}")

if __name__ == "__main__":
    main()
