#!/usr/bin/env python3
"""ECB-C full-dataset test + per-message raw result export.

Extensions (in response to the user request): (1) ECB-C is run on all 6 datasets (for datasets without a
text-signal convention the deterministic channel is empty, reported faithfully as N/A);
(2) output **per-message** raw decision rows (not statistical aggregates):
  dataset, algorithm, conv_id, msg_pos, msg_id, author, has_mention,
  n_gold_parents, gold_parent_pos, n_pred_parents, pred_parent_pos,
  edge_hit (predicted parent INTERSECT gold parent), ancestor_jaccard (backtrack closure of that message)
Output: results/ecb_v3_all.{jsonl,csv,xlsx} (per_message rows + per_conv + summary)
"""
import os, sys, json, time, re
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM

W = FB.W
OUT_J = os.path.join(ROOT, "results", "ecb_v3_all.jsonl")
OUT_C = os.path.join(ROOT, "results", "ecb_v3_all.csv")
OUT_X = os.path.join(ROOT, "results", "ecb_v3_all.xlsx")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def strategy_c(msgs):
    """Deterministic mention resolution: a <@name> in message i -> the most recent message by author name before i."""
    n = len(msgs)
    ap = {}
    for i, m in enumerate(msgs):
        a = m.get("author")
        if a: ap.setdefault(a, []).append(i)
    import bisect
    pred = [[] for _ in range(n)]
    stat = {"mentions": 0, "resolved": 0}
    for i, m in enumerate(msgs):
        for name in re.findall(r"<@([^>]+)>", m.get("text") or ""):
            stat["mentions"] += 1
            pos = ap.get(name, [])
            j = bisect.bisect_left(pos, i) - 1
            if j >= 0:
                pred[i].append(pos[j]); stat["resolved"] += 1
    return pred, stat

def anc_j(gold_par, pred_par, i):
    if not gold_par[i]:
        return None
    ga = CM.ancestor_closure(gold_par, i)
    pa = CM.ancestor_closure(pred_par, i)
    return len(ga & pa) / len(ga | pa) if ga else None

def main():
    t0 = time.time()
    f = open(OUT_J, "a")
    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}
    done = set()
    if os.path.exists(OUT_J):
        for l in open(OUT_J):
            try:
                r = json.loads(l); done.add((r["dataset"], r["conv_id"]))
            except Exception: pass
    agg = defaultdict(lambda: [0, 0.0])
    per_conv = []
    for name, loader in loaders.items():
        log(f"=== {name} ===")
        splits = loader()
        for conv in splits["test"]:
            if (name, conv.get("conv_id", "")) in done:
                continue
            msgs = conv["msgs"]
            n = len(msgs)
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            pred_par, stat = strategy_c(msgs)
            pc, pr_, _ = CM.build_forest(pred_par)
            pp, _ = CM.enumerate_paths(pc, pr_)
            pcomp = CM.components(pred_par)
            for i, m in enumerate(msgs):
                cands = set(range(max(0, i - W), i))
                g = set(gold_par[i]) & cands
                pr = set(pred_par[i]) & cands
                row = {"dataset": name, "algorithm": "ECB-C", "conv_id": conv.get("conv_id", ""),
                       "msg_pos": i, "msg_id": m["id"], "author": m.get("author"),
                       "has_mention": bool("<@" in (m.get("text") or "")),
                       "n_gold_parents": len(gold_par[i]), "n_pred_parents": len(pred_par[i]),
                       "n_gold_in_window": len(g), "edge_hit": len(g & pr),
                       "gold_parent_pos": sorted(g), "pred_parent_pos": sorted(pr),
                       "ancestor_jaccard": round(anc_j(gold_par, pred_par, i), 4)
                                           if anc_j(gold_par, pred_par, i) is not None else None}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            # conversation level
            gc, gr, gl = CM.build_forest(gold_par)
            pc2, pr2, pl2 = CM.build_forest(pred_par)
            gp, _ = CM.enumerate_paths(gc, gr)
            ppd, _ = CM.enumerate_paths(pc2, pr2)
            gs, ps = set(gp), set(ppd)
            inter = len(gs & ps)
            f1 = 2 * (inter/len(ps)) * (inter/len(gs)) / max(1e-9, inter/len(ps) + inter/len(gs)) if ps and gs else None
            gcomp, pcomp = CM.components(gold_par), CM.components(pred_par)
            anc = [anc_j(gold_par, pred_par, i) for i in range(n) if gold_par[i] and anc_j(gold_par, pred_par, i) is not None]
            per_conv.append({"dataset": name, "conv_id": conv.get("conv_id", ""), "n_msgs": n,
                             "path_F1": round(100*f1, 2) if f1 else 0.0,
                             "component_recall": round(100*len(gcomp & pcomp)/len(gcomp), 2) if gcomp else None,
                             "ancestor_jaccard": round(100*np.mean(anc), 2) if anc else None,
                             **stat})
            a = agg[name]; a[0] += len(anc); a[1] += sum(anc)
            log(f"  {name}/{conv.get('conv_id','?')[:20]} n={n} ancJ={per_conv[-1]['ancestor_jaccard']}")
        del splits
    f.close()
    import pandas as pd
    msg_rows = [json.loads(l) for l in open(OUT_J)]
    dfm = pd.DataFrame(msg_rows)
    dfc = pd.DataFrame(per_conv)
    with pd.ExcelWriter(OUT_X, engine="openpyxl") as w:
        dfm.to_excel(w, sheet_name="per_message", index=False)
        dfc.to_excel(w, sheet_name="per_conv", index=False)
        (dfc.groupby("dataset")[["path_F1", "component_recall", "ancestor_jaccard"]]
         .mean().round(2).to_excel(w, sheet_name="summary"))
    dfm.to_csv(OUT_C, index=False)
    log(f"ALL_DONE msgs={len(msg_rows)} convs={len(per_conv)} in {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
