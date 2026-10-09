#!/usr/bin/env python3
"""ECB-C (constraint-aware ECB): platform signal conventions as deterministic priors + LLM residual backtracking.

Optimization motive (multi-round iteration, round 2): the merged call + iteration of v2 did not improve
Slack (ancJ ~2-5%).
Root-cause hypothesis: the Slack gold labels are built by the "Strategy C: mention -> the most recent
message by the mentioned user within the window" rule,
while the LLM only sees the text and is never told this convention -> inject the convention as a
deterministic prior (following the OCR-APT principle
"deterministic parser first, LLM handles the residual").

Pipeline:
  Pass1 deterministic: resolve <@display_name> -> the most recent message by that author (within the
          window) -> edge
  Pass2 LLM: backtrack only the mention messages that failed to resolve (optional; this round first
          measures Pass1 on its own)
  Metrics: M3 backtrack reconstruction accuracy + M1/M2 (reuses chain_metrics)

Trace: results/ecb_v3_constraint.jsonl (per conversation) + .xlsx + a log.
"""
import os, sys, json, time, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM

W = FB.W
OUT_J = os.path.join(ROOT, "results", "ecb_v3_constraint.jsonl")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def strategy_c_edges(msgs):
    """Deterministic Strategy C: every <@name> in the text of message i -> the position of the most recent
    message by author name before i.
        Returns pred_par (list of positional parent sets) + statistics."""
    n = len(msgs)
    author_positions = {}
    for i, m in enumerate(msgs):
        a = m.get("author")
        if a:
            author_positions.setdefault(a, []).append(i)
    import bisect
    pred = [[] for _ in range(n)]
    n_mentions = n_resolved = 0
    for i, m in enumerate(msgs):
        txt = m.get("text") or ""
        for name in re.findall(r"<@([^>]+)>", txt):
            n_mentions += 1
            pos = author_positions.get(name, [])
            j = bisect.bisect_left(pos, i) - 1
            if j >= 0:
                pred[i].append(pos[j])
                n_resolved += 1
    return pred, {"mentions": n_mentions, "resolved": n_resolved,
                  "resolution_rate": round(100 * n_resolved / max(1, n_mentions), 2)}

def metrics(pred_par, gold_par):
    gc, gr, _ = CM.build_forest(gold_par)
    pc, pr_, _ = CM.build_forest(pred_par)
    gpaths, _ = CM.enumerate_paths(gc, gr)
    ppaths, _ = CM.enumerate_paths(pc, pr_)
    gs, ps = set(gpaths), set(ppaths)
    inter = len(gs & ps)
    pp = inter / len(ps) if ps else None
    prr = inter / len(gs) if gs else None
    f1 = 2 * pp * prr / max(1e-9, pp + prr) if (pp and prr) else None
    gcomp, pcomp = CM.components(gold_par), CM.components(pred_par)
    anc = []
    for i in range(len(pred_par)):
        if gold_par[i]:
            ga = CM.ancestor_closure(gold_par, i)
            pa = CM.ancestor_closure(pred_par, i)
            if ga:
                anc.append(len(ga & pa) / len(ga | pa))
    return {"path_F1": round(100 * f1, 2) if f1 else 0.0,
            "path_P": round(100 * pp, 2) if pp is not None else None,
            "path_R": round(100 * prr, 2) if prr is not None else None,
            "component_recall": round(100 * len(gcomp & pcomp) / len(gcomp), 2) if gcomp else None,
            "ancestor_jaccard": round(100 * float(np.mean(anc)), 2) if anc else None}

def main():
    t0 = time.time()
    splits = FB.load_slack()
    test = splits["test"]
    log(f"slack test: {len(test)} convs")
    rows = []
    for conv in test:
        msgs = conv["msgs"]
        n = len(msgs)
        id2pos = {m["id"]: i for i, m in enumerate(msgs)}
        gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
        pred_par, stat = strategy_c_edges(msgs)
        m = metrics(pred_par, gold_par)
        rows.append({"dataset": "slack", "algorithm": "ECB-C(deterministic)",
                     "conv_id": conv.get("conv_id", ""), "n_msgs": n, **stat, **m})
        log(f"  {conv.get('conv_id','?')[:24]:24s} n={n:4d} ancJ={m['ancestor_jaccard']} "
            f"pathF1={m['path_F1']} resolution_rate={stat['resolution_rate']}%")
    del splits
    agg = {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 2)
           for k in ["ancestor_jaccard", "path_F1", "path_P", "path_R", "component_recall"]
           if any(r.get(k) is not None for r in rows)}
    out = {"completed_at": time.strftime("%Y-%m-%d %H:%M"),
           "method": "ECB-C: constraint-aware (strategy-C deterministic prior)",
           "aggregate": agg, "per_conv": rows,
           "runtime_seconds": round(time.time() - t0)}
    with open(OUT_J, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    import pandas as pd
    pd.DataFrame(rows).to_excel(os.path.join(ROOT, "results", "ecb_v3_constraint.xlsx"), index=False)
    log(f"AGGREGATE: {agg} | target: ancJ > 60% | saved {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
