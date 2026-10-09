#!/usr/bin/env python3
"""ECB v2: merged call + L6 multi-round backtracking iterated to a fixed point (a traceable iteration experiment).

v1 -> v2 improvements: (1) CallA/CallB merged into a single call (chain partition and backtrack
emitted together);
(2) after completion, unassigned messages are re-detected and backtracking is iterated until a
fixed point (at most 3 rounds);
(3) metrics are recorded for every round (per conversation, per round, as jsonl) to support
multi-round optimization comparisons.
Output: results/ecb_v2_iterations.jsonl + ecb_v2_iterations.xlsx + a v1/v2 comparison table.
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM
import ecb_deepseek_pilot as EP

W = FB.W
OUT_J = os.path.join(ROOT, "results", "ecb_v2_iterations.jsonl")
MAX_ROUNDS = 3

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def chat_merged(key, msgs):
    n = len(msgs)
    prompt = ("Below is a multi-party chat (index [i] per message). Do TWO things:\n"
              "1) Partition ALL messages into complete event chains (topic-coherent threads); "
              "every message must belong to exactly one chain.\n"
              '2) For messages whose preceding message is ambiguous or missing, provide a '
              'backtrack parent (the message they most likely respond to), or -1 if truly none.\n'
              'Output JSON: {"chains":[{"topic":str,"message_ids":[ints]}],'
              '"backtrack":[{"message_id":int,"parent":int}]}\n\n'
              "Chat:\n" + EP.conv_text(msgs))
    out = EP.chat(key, [{"role": "user", "content": prompt}], EP.dyn_len(n))
    if not out:
        return None, {}
    try:
        d = json.loads(out)
    except Exception:
        try:
            d = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except Exception:
            return None, {}
    return d.get("chains"), {int(x["message_id"]): int(x["parent"])
                             for x in d.get("backtrack", []) if 0 <= int(x["parent"]) < n}

def metrics_of(pred_par, gold_par):
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
            "component_recall": round(100 * len(gcomp & pcomp) / len(gcomp), 2) if gcomp else None,
            "ancestor_jaccard": round(100 * float(np.mean(anc)), 2) if anc else None}

def main():
    key = EP.load_key()
    t0 = time.time()
    hasher = TE.Hasher(); hasher.fit(None)
    msg_vec = lambda m: hasher.msg_vec(m["text"])
    loaders = {"discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack}
    v1 = {}
    if os.path.exists(os.path.join(ROOT, "results", "ecb_deepseek_pilot.jsonl")):
        for l in open(os.path.join(ROOT, "results", "ecb_deepseek_pilot.jsonl")):
            r = json.loads(l)
            v1[(r["dataset"], r["conv_id"])] = r
    rows = []
    for name, loader in loaders.items():
        splits = loader()
        test = splits["test"][:EP.PILOT_N[name]]
        log(f"=== {name}: {len(test)} convs ===")
        for conv in test:
            msgs = conv["msgs"]
            n = len(msgs)
            if n < 5:
                continue
            cid = conv.get("conv_id", "")
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            it_log = []
            chains, bt = chat_merged(key, msgs)
            pred = EP.chains_to_edges(msgs, chains, bt)
            m = metrics_of(pred, gold_par)
            it_log.append({"round": 0, "unassigned": n - len({x for ch in (chains or []) for x in ch.get("message_ids", [])}), **m})
            # L6 multi-round backtracking iteration: unassigned messages -> add parent edges -> recompute,
            # until a fixed point
            for rnd in range(1, MAX_ROUNDS):
                un = [i for i in range(n) if not pred[i] and i > 0]
                if not un:
                    break
                bt2 = EP.call_b(msgs, un, key)
                if not bt2:
                    break
                added = 0
                for mid, par in bt2.items():
                    if 0 <= par < n and par != mid and par not in pred[mid]:
                        pred[mid].append(par); added += 1
                if added == 0:
                    break
                m = metrics_of(pred, gold_par)
                it_log.append({"round": rnd, "added": added, **m})
                log(f"  {name}/{cid[:20]} round{rnd}: +{added} edges ancJ={m['ancestor_jaccard']}")
            base = v1.get((name, cid), {})
            rows.append({"dataset": name, "conv_id": cid, "n_msgs": n,
                         "n_chains": len(chains or []),
                         "iterations": len(it_log),
                         "round0_ancJ": it_log[0]["ancestor_jaccard"],
                         "final_path_F1": m["path_F1"],
                         "final_compR": m["component_recall"],
                         "final_ancJ": m["ancestor_jaccard"],
                         "v1_ancJ": base.get("ancestor_jaccard_mean"),
                         "v1_path_F1": base.get("path_F1"),
                         "gain_ancJ_vs_v1": (round(m["ancestor_jaccard"] - base["ancestor_jaccard_mean"], 2)
                                             if base.get("ancestor_jaccard_mean") is not None else None),
                         "per_round": it_log,
                         "raw_chains": chains})
            log(f"  {name}/{cid[:20]} iters={len(it_log)} final ancJ={m['ancestor_jaccard']} "
                f"(v1={base.get('ancestor_jaccard_mean')})")
        del splits
    with open(OUT_J, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("raw_chains", "per_round")} for r in rows])
    df.to_excel(os.path.join(ROOT, "results", "ecb_v2_iterations.xlsx"), index=False)
    df.to_csv(os.path.join(ROOT, "results", "ecb_v2_iterations.csv"), index=False)
    log(f"ALL_DONE rows={len(rows)} in {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
