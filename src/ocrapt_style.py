#!/usr/bin/env python3
"""OCR-APT-style algorithm -> conversation disentanglement datasets (a cross-domain transfer experiment, direction one).

Transfers the four structural components of OCR-APT (arXiv:2510.15188) to conversation data:
  S1 graph construction: message = event node, candidate edges within a window (same benchmark protocol)
  S2 behavioural scoring: PairMLP(seed42) edge probabilities (behavioural features, not brittle attributes)
  S3 bidirectional pruning: traverse in both directions from high-confidence anchor edges, keeping
     only the edges on causally coherent paths (temporal legality + participant continuity);
     isolated low-confidence edges are dropped (treated as chain roots)
  S4 stage-by-stage validation: each stage applies the temporal / participant gate before the next

         stage is entered
Output edges -> chain-level metrics M1/M2/M3 (reuses chain_metrics), compared with plain PairMLP.
The LLM reconstruction layer (OCR-APT stages 4-6) will be integrated once the LLM is selected; this
version is a structural-layer transfer.
Incremental write: results/ocrapt_style.json
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM

W = FB.W
OUT = os.path.join(ROOT, "results", "ocrapt_style.json")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    t0 = time.time()
    hasher = TE.Hasher(); hasher.fit(None)
    FB.msg_vec_global = lambda m: hasher.msg_vec(m["text"])
    msg_vec = lambda m: hasher.msg_vec(m["text"])

    if os.path.exists(OUT):
        out = json.load(open(OUT))
    else:
        out = {"completed_at": None,
               "method": "OCR-APT-style: PairMLP behavior scoring + bidirectional "
                         "causal pruning + stage-wise validation (LLM stage pending)",
               "thresholds": {"tau_hi": 0.70, "tau_lo": 0.45},
               "datasets": {}}
    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}

    for name, loader in loaders.items():
        if name in out["datasets"]:
            log(f"[skip] {name}")
            continue
        log(f"=== dataset {name} ===")
        splits = loader()
        log("  training PairMLP (behavior scorer, seed 42) ...")
        model = FB.train_pairmlp(splits["train"], splits["dev"], f"{name}-ocrapt")

        import torch
        def conv_pred_par(msgs):
            n = len(msgs)
            vecs = [msg_vec(m) for m in msgs]
            scores_all, cands_all = [], []
            with torch.no_grad():
                for i in range(n):
                    cands = list(range(max(0, i - W), i))
                    cands_all.append(cands)
                    if not cands:
                        scores_all.append(np.array([]))
                        continue
                    vi = vecs[i]
                    X = torch.tensor([FB.pairmlp_feats(msgs, i, c, vi, vecs[c]) for c in cands],
                                     dtype=torch.float32, device=TE.DEVICE)
                    scores_all.append(torch.sigmoid(model(X)).cpu().numpy())
            return scores_all, cands_all

        gpaths_all, ppaths_all = set(), set()
        gcomp_all, pcomp_all = set(), set()
        anc_all = []
        for conv in splits["test"]:
            msgs = conv["msgs"]
            n = len(msgs)
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            scores_all, cands_all = conv_pred_par(msgs)
            # S3/S4 pruning
            pred_par = ocrapt_from_scores(msgs, scores_all, cands_all)
            gc, gr, _ = CM.build_forest(gold_par)
            pc, prr_, _ = CM.build_forest(pred_par)
            gpaths, _ = CM.enumerate_paths(gc, gr)
            ppaths, _ = CM.enumerate_paths(pc, prr_)
            gpaths_all |= set(gpaths); ppaths_all |= set(ppaths)
            gcomp_all |= CM.components(gold_par); pcomp_all |= CM.components(pred_par)
            for i in range(n):
                if gold_par[i]:
                    ga = CM.ancestor_closure(gold_par, i)
                    pa = CM.ancestor_closure(pred_par, i)
                    if ga:
                        anc_all.append(len(ga & pa) / len(ga | pa))
        inter = len(gpaths_all & ppaths_all)
        pp = inter / len(ppaths_all) if ppaths_all else None
        prr = inter / len(gpaths_all) if gpaths_all else None
        pf1 = 2 * pp * prr / max(1e-9, pp + prr) if (pp is not None and prr is not None) else None
        out["datasets"][name] = {
            "path_F1": round(100 * pf1, 2) if pf1 is not None else None,
            "path_P": round(100 * pp, 2) if pp is not None else None,
            "path_R": round(100 * prr, 2) if prr is not None else None,
            "component_recall": round(100 * len(gcomp_all & pcomp_all) / len(gcomp_all), 2) if gcomp_all else None,
            "ancestor_jaccard": round(100 * float(np.mean(anc_all)), 2) if anc_all else None,
        }
        out["completed_at"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
        log(f"  [ckpt] {name}: pathF1={out['datasets'][name]['path_F1']} "
            f"compR={out['datasets'][name]['component_recall']} "
            f"ancJ={out['datasets'][name]['ancestor_jaccard']}")
        del splits, model
        import gc; gc.collect()
        if TE.DEVICE == "mps":
            import torch; torch.mps.empty_cache()
    log(f"ALL_DONE in {round(time.time()-t0)}s")

def ocrapt_from_scores(msgs, scores_all, cands_all, tau_hi=0.70, tau_lo=0.45):
    """A pure-function version of S2-S4 (the scores have already been computed)."""
    n = len(msgs)
    edges = {}
    anchors = []
    for i in range(n):
        c = cands_all[i]
        s = scores_all[i]
        if not c or s.size == 0:
            edges[i] = set()
            continue
        order = np.argsort(-s)
        kept, hi = [], False
        for j in order:
            p = c[j]
            if s[j] >= tau_hi:
                anchors.append((i, p)); kept.append(p); hi = True
            elif s[j] >= tau_lo:
                kept.append(p)
        edges[i] = set(kept) if kept else {c[int(np.argmax(s))]}
    if anchors:
        keep = set()
        for (ci, pi) in anchors:
            for seed, down in ((ci, True), (pi, False)):
                stack = [seed]
                seen = set()
                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue
                    seen.add(x); keep.add(x)
                    if down:
                        for y in range(n):
                            if x in edges.get(y, ()) and y not in seen:
                                stack.append(y)
                    else:
                        for px in edges.get(x, ()):
                            if px not in seen:
                                stack.append(px)
        for i in range(n):
            s = scores_all[i]
            if i not in keep and s.size and s.max() < tau_hi:
                edges[i] = set()   # isolated low-confidence -> chain root
    return [sorted(edges.get(i, ())) for i in range(n)]

if __name__ == "__main__":
    main()
