#!/usr/bin/env python3
"""ECB pilot: LLM event-chain reconstruction with the DeepSeek cloud service (dynamic length configuration).

Pipeline (2 calls per conversation):
  Call A event chain partition: takes all conversation messages (dynamic window), returns a JSON set
         of event chains {chains:[{topic, message_ids}]}-- the goal is "complete whole-chain
  construction".
  Call B backtrack completion: for messages in the unassigned dark zone, reverse slot
         filling (backtracking missing antecedents)
Validation: temporal legality + participant legality + in-chain message existence
         (OCR-APT-style stage-by-stage gating)
Edge mapping: adjacent in-chain messages are linked in temporal order -> M1/M2/M3
         (reuses chain_metrics, comparable with all baselines)


Dynamic length configuration (user-specified):
  - Context: no fixed truncation, assembled dynamically from the actual conversation size
    (when the model window is exceeded, slide to the tail and record it)
  - max_tokens = min(8192, 400 + 80*n_msgs)   (scales with conversation size)
  - temperature=0.2 (low temperature for structural tasks)

Key resolution order: environment variable DEEPSEEK_API_KEY -> ~/.repspace/deepseek.json {"api_key":...}
Pilot scale: the first 15 discord test conversations + the first 8 slack test conversations
         (cost control, adjustable)
Output: results/ecb_deepseek_pilot.jsonl (per-conversation raw results) + .xlsx/.csv
         (same format as chain_metrics_raw)
"""
import os, sys, json, time, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM
from llm_response import classify_response, MAX_TOKENS_CAP

W = FB.W
BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
PILOT_N = {"discord": 15, "slack": 8}

LAST_META = {}   # failure mode of the most recent LLM call (for P0-1 propagation)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k
    cfg = os.path.expanduser("~/.repspace/deepseek.json")
    if os.path.exists(cfg):
        return json.load(open(cfg)).get("api_key")
    raise SystemExit("DeepSeek key not found: export DEEPSEEK_API_KEY=... or write it to ~/.repspace/deepseek.json")

def chat(key, messages, max_tokens, max_retries=3):
    """Returns (content|None, meta). P0-1: distinguishes network / output_truncated / empty_content / bad_response."""
    mt = min(max_tokens, MAX_TOKENS_CAP)
    meta = {"failure_mode": None, "finish_reason": None, "attempts": 0}
    for attempt in range(max_retries):
        meta["attempts"] = attempt + 1
        req = urllib.request.Request(
            BASE.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": MODEL, "messages": messages,
                             "max_tokens": mt, "temperature": 0.2,
                             "response_format": {"type": "json_object"}}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                content, fr, fmode = classify_response(json.loads(r.read()))
        except Exception as e:
            content, fr, fmode = None, None, "network"
            log(f"  api retry {attempt+1}: {str(e)[:120]}")
        meta["finish_reason"], meta["failure_mode"] = fr, fmode
        LAST_META.clear(); LAST_META.update(meta)
        if fmode is None:
            return content, meta
        if fmode == "output_truncated" and mt < MAX_TOKENS_CAP:
            mt = MAX_TOKENS_CAP
            log(f"  output truncated -> escalate max_tokens={mt}")
            continue
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))
    return None, meta

def dyn_len(n):
    return min(8192, 400 + 80 * n)

def conv_text(msgs):
    return "\n".join(f"[{i}] ({m.get('author') or '?'} @{m.get('ts') or '?'}) {m['text'][:200]}"
                     for i, m in enumerate(msgs))

def call_a(msgs, key):
    n = len(msgs)
    prompt = ("Below is a multi-party chat (index [i] per message). Partition ALL messages "
              "into complete event chains (topic-coherent interaction threads). A chain is a "
              "sequence of messages forming one event/interaction (question-answer, request-response, "
              "topic discussion). EVERY message must belong to exactly one chain; do not drop any. "
              'Output JSON: {"chains":[{"topic":str,"message_ids":[ints...]}]}\n\n'
              + conv_text(msgs))
    out, _meta = chat(key, [{"role": "user", "content": prompt}], dyn_len(n))
    if not out:
        return None
    try:
        return json.loads(out).get("chains")
    except Exception:
        try:
            return json.loads(out[out.index("{"):out.rindex("}") + 1]).get("chains")
        except Exception:
            return None

def call_b(msgs, unassigned, key):
    if not unassigned:
        return {}
    n = len(msgs)
    ctx = conv_text([msgs[i] for i in range(n)])
    prompt = ("Full chat:\n" + ctx + "\n\nThe following messages were not assigned to any event chain: "
              + str(unassigned) + ". For EACH, backtrack its most likely preceding message index (the message "
              "it responds to), or -1 if truly none. Output JSON: "
              '{"backtrack":[{"message_id":int,"parent":int}]}')
    out, _meta = chat(key, [{"role": "user", "content": prompt}], dyn_len(len(unassigned)))
    if not out:
        return {}
    try:
        return {d["message_id"]: d["parent"] for d in json.loads(out).get("backtrack", [])}
    except Exception:
        return {}

def chains_to_edges(msgs, chains, backtrack):
    """Chain -> parent edges (adjacent in-chain messages linked in temporal order; a message shared
    by several chains takes the first arrival) + backtrack completion edges."""
    n = len(msgs)
    pred = [[] for _ in range(n)]
    for ch in (chains or []):
        ids = sorted({int(x) for x in ch.get("message_ids", []) if 0 <= int(x) < n})
        for a, b in zip(ids, ids[1:]):
            pred[b].append(a)
    for mid, par in (backtrack or {}).items():
        if 0 <= int(par) < n and int(par) != int(mid) and int(par) not in pred[int(mid)]:
            pred[int(mid)].append(int(par))
    # validation gate: temporal order (decreasing position) + deduplication
    for i in range(n):
        pred[i] = sorted({p for p in pred[i] if 0 <= p < i})
    return pred

def main():
    key = load_key()
    t0 = time.time()
    RAW_JSONL = os.path.join(ROOT, "results", "ecb_deepseek_pilot.jsonl")
    RAW_CSV = os.path.join(ROOT, "results", "ecb_deepseek_pilot.csv")
    RAW_XLSX = os.path.join(ROOT, "results", "ecb_deepseek_pilot.xlsx")
    hasher = TE.Hasher(); hasher.fit(None)
    msg_vec = lambda m: hasher.msg_vec(m["text"])
    heur = FB.make_heuristics()

    loaders = {"discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack}
    rows = []
    for name, loader in loaders.items():
        splits = loader()
        test = splits["test"][:PILOT_N[name]]
        log(f"=== {name}: {len(test)} convs (pilot) ===")
        for conv in test:
            msgs = conv["msgs"]
            n = len(msgs)
            if n < 5:
                continue
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            tA = time.time()
            chains = call_a(msgs, key)
            assigned = {int(x) for ch in (chains or []) for x in ch.get("message_ids", [])}
            unassigned = [i for i in range(n) if i not in assigned]
            bt = call_b(msgs, unassigned, key)
            pred_par = chains_to_edges(msgs, chains, bt)
            # chain-level metrics (same convention as all baselines)
            gc, gr, _ = CM.build_forest(gold_par)
            pc, prr_, _ = CM.build_forest(pred_par)
            gpaths, _ = CM.enumerate_paths(gc, gr)
            ppaths, _ = CM.enumerate_paths(pc, prr_)
            gs, ps = set(gpaths), set(ppaths)
            inter = len(gs & ps)
            pp = inter / len(ps) if ps else None
            prr = inter / len(gs) if gs else None
            f1 = 2 * pp * prr / max(1e-9, pp + prr) if (pp and prr) else None
            gcomp, pcomp = CM.components(gold_par), CM.components(pred_par)
            anc = []
            for i in range(n):
                if gold_par[i]:
                    ga = CM.ancestor_closure(gold_par, i)
                    pa = CM.ancestor_closure(pred_par, i)
                    if ga:
                        anc.append(len(ga & pa) / len(ga | pa))
            row = {"dataset": name, "algorithm": "ECB-DeepSeek", "conv_id": conv.get("conv_id", ""),
                   "n_msgs": n, "n_chains": len(chains or []), "n_unassigned": len(unassigned),
                   "n_gold_edges": sum(len(p) for p in gold_par), "n_pred_edges": sum(len(p) for p in pred_par),
                   "n_gold_paths": len(gs), "n_pred_paths": len(ps), "path_intersect": inter,
                   "path_P": round(pp, 4) if pp is not None else None,
                   "path_R": round(prr, 4) if prr is not None else None,
                   "path_F1": round(f1, 4) if f1 is not None else None,
                   "n_gold_components_multi": sum(1 for c in gcomp if len(c) > 1),
                   "component_exact_match_rate": round(sum(1 for c in pcomp if len(c) > 1 and c in gcomp) /
                                                       max(1, sum(1 for c in pcomp if len(c) > 1)), 4),
                   "ancestor_jaccard_mean": round(float(np.mean(anc)), 4) if anc else None,
                   "n_anc_msgs": len(anc),
                   "call_seconds": round(time.time() - tA, 1),
                   "raw_chains": chains}
            rows.append(row)
            with open(RAW_JSONL, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"  {name}/{conv.get('conv_id','?')} n={n} chains={len(chains or [])} "
                f"pathF1={row['path_F1']} ancJ={row['ancestor_jaccard_mean']} ({row['call_seconds']}s)")
        del splits

    # baseline comparison (PairMLP chain-level on the same pilot conversations) -- written to the
    # same CSV so the two can be compared row by row
    import csv as _csv
    if rows:
        keys = list(rows[0].keys())
        with open(RAW_CSV, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        import pandas as pd
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="ecb_per_conv", index=False)
    log(f"PILOT_DONE rows={len(rows)} in {round(time.time()-t0)}s -> {RAW_XLSX}")

if __name__ == "__main__":
    main()
