#!/usr/bin/env python3
"""ECB-DeepSeek full-dataset validation + full LLM call log (kept as experimental evidence).

One evidence log record per LLM call (results/llm_call_log.jsonl):
  ts / endpoint / model / purpose / conv_id / prompt_chars / est_prompt_tokens /
  max_tokens / latency_s / http_status / response (full raw text) / usage
Conversation-level results (including the raw chains of every round) are written incrementally to
results/ecb_deepseek_all.{jsonl,xlsx}, with resume support.
Dynamic length: max_tokens=min(8192,400+80n); the context is trimmed to the tail by estimated token
count and the truncation is recorded.
"""
import os, sys, json, time, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import transfer_experiment as TE
import full_benchmark as FB
import chain_metrics as CM
import ecb_deepseek_pilot as EP
from llm_response import classify_response, MAX_TOKENS_CAP

W = FB.W
BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
LOG = os.path.join(ROOT, "results", "llm_call_log.jsonl")
OUT_J = os.path.join(ROOT, "results", "ecb_deepseek_all.jsonl")
OUT_X = os.path.join(ROOT, "results", "ecb_deepseek_all.xlsx")
CTX_TOKEN_BUDGET = 60000  # dynamic context budget (estimated tokens)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def call_logged(key, purpose, conv_id, prompt, max_tokens, logf, max_retries=3):
    """LLM call with an evidence log. Returns (resp|None, meta).

    meta: {failure_mode, finish_reason, attempts, max_tokens_final}
    Failure modes are mutually exclusive and decidable: network / output_truncated / empty_content / bad_response
    """
    est_tok = len(prompt) // 2
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "endpoint": BASE + "/chat/completions",
           "model": MODEL, "purpose": purpose, "conv_id": conv_id,
           "prompt_chars": len(prompt), "est_prompt_tokens": est_tok,
           "max_tokens": max_tokens, "truncated": est_tok > CTX_TOKEN_BUDGET}
    if est_tok > CTX_TOKEN_BUDGET:
        prompt = prompt[-CTX_TOKEN_BUDGET * 2:]
        rec["truncated"] = True
        rec["prompt_chars_after"] = len(prompt)
    body = None; status = ""; resp = None; fr = None; fmode = None; usage = None
    mt = min(max_tokens, MAX_TOKENS_CAP)
    attempts = 0
    for attempt in range(max_retries):
        attempts = attempt + 1
        req = urllib.request.Request(BASE + "/chat/completions",
            data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": mt, "temperature": 0.2,
                             "response_format": {"type": "json_object"}}).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.loads(r.read()); status = "HTTP 200"
        except Exception as e:
            status = "ERROR: " + str(e)[:200]; body = None
        if body is not None:
            resp, fr, fmode = classify_response(body)
            usage = body.get("usage")
        if fmode is None:
            break
        if fmode == "output_truncated" and mt < MAX_TOKENS_CAP:
            mt = MAX_TOKENS_CAP          # truncated -> raise the output budget and retry
            rec["max_tokens_escalated_to"] = mt
            continue
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))
    rec.update({"latency_s": round(time.time() - t0, 1), "status": status,
                "response": resp, "usage": usage, "finish_reason": fr,
                "failure_mode": fmode, "attempts": attempts, "max_tokens_final": mt})
    logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
    return (None if fmode else resp), {"failure_mode": fmode, "finish_reason": fr,
                                       "attempts": attempts, "max_tokens": mt}

def parse_chains_and_bt(resp, n):
    if not resp:
        return None, {}
    try:
        d = json.loads(resp)
    except Exception:
        try:
            d = json.loads(resp[resp.index("{"):resp.rindex("}") + 1])
        except Exception:
            return None, {}
    return (d.get("chains"), {int(x["message_id"]): int(x["parent"])
            for x in d.get("backtrack", []) if 0 <= int(x["parent"]) < n})

def metrics_of(pred, gold):
    """Delegates to the authoritative chain_metrics implementation (P0-2/P0-3), keeping the legacy
    field aliases so that historical results stay comparable."""
    m = CM.conv_chain_metrics(gold, pred)
    # legacy column names (older results use ancestor_jaccard / component_recall)
    m["ancestor_jaccard"] = m.get("m3_ancestor_jaccard")
    m["component_recall"] = m.get("m2_component_recall")
    m["component_precision"] = m.get("m2_component_precision")
    return m

def main():
    key = EP.load_key()
    t0 = time.time()
    logf = open(LOG, "a")
    OUT_J = os.path.join(ROOT, "results", "ecb_deepseek_all.jsonl")
    done = set()
    if os.path.exists(OUT_J):
        for l in open(OUT_J):
            try:
                r = json.loads(l); done.add((r["dataset"], r["conv_id"]))
            except Exception: pass
    hasher = TE.Hasher(); hasher.fit(None)
    loaders = {"irc": TE.load_irc, "molweni": TE.load_molweni,
               "discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack, "hn": FB.load_hn, "reddit": FB.load_reddit}
    f = open(OUT_J, "a")
    for name, loader in loaders.items():
        splits = loader()
        log(f"=== {name}: {len(splits['test'])} test convs ===")
        for conv in splits["test"]:
            cid = conv.get("conv_id", "")
            if (name, cid) in done:
                continue
            msgs = conv["msgs"]; n = len(msgs)
            if n < 5:
                continue
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            # Call A: whole-conversation chain partition + backtrack (merged, dynamic length)
            prompt = ("Below is a multi-party chat (index [i] per message). Do TWO things:\n"
                      "1) Partition ALL messages into complete event chains (topic-coherent threads); "
                      "every message must belong to exactly one chain.\n"
                      "2) For messages whose preceding message is ambiguous or missing, provide a "
                      "backtrack parent, or -1 if truly none.\n"
                      'Output JSON: {"chains":[{"topic":str,"message_ids":[ints]}],'
                      '"backtrack":[{"message_id":int,"parent":int}]}\n\n'
                      "Chat:\n" + EP.conv_text(msgs))
            resp, meta_a = call_logged(key, "chains+backtrack", f"{name}/{cid}", prompt,
                                       EP.dyn_len(n), logf)
            n_calls = meta_a["attempts"]
            failure_modes = [meta_a["failure_mode"]] if meta_a["failure_mode"] else []
            chains, bt = parse_chains_and_bt(resp, n)
            pred = EP.chains_to_edges(msgs, chains, bt)
            iters = 1
            # Call B iteration: messages that are still unassigned (<=2 rounds)
            for rnd in range(2):
                un = [i for i in range(n) if not pred[i] and i > 0]
                if not un:
                    break
                p2 = ("Full chat:\n" + EP.conv_text(msgs) +
                      "\n\nUnassigned messages: " + str(un) +
                      ". For EACH, output its most likely preceding message index, or -1. "
                      'Output JSON: {"backtrack":[{"message_id":int,"parent":int}]}')
                r2, meta_b = call_logged(key, "backtrack-iter", f"{name}/{cid}", p2,
                                         EP.dyn_len(len(un)), logf)
                n_calls += meta_b["attempts"]
                if meta_b["failure_mode"]:
                    failure_modes.append(meta_b["failure_mode"])
                _, bt2 = parse_chains_and_bt(r2, n)
                added = 0
                for mid, par in bt2.items():
                    if 0 <= par < n and par != mid and par not in pred[mid]:
                        pred[mid].append(par); added += 1
                iters += 1
                if added == 0:
                    break
            m = metrics_of(pred, gold)
            row = {"dataset": name, "algorithm": "ECB-DeepSeek", "conv_id": cid,
                   "n_msgs": n, "n_chains": len(chains or []), "iterations": iters,
                   "n_llm_calls": n_calls,
                   "failure_modes": ";".join(failure_modes) if failure_modes else None,
                   "degraded": bool(failure_modes), **m,
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            log(f"  {name}/{cid[:22]} n={n} ancJ={m['ancestor_jaccard']} pathF1={m['path_F1']}")
            import gc; gc.collect()
        del splits
        import gc; gc.collect()
    f.close(); logf.close()
    import pandas as pd
    rows = [json.loads(l) for l in open(OUT_J)]
    pd.DataFrame([{k: v for k, v in r.items()} for r in rows]).to_excel(
        OUT_X, index=False)
    log(f"ALL_DONE convs={len(rows)} in {round(time.time()-t0)}s -> {OUT_X}")

if __name__ == "__main__":
    main()
