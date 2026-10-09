#!/usr/bin/env python3
"""ECB scale-up: all test conversations + windowing of long Slack conversations + a DeepSeek reasoner control.
All intermediate results are persisted: results/ecb_full.jsonl (per conversation / per window) + xlsx/csv.
Resume support: completed entries are skipped by (dataset, conv_id, window_id, model).
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
WIN = 300
OUT_JSONL = os.path.join(ROOT, "results", "ecb_full.jsonl")
OUT_XLSX = os.path.join(ROOT, "results", "ecb_full.xlsx")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def done_keys():
    done = set()
    if os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL):
            try:
                r = json.loads(line)
                done.add((r["dataset"], r["conv_id"], r.get("window_id", 0), r["algorithm"]))
            except Exception:
                pass
    return done

def main():
    key = EP.load_key()
    t0 = time.time()
    done = done_keys()
    f = open(OUT_JSONL, "a")
    hasher = TE.Hasher(); hasher.fit(None)
    msg_vec = lambda m: hasher.msg_vec(m["text"])
    loaders = {"discord": lambda: TE.split_convs(TE.load_discord()),
               "slack": FB.load_slack}
    for name, loader in loaders.items():
        splits = loader()
        test = splits["test"]
        log(f"=== {name}: {len(test)} test convs ===")
        for conv in test:
            msgs = conv["msgs"]
            n = len(msgs)
            if n < 5:
                continue
            id2pos = {m["id"]: i for i, m in enumerate(msgs)}
            gold_par = [[id2pos[p] for p in m["parents"] if p in id2pos] for m in msgs]
            # window long conversations (Slack 1200 messages -> 4x300)
            wins = [(s, min(s + WIN, n)) for s in range(0, n, WIN)] or [(0, n)]
            merged_pred = [[] for _ in range(n)]
            n_chains = 0
            fail_modes = []
            for wid, (s, e) in enumerate(wins):
                seg = msgs[s:e]
                if len(seg) < 3:
                    continue
                ck = (name, conv.get("conv_id", ""), wid, "deepseek-chat")
                if ck in done:
                    continue
                chains = EP.call_a(seg, key)
                if EP.LAST_META.get("failure_mode"):
                    fail_modes.append("A:" + EP.LAST_META["failure_mode"])
                assigned = {int(x) for ch in (chains or []) for x in ch.get("message_ids", [])}
                unres = [s + j for j in range(len(seg)) if j not in assigned]
                bt = EP.call_b(msgs, unres, key) if unres else {}
                if unres and EP.LAST_META.get("failure_mode"):
                    fail_modes.append("B:" + EP.LAST_META["failure_mode"])
                pred = EP.chains_to_edges(seg, chains, {k - s: v - s for k, v in bt.items() if s <= v < e})
                for i in range(s, e):
                    for p in pred[i - s]:
                        merged_pred[i].append(p)
                n_chains += len(chains or [])
                row = {"dataset": name, "algorithm": "ECB-DeepSeek",
                       "conv_id": conv.get("conv_id", ""), "window_id": wid,
                       "window": [s, e], "n_msgs": len(seg),
                       "n_chains": len(chains or []),
                       "n_unassigned_in_window": len(unres),
                       "raw_chains": chains, "saved_at": time.strftime("%H:%M:%S")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            for i in range(n):
                merged_pred[i] = sorted({p for p in merged_pred[i] if 0 <= p < i})
            # P0-2/P0-3: delegate to the authoritative implementation (both M2 conventions + path
            # truncation recorded + no 0 fill for degenerate inputs)
            mm = CM.conv_chain_metrics(gold_par, merged_pred)
            summ = {"dataset": name, "algorithm": "ECB-DeepSeek",
                    "conv_id": conv.get("conv_id", ""), "window_id": -1, "n_msgs": n,
                    "n_chains": n_chains, "n_windows": len(wins),
                    "metric_version": mm["metric_version"],
                    "path_F1": mm["path_F1"], "path_F1_zero_filled": mm["path_F1_zero_filled"],
                    "component_recall": mm["m2_component_recall"],
                    "component_precision": mm["m2_component_precision"],
                    "ancestor_jaccard": mm["m3_ancestor_jaccard"],
                    "path_gold_truncated": mm["path_gold_truncated"],
                    "path_pred_truncated": mm["path_pred_truncated"],
                    "n_llm_failures": len(fail_modes),
                    "failure_modes": ";".join(fail_modes) if fail_modes else None,
                    "saved_at": time.strftime("%H:%M:%S")}
            f.write(json.dumps(summ, ensure_ascii=False) + "\n"); f.flush()
            log(f"  {name}/{summ['conv_id']} n={n} wins={len(wins)} pathF1={summ['path_F1']} "
                f"compR={summ['component_recall']} ancJ={summ['ancestor_jaccard']}")
        del splits
        import gc; gc.collect()
    f.close()
    # Excel
    import pandas as pd
    rows = [json.loads(l) for l in open(OUT_JSONL)]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "raw_chains"} for r in rows])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="ecb_all", index=False)
    log(f"ALL_DONE in {round(time.time()-t0)}s rows={len(rows)} -> {OUT_XLSX}")

if __name__ == "__main__":
    main()
