import os
#!/usr/bin/env python3
"""Unified cross-dataset representation space statistics.

Unified intermediate representation: conv = [{id, author, parents:[id], ts(epoch s)|None}]
Data sources:
  1. irc      — data/irc-disentanglement/data/{train,dev,test}/*.{tok,annotation,ascii}
  2. molweni  - data/molweni/DP_{train,dev,test}.json (EDU level, x=parent y=child)
  3. hn       - data/candidates/hn_threads.jsonl (Algolia crawl)
  4. bilibili - data/candidates/bilibili_threads.jsonl (public API crawl)
  5. discord  - data/candidates/discord/*.jsonl (Discord-Unveiled sample, HF mirror)

Output results/representation_space.json
"""
import json, os, re, glob, statistics
from datetime import datetime
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "representation_space.json")


def load_irc():
    convs = []
    for split in ("train", "dev", "test"):
        base = os.path.join(ROOT, "data", "irc-disentanglement", "data", split)
        if not os.path.isdir(base):
            continue
        for tok_f in sorted(glob.glob(os.path.join(base, "*.tok.txt")))[:80]:
            stem = tok_f[:-len(".tok.txt")]
            ann_f = stem + ".annotation.txt"
            ascii_f = stem + ".ascii.txt"
            if not (os.path.exists(ann_f) and os.path.exists(ascii_f)):
                continue
            msgs = {}
            for i, line in enumerate(open(tok_f, encoding="utf-8", errors="ignore")):
                t = line.strip()
                if t.startswith("<s>"):
                    t = t[3:]
                if t.endswith("</s>"):
                    t = t[:-4]
                msgs[i] = {"id": i, "author": None, "parents": [], "ts": None,
                           "tlen": len(t.strip())}
            for i, line in enumerate(open(ascii_f, encoding="utf-8", errors="ignore")):
                if i not in msgs:
                    continue
                m = re.match(r"\[(\d{2}):(\d{2})\]\s+<([^>]+)>", line)
                if m:
                    msgs[i]["ts"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    msgs[i]["author"] = m.group(3)
            for line in open(ann_f):
                parts = line.split()
                if len(parts) < 2:
                    continue
                a, b = int(parts[0]), int(parts[1])
                if a == b or a not in msgs or b not in msgs:
                    continue  # self-link = root
                msgs[max(a, b)]["parents"].append(min(a, b))
            convs.append({"dataset": "irc", "conv_id": os.path.basename(stem),
                          "msgs": list(msgs.values())})
    return convs


def load_molweni():
    convs = []
    for split in ("train", "dev", "test"):
        p = os.path.join(ROOT, "data", "molweni", f"DP_{split}.json")
        if not os.path.exists(p):
            continue
        data = json.load(open(p))
        for ci, conv in enumerate(data):
            n = len(conv["edus"])
            msgs = [{"id": k, "author": conv["edus"][k].get("speaker"),
                     "parents": [], "ts": None,
                     "tlen": len(conv["edus"][k].get("text") or "")} for k in range(n)]
            for rel in conv["relations"]:
                x, y = rel["x"], rel["y"]
                if 0 <= x < n and 0 <= y < n:
                    msgs[y]["parents"].append(x)
            convs.append({"dataset": "molweni", "conv_id": f"{split}-{ci}", "msgs": msgs})
    return convs


def load_jsonl(path):
    convs = []
    if not os.path.exists(path):
        return convs
    for line in open(path):
        d = json.loads(line)
        convs.append({"dataset": d["dataset"], "conv_id": d["conv_id"],
                      "msgs": [{"id": m["id"], "author": m.get("author"),
                                "parents": ([m["parent"]] if m.get("parent") else []),
                                "ts": m.get("ts"), "tlen": m.get("tlen", 0)}
                               for m in d["msgs"]]})
    return convs


def load_discord():
    convs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "candidates", "discord", "*.json"))):
        by_chan = defaultdict(list)
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("is_bot") or d.get("type") not in (0, 19, 21):
                continue
            day = (d.get("timestamp") or "")[:10]
            ts = None
            if d.get("timestamp"):
                try:
                    ts = int(datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).timestamp())
                except Exception:
                    ts = None
            week = None
            by_chan[(d.get("channel_id"), week)].append({
                "id": d["id"], "author": d["author"]["id"],
                "parents": ([d["message_reference"]["message_id"]]
                            if d.get("message_reference", {}).get("message_id") else []),
                "ts": ts, "tlen": len(d.get("content") or ""),
                "chan": d.get("channel_name")})
        for (cid, week), msgs in by_chan.items():
            if len(msgs) < 15:
                continue
            ids = {m["id"] for m in msgs}
            for m in msgs:
                m["parents"] = [p for p in m["parents"] if p in ids]
            msgs.sort(key=lambda m: (m["ts"] if m["ts"] else 0))
            convs.append({"dataset": "discord", "conv_id": f"{cid}-w{week}",
                          "msgs": msgs})
    return convs


def load_slack():
    """software_slacks (HF spencer/software_slacks): 1.39M Slack messages, without explicit threads.

    Unified conv=(workspace, channel, day); parent is always empty; mention_rate is attached as a
    reply proxy.
    Sampling: descending by workspace-channel volume, accumulating until ~80k messages.
    """
    import pyarrow.parquet as pq
    p = os.path.join(ROOT, "data", "candidates", "software_slacks.parquet")
    if not os.path.exists(p):
        return []
    t = pq.read_table(p, columns=["workspace", "channel", "text", "ts", "user"])
    rows = t.to_pylist()
    by_chan = defaultdict(list)
    for r in rows:
        if not r["ts"]:
            continue
        by_chan[(r["workspace"], r["channel"], r["ts"][:10])].append(r)
    # filter out small conversations, sort by volume
    big = [(k, v) for k, v in by_chan.items() if len(v) >= 30]
    big.sort(key=lambda kv: -len(kv[1]))
    convs, total = [], 0
    for (ws, ch, day), rs in big:
        if total >= 80_000:
            break
        rs = sorted(rs, key=lambda r: r["ts"])
        msgs = []
        for r in rs:
            ts = None
            try:
                ts = int(datetime.fromisoformat(r["ts"]).timestamp())
            except Exception:
                ts = None
            msgs.append({"id": None, "author": r.get("user"), "parents": [],
                         "ts": ts, "tlen": len(r.get("text") or ""),
                         "mention": "<@" in (r.get("text") or "")})
        convs.append({"dataset": "slack", "conv_id": f"{ws}/{ch}/{day}",
                      "structured": False, "msgs": msgs})
        total += len(msgs)
    return convs


def load_jsonl_multi(paths):
    convs = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            convs.append({"dataset": d["dataset"], "conv_id": d["conv_id"],
                          "structured": True,
                          "msgs": [{"id": m["id"], "author": m.get("author"),
                                    "parents": ([m["parent"]] if m.get("parent") else []),
                                    "ts": m.get("ts"), "tlen": m.get("tlen", 0)}
                                   for m in d["msgs"]]})
    return convs


def depth_of(msgs):
    par = {m["id"]: [p for p in m["parents"] if p != m["id"]] for m in msgs}
    memo, in_stack = {}, set()
    def dp(i):
        if i in memo:
            return memo[i]
        if i in in_stack:
            return 0  # cycle guard
        in_stack.add(i)
        ps = par.get(i, [])
        d = 1 + max((dp(p) for p in ps), default=-1) if ps else 0
        in_stack.discard(i)
        memo[i] = d
        return d
    return [dp(m["id"]) for m in msgs]


def stats_dataset(convs):
    if not convs:
        return None
    structured = all(c.get("structured", True) for c in convs)
    n_msg = n_root = n_multi = 0
    n_mention = 0
    parent_counts = []
    conv_sizes, conv_parts = [], []
    lag_idx, lag_time, interleave, depths = [], [], [], []
    gaps = []
    hour_hist = Counter()
    for conv in convs:
        msgs = conv["msgs"]
        conv_sizes.append(len(msgs))
        conv_parts.append(len({m["author"] for m in msgs if m["author"]}))
        id2pos = {m["id"]: i for i, m in enumerate(msgs)} if structured else {}
        # IRC semantics: the first 1000 lines are context and are labelled as having no parent edges
        # -> not counted in the root statistics
        min_rootable = 1000 if conv["dataset"] == "irc" else 0
        prev_ts = None
        for m in msgs:
            countable = (id2pos.get(m["id"], 0) >= min_rootable) or min_rootable == 0
            if countable:
                n_msg += 1
            if m.get("mention"):
                n_mention += 1
            if m["ts"] and prev_ts is not None and m["ts"] > prev_ts:
                gaps.append(m["ts"] - prev_ts)
            prev_ts = m["ts"] if m["ts"] else prev_ts
            ps = m["parents"]
            if countable and structured:
                parent_counts.append(len(ps))
            if not ps:
                if countable and structured:
                    n_root += 1
                continue
            if not structured:
                continue
            if len(ps) > 1:
                if countable:
                    n_multi += 1
            i = id2pos.get(m["id"])
            for p in ps:
                j = id2pos.get(p)
                if j is not None and i is not None and i > j:
                    lag_idx.append(i - j)
                    interleave.append(i - j - 1)
                if m["ts"]:
                    pts = next((x["ts"] for x in msgs if x["id"] == p and x["ts"]), None)
                    if pts is not None and m["ts"] >= pts:
                        lag_time.append(m["ts"] - pts)
            if m["ts"]:
                hour_hist[datetime.utcfromtimestamp(m["ts"]).hour] += 1
        if structured:
            depths.extend(depth_of(msgs))
    q = lambda a, p: round(statistics.quantiles(a, n=10)[p - 1], 1) if len(a) >= 10 else None
    out = {
        "structured": structured,
        "n_conv": len(convs), "n_msg": n_msg,
        "msgs_per_conv": {"median": statistics.median(conv_sizes),
                          "mean": round(statistics.mean(conv_sizes), 1),
                          "p90": q(conv_sizes, 9)},
        "participants_per_conv": {"median": statistics.median(conv_parts),
                                  "mean": round(statistics.mean(conv_parts), 1)},
        "msg_gap_sec": {"median": int(statistics.median(gaps)) if gaps else None,
                        "p90": int(q(gaps, 9)) if gaps and q(gaps, 9) is not None else None},
        "mention_rate_pct": round(100 * n_mention / n_msg, 2) if n_msg else None,
    }
    if structured:
        out.update({
            "root_pct": round(100 * n_root / n_msg, 2),
            "multi_parent_pct": round(100 * n_multi / n_msg, 2),
            "avg_parents_per_msg": round(statistics.mean(parent_counts), 3) if parent_counts else 0,
            "parent_lag_index": {"median": statistics.median(lag_idx) if lag_idx else None,
                                 "within5_pct": round(100 * sum(1 for x in lag_idx if x <= 5) / len(lag_idx), 1) if lag_idx else None,
                                 "gt20_pct": round(100 * sum(1 for x in lag_idx if x > 20) / len(lag_idx), 1) if lag_idx else None},
            "interleave_msgs": {"median": statistics.median(interleave) if interleave else None},
            "parent_child_time_lag_sec": {"median": int(statistics.median(lag_time)) if lag_time else None,
                                          "p90": q(lag_time, 9)} if lag_time else None,
            "reply_depth": {"mean": round(statistics.mean(depths), 2),
                            "max": max(depths)} if depths else None,
        })
    out["hour_hist_utc"] = dict(sorted(hour_hist.items()))
    out["n_edges"] = sum(1 for c in convs for m in c["msgs"] for _ in m["parents"])
    return out


def main():
    cand = os.path.join(ROOT, "data", "candidates")
    all_convs = (load_irc() + load_molweni() + load_slack() + load_discord()
                 + load_jsonl_multi([
                     os.path.join(cand, "hn_threads.jsonl"),
                     os.path.join(cand, "bilibili_threads.jsonl"),
                     os.path.join(cand, "bilibili_threads.jsonl.scale"),
                     os.path.join(cand, "reddit_threads.jsonl")]))
    by_ds = defaultdict(list)
    for c in all_convs:
        by_ds[c["dataset"]].append(c)
    out = {"completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "datasets": {}}
    for ds in ("irc", "molweni", "hn", "reddit", "bilibili", "discord", "slack"):
        s = stats_dataset(by_ds.get(ds, []))
        if s:
            out["datasets"][ds] = s
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    for ds, s in out["datasets"].items():
        print(f"\n=== {ds} ===")
        print(f"  convs={s['n_conv']} msgs={s['n_msg']} msgs/conv={s['msgs_per_conv']['median']}"
              f" participants/conv={s['participants_per_conv']['median']}"
              f" gap_med={s['msg_gap_sec']['median']}s")
        if s.get("structured"):
            _tl = s.get("parent_child_time_lag_sec") or {}
            print(f"  root%={s['root_pct']} multi-parent%={s['multi_parent_pct']}"
                  f" parent-lag med={s['parent_lag_index']['median']}"
                  f" time-lag med={_tl.get('median')}s"
                  f" depth={s['reply_depth']['mean']}")
        else:
            print(f"  [no explicit reply structure] mention_rate={s['mention_rate_pct']}%")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
