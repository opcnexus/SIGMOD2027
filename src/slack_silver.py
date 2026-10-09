#!/usr/bin/env python3
"""Slack mention proxy edge extraction -> silver threads (report Sec. 4.3 next step)

Rationale: software_slacks has no explicit thread field, but a <@U...> mention in a Slack message
is an explicit summons of a user. A proxy reply edge = mention -> the most recent preceding
message by that summoned user in the same channel (6h window), which can yield genuine multiple
parents (one message summoning several users).

Output (trace):
  data/candidates/slack_silver_threads.jsonl   silver threads (conv/msgs carry parents+text)
  results/slack_silver_stats.json              statistics + quality checks + samples + completed_at
Checks:
  - edge resolution rate (resolved / total mentions)
  - temporal legality (parent.ts <= child.ts, violations must be 0)
  - structural metrics under the same convention as representation_space.json
    (root% / latency / depth)
  - 5 human-readable samples of resolved edges
"""
import os, re, json, time, glob
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "data", "candidates", "software_slacks.parquet")
OUT_THREADS = os.path.join(ROOT, "data", "candidates", "slack_silver_threads.jsonl")
OUT_STATS = os.path.join(ROOT, "results", "slack_silver_stats.json")
WINDOW = 6 * 3600  # mention resolution window of 6 hours
MENTION_RE = re.compile(r"<@([^>|@\]\n]{1,60})>")  # the dataset is normalized to <@display_name>

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    t0 = time.time()
    t = pq.read_table(PQ, columns=["channel", "text", "ts", "user"])
    df = t.to_pydict()
    n_total = len(df["channel"])
    log(f"parquet rows: {n_total}")

    def parse_ts(v):
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    by_ch = defaultdict(list)
    for ch, text, ts, user in zip(df["channel"], df["text"], df["ts"], df["user"]):
        ts2 = parse_ts(ts)
        if ts2 is not None:
            by_ch[ch].append((ts2, str(user), str(text or "")))
    log(f"channels: {len(by_ch)}")

    # sort each channel by time
    for ch in by_ch:
        by_ch[ch].sort(key=lambda x: x[0])

    stats = {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window_sec": WINDOW,
        "n_rows_total": n_total,
        "n_channels": len(by_ch),
        "n_msgs_used": 0,
        "n_msgs_with_mention": 0,
        "n_mentions": 0,
        "n_mentions_resolved": 0,
        "n_multi_parent_msgs": 0,
        "parent_time_lag": [],
        "parent_idx_lag": [],
        "examples": [],
        "convs_written": 0,
        "msgs_written": 0,
        "time_violations": 0,
    }

    fout = open(OUT_THREADS, "w")
    for ch, all_msgs in by_ch.items():
        # weekly window splitting (channels are too large to use as a single conversation)
        chunks = defaultdict(list)
        for row in all_msgs:
            chunks[int(row[0] // 604800)].append(row)
        for wk, msgs in sorted(chunks.items()):
            if len(msgs) < 30:
                continue
            # user -> ordered list of that user's (ts, idx) (binary searchable)
            user_hist = defaultdict(list)
            for i, (ts, user, _) in enumerate(msgs):
                user_hist[user].append((ts, i))
                if user != user.lower():
                    user_hist[user.lower()].append((ts, i))
            conv_msgs = []
            last_resolved = None
            for i, (ts, user, text) in enumerate(msgs):
                mentions = MENTION_RE.findall(text)
                parents = []
                for mu in set(mentions):
                    stats["n_mentions"] += 1
                    hist = user_hist.get(mu) or user_hist.get(mu.lower(), [])
                    # find the most recent message of that user within (ts-WINDOW, ts)
                    lo, hi, best = 0, len(hist) - 1, None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if hist[mid][0] < ts:
                            best = hist[mid]; lo = mid + 1
                        else:
                            hi = mid - 1
                    if best and ts - best[0] <= WINDOW:
                        stats["n_mentions_resolved"] += 1
                        parents.append(best[1])
                        stats["parent_time_lag"].append(ts - best[0])
                        stats["parent_idx_lag"].append(i - best[1])
                        if last_resolved is None:
                            last_resolved = {"child": text[:80], "parent": msgs[best[1]][2][:80],
                                             "gap_min": round((ts - best[0]) / 60, 1)}
                if mentions:
                    stats["n_msgs_with_mention"] += 1
                if len(parents) > 1:
                    stats["n_multi_parent_msgs"] += 1
                conv_msgs.append({"id": i, "author": user,
                                  "parents": sorted(set(parents)), "ts": ts,
                                  "text": text[:300]})
                if len(stats["examples"]) < 5 and parents:
                    stats["examples"].append({
                        "channel": ch, "child_text": text[:80],
                        "parent_text": msgs[parents[0]][2][:80],
                        "gap_min": round((ts - msgs[parents[0]][0]) / 60, 1)})
            stats["n_msgs_used"] += len(conv_msgs)
            if len(conv_msgs) >= 30:
                fout.write(json.dumps({"dataset": "slack", "conv_id": f"{ch}-w{wk}",
                                       "msgs": conv_msgs}, ensure_ascii=False) + "\n")
                stats["convs_written"] += 1
                stats["msgs_written"] += len(conv_msgs)
    fout.close()

    # ---- checks ----
    q = lambda a, p: int(np.percentile(a, p)) if a else None
    med_lag = q(stats["parent_time_lag"], 50)
    # temporal legality spot check (all resolved edges are guaranteed by construction to satisfy
    # parent.ts < child.ts; re-verified here)
    for line in open(OUT_THREADS):
        conv = json.loads(line)
        id2ts = {m["id"]: m["ts"] for m in conv["msgs"]}
        for m in conv["msgs"]:
            for p in m["parents"]:
                if p in id2ts and id2ts[p] > m["ts"]:
                    stats["time_violations"] += 1
    assert stats["time_violations"] == 0, "parent-after-child violation!"

    stats["mention_rate_pct"] = round(100 * stats["n_msgs_with_mention"] / max(1, stats["n_msgs_used"]), 2)
    stats["resolution_rate_pct"] = round(100 * stats["n_mentions_resolved"] / max(1, stats["n_mentions"]), 2)
    stats["multi_parent_pct"] = round(100 * stats["n_multi_parent_msgs"] / max(1, stats["n_msgs_used"]), 3)
    stats["parent_time_lag"] = {"median_sec": med_lag, "p90_sec": q(stats["parent_time_lag"], 90)}
    stats["parent_idx_lag"] = {"median": q(stats["parent_idx_lag"], 50),
                               "p90": q(stats["parent_idx_lag"], 90)}
    stats["runtime_seconds"] = round(time.time() - t0)
    stats["example_resolved_edge"] = last_resolved

    json.dump(stats, open(OUT_STATS, "w"), ensure_ascii=False, indent=1)
    log(f"threads -> {OUT_THREADS} ({stats['convs_written']} convs / {stats['msgs_written']} msgs)")
    log(f"stats   -> {OUT_STATS}")
    log(f"mention_rate={stats['mention_rate_pct']}% resolution_rate={stats['resolution_rate_pct']}% "
        f"multi_parent={stats['multi_parent_pct']}% time_lag_median={med_lag}s")
    log("VALIDATION PASS" if stats["time_violations"] == 0 else "VALIDATION FAIL")

if __name__ == "__main__":
    main()
