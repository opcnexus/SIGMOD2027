"""Solution-space state analysis: reply structure statistics of the Ubuntu IRC
dataset.

Core questions:
  Q1 Does any message reply to two (or more) preceding messages? (multi-parent
     messages)
  Q2 What is the parent distance / distribution of multi-parent messages? What
     share falls inside the candidate window?
  Q3 What is the share of tree structure (single-parent chains) versus DAG
     (multi-parent)? -> does the single-label pointer softmax assumption hold?
  Q4 root ratio and parent-child distance distribution -> the basis for setting the
     decoder window / backtracking depth.

Output: results/data_space_analysis.json (completed_at at minute precision)
"""

import os
import sys
import json
from datetime import datetime
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import DATA_ROOT, TEST_START

OUT = "./results/data_space_analysis.json"
WINDOW = 100  # same candidate window as FF/DiGAT (self + previous 100)


def analyze_file(log, W=WINDOW):
    """Return the structural statistics of this file. Note that the gold parents
    are given as source -> [targets]."""
    n = len(log.messages)
    n_annotated = 0          # messages that appear in parents (labelled queries)
    n_root = 0               # labelled as root (the only parent is itself)
    n_single = 0             # exactly 1 parent different from itself
    n_multi = 0              # >=2 parents different from itself (genuinely multi-parent)
    multi_depths = []        # list of parent distances of multi-parent messages
    single_depths = []       # parent distances of single-parent messages
    out_of_window = 0        # number of messages with a parent outside the window (j < i-W)
    multi_pairs = []         # (i, distance list) samples, for case studies

    for i in range(n):
        ts = log.parents.get(i)
        if ts is None:
            continue
        n_annotated += 1
        real = sorted(t for t in ts if t != i)
        if not real:
            n_root += 1
            continue
        if len(real) == 1:
            n_single += 1
            single_depths.append(i - real[0])
        else:
            n_multi += 1
            ds = [i - t for t in real]
            multi_depths.extend(ds)
            if len(multi_pairs) < 8:
                multi_pairs.append({
                    "file": os.path.basename(log.name),
                    "msg_i": i,
                    "parents": real,
                    "distances": ds,
                })
        if real and min(real) < i - W:
            out_of_window += 1
    return dict(
        n=n, n_annotated=n_annotated, n_root=n_root, n_single=n_single,
        n_multi=n_multi, out_of_window=out_of_window,
        single_depths=single_depths, multi_depths=multi_depths,
        multi_pairs=multi_pairs,
    )


def pct(x, total):
    return round(100.0 * x / max(1, total), 2)


def main():
    all_stats = {s: [] for s in ("train", "dev", "test")}
    for split in ("train", "dev", "test"):
        logs = load_split(split, DATA_ROOT)
        for log in logs:
            st = analyze_file(log)
            st["file"] = os.path.basename(log.name)
            all_stats[split].append(st)

    def agg(split):
        sts = all_stats[split]
        tot_ann = sum(s["n_annotated"] for s in sts)
        tot_root = sum(s["n_root"] for s in sts)
        tot_single = sum(s["n_single"] for s in sts)
        tot_multi = sum(s["n_multi"] for s in sts)
        tot_ow = sum(s["out_of_window"] for s in sts)
        sd = np.array([d for s in sts for d in s["single_depths"]], dtype=np.float64)
        md = np.array([d for s in sts for d in s["multi_depths"]], dtype=np.float64)
        n_msg = sum(s["n"] for s in sts)
        out = {
            "n_files": len(sts),
            "n_messages": n_msg,
            "n_annotated_queries": tot_ann,
            "root": {"count": tot_root, "pct_of_annotated": pct(tot_root, tot_ann)},
            "single_parent": {"count": tot_single,
                              "pct_of_annotated": pct(tot_single, tot_ann)},
            "multi_parent": {"count": tot_multi,
                             "pct_of_annotated": pct(tot_multi, tot_ann),
                             "pct_of_nonroot": pct(tot_multi, tot_single + tot_multi)},
            "parent_out_of_window_msgs": {"count": tot_ow,
                                          "pct_of_annotated": pct(tot_ow, tot_ann)},
            "single_parent_distance": {
                "mean": round(float(sd.mean()), 2) if len(sd) else None,
                "p50": round(float(np.percentile(sd, 50)), 1) if len(sd) else None,
                "p90": round(float(np.percentile(sd, 90)), 1) if len(sd) else None,
                "p99": round(float(np.percentile(sd, 99)), 1) if len(sd) else None,
            } if len(sd) else {},
            "multi_parent_distance": {
                "mean": round(float(md.mean()), 2) if len(md) else None,
                "p50": round(float(np.percentile(md, 50)), 1) if len(md) else None,
            } if len(md) else {},
            "multi_parent_examples": [p for s in sts for p in s["multi_pairs"]][:8],
        }
        # parent distance histogram (single parent, bins: 1, 2-3, 4-10, 11-50,
        # 51-100, >100)
        if len(sd):
            bins = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 100), (101, 10**9)]
            labels = ["1", "2-3", "4-10", "11-50", "51-100", ">100"]
            out["single_parent_distance_hist"] = {
                lab: int(((sd >= a) & (sd <= b)).sum()) for lab, (a, b) in zip(labels, bins)
            }
        return out

    result = {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window": WINDOW,
        "splits": {s: agg(s) for s in ("train", "dev", "test")},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", OUT)

    # print a summary (train/dev/test merged view + test)
    tot_ann = sum(result["splits"][s]["n_annotated_queries"] for s in ("train", "dev", "test"))
    tot_multi = sum(result["splits"][s]["multi_parent"]["count"] for s in ("train", "dev", "test"))
    tot_root = sum(result["splits"][s]["root"]["count"] for s in ("train", "dev", "test"))
    tot_single = sum(result["splits"][s]["single_parent"]["count"] for s in ("train", "dev", "test"))
    tot_ow = sum(result["splits"][s]["parent_out_of_window_msgs"]["count"] for s in ("train", "dev", "test"))
    print("== full set (173 files) ==")
    print("annotated queries:", tot_ann)
    print("root      : {} ({:.2f}%)".format(tot_root, pct(tot_root, tot_ann)))
    print("single-par: {} ({:.2f}%)".format(tot_single, pct(tot_single, tot_ann)))
    print("multi-par : {} ({:.2f}% of annotated, {:.2f}% of non-root)".format(
        tot_multi, pct(tot_multi, tot_ann), pct(tot_multi, tot_single + tot_multi)))
    print("msgs with parent outside window: {} ({:.2f}%)".format(tot_ow, pct(tot_ow, tot_ann)))
    t = result["splits"]["train"]
    print("train single-parent distance hist:", t.get("single_parent_distance_hist"))
    for ex in t["multi_parent_examples"][:3]:
        print("multi-parent example:", ex)


if __name__ == "__main__":
    main()
