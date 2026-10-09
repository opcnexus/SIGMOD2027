#!/usr/bin/env python3
"""Temporal-dimension feature extraction v2 (parses the [HH:MM] timestamps of the ascii files directly).

Features (2 dimensions per message):
1. gap_log = log1p(minutes since the previous message) (wraps around midnight; the first message = 0)
2. active_flag = whether the hour of the message is "active" (count in the train-pool hour histogram
   >= median)

Alignment: the line numbers of ascii.txt and tok.txt are 1:1 (already verified by data_loader);
the line number = the message id.
npz key = short_name(basename) (consistent with the bert_feats / predicted-edge caches).
Output: data/time_feats.npz + data/time_feats_meta.json
"""

import os
import sys
import json
import re
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_reproduction import DATA_ROOT, short_name  # noqa
from digat import load_split_paths  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIME_RE = re.compile(r"^\[(\d{2}):(\d{2})\]")


def file_times(prefix: str):
    """Returns (times_minutes list, n_lines); a missing timestamp is None."""
    path = prefix + ".ascii.txt"
    times = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _TIME_RE.match(line)
        times.append(int(m.group(1)) * 60 + int(m.group(2)) if m else None)
    return times


def feats_from_times(times, active_hours):
    n = len(times)
    feats = np.zeros((n, 2), dtype=np.float32)
    prev = None
    for k, tm in enumerate(times):
        if tm is None:
            feats[k] = (0.0, 1.0)
            continue
        if prev is not None:
            d = tm - prev
            if d < 0:
                d += 1440
            feats[k, 0] = np.log1p(d)
        feats[k, 1] = 1.0 if (tm // 60) in active_hours else 0.0
        prev = tm
    return feats


def main():
    split = load_split_paths()
    # a split entry looks like "train/2004-12-25.train-c" (it contains the original subdirectory)
    all_names = [n for n in split["train"]]
    # the active-hour histogram only uses the train pool
    hist = np.zeros(24, dtype=np.int64)
    for name in all_names:
        for tm in file_times(os.path.join(DATA_ROOT, name)):
            if tm is not None:
                hist[tm // 60] += 1
    med = float(np.median(hist))
    active_hours = set(int(h) for h in range(24) if hist[h] >= med)
    print("hour histogram:", hist.tolist())
    print("active hours (>=median {:.0f}):".format(med), sorted(active_hours))

    out = {}
    n_missing = 0
    for split_name in ("train", "dev", "test"):
        for name in split[split_name]:
            times = file_times(os.path.join(DATA_ROOT, name))
            n_missing += sum(1 for t in times if t is None)
            out[name.split("/")[-1].split(".")[0]] = feats_from_times(times, active_hours)

    np.savez_compressed(os.path.join(ROOT, "data", "time_feats.npz"), **out)
    meta = {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_files": len(out),
        "n_messages_missing_time": int(n_missing),
        "hour_histogram_train": hist.tolist(),
        "active_hours": sorted(active_hours),
        "active_rule": "hour message count (train pool) >= median",
        "features": ["log1p(gap_minutes_to_previous)", "active_flag"],
    }
    with open(os.path.join(ROOT, "data", "time_feats_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved: {} files, missing-time msgs: {}".format(len(out), n_missing))
    print("completed_at:", meta["completed_at"])


if __name__ == "__main__":
    main()
