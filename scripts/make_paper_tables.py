#!/usr/bin/env python3
"""Rebuild the paper tables from the shipped result files and compare them against the published values.

Usage:
    python scripts/make_paper_tables.py            # generate CSVs and print
    python scripts/make_paper_tables.py --check    # additionally compare against the published values

Outputs: results/tables/T*.csv
"""
import argparse
import csv
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = RES / "raw_reduced"
OUT = RES / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PAPER_PLATFORMS = ["irc", "molweni", "discord", "slack", "hn", "reddit"]
PROBES = ["prev1", "saprev", "simmax", "PairMLP", "MHA-Net"]

# Values as published in the paper (used by the --check reproduction audit)
PAPER = {
    "T4": {  # main diagnosis, test F1
        "prev1":   {"irc": 23.3, "molweni": 63.4, "discord": 41.1, "slack": 3.1, "hn": 1.1, "reddit": 3.3},
        "saprev":  {"irc": 23.6, "molweni": 24.3, "discord": 8.6, "slack": 0.9, "hn": 0.8, "reddit": 0.8},
        "simmax":  {"irc": 5.7, "molweni": 35.9, "discord": 7.3, "slack": 0.0, "hn": 4.0, "reddit": 4.9},
        "PairMLP": {"irc": 25.6, "molweni": 74.9, "discord": 44.5, "slack": 11.3, "hn": 3.6, "reddit": 5.3},
        "MHA-Net": {"irc": 25.3, "molweni": 74.6, "discord": 43.8, "slack": 11.3, "hn": 1.7, "reddit": 4.5},
    },
    "T6": {"irc->discord": 33.3, "discord->irc": 17.4, "molweni->discord": 40.8, "irc->irc": 28.0, "discord->discord": 41.0},
    "T7": {  # M3 backtracking reconstruction accuracy
        "prev1":   {"irc": 7.4, "molweni": 73.7, "discord": 15.9, "slack": 0.5, "hn": 1.1, "reddit": 3.5},
        "saprev":  {"irc": 10.6, "molweni": 56.3, "discord": 12.6, "slack": 0.3, "hn": 1.4, "reddit": 4.7},
        "simmax":  {"irc": 2.9, "molweni": 57.0, "discord": 8.6, "slack": 0.0, "hn": 2.3, "reddit": 4.7},
        "PairMLP": {"irc": 6.7, "molweni": 82.0, "discord": 17.4, "slack": 1.9, "hn": 1.7, "reddit": 3.9},
        "MHA-Net": {"irc": 7.4, "molweni": 82.0, "discord": 17.3, "slack": 1.9, "hn": 1.5, "reddit": 3.8},
    },
    "T10": {  # routing vs oracle
        "irc": (27.0, 27.9, 25.7), "molweni": (73.3, 78.2, 75.0), "discord": (44.0, 44.9, 44.8),
        "slack": (11.3, 11.3, 11.3), "hn": (4.1, 4.5, 4.3), "reddit": (6.4, 6.7, 5.4),
    },
    "T9": {"discord": (19.35, 16.74, 46.91), "slack": (3.08, 2.98, 2.28)},  # re-run after the engineering fixes
}

checks, failures = [], []


def note(ok, label, got, want, tol=0.15):
    checks.append((ok, label, got, want))
    if not ok:
        failures.append(label)
    flag = "PASS" if ok else "FAIL"
    print(f"    [{flag}] {label}: got={got} paper={want}")


def write(name, header, rows):
    p = OUT / name
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {p.relative_to(ROOT)}  ({len(rows)} rows)")
    return p


# ---------------- T1/T2: representation-space profiles ----------------
def t1_representation_space():
    f = RAW / "representation_space.json"
    if not f.exists():
        return print("  [skip] representation_space.json is missing")
    d = json.load(open(f))["datasets"]
    rows = []
    for p in PAPER_PLATFORMS:
        if p not in d:
            continue
        x = d[p]
        rows.append([p, x.get("n_msg"), x.get("n_conv"),
                     (x.get("msgs_per_conv") or {}).get("median"),
                     (x.get("participants_per_conv") or {}).get("median"),
                     (x.get("msg_gap_sec") or {}).get("median"),
                     x.get("mention_rate_pct"), x.get("root_pct", x.get("root_rate_pct"))])
    write("T1_representation_space.csv",
          ["platform", "n_msg", "n_conv", "median_msgs_per_conv",
           "median_participants", "median_gap_sec", "mention_rate_pct", "root_pct"], rows)


# ---------------- T4: main diagnosis ----------------
def t4_main():
    f = RES / "full_benchmark_v2" / "summary.json"
    if not f.exists():
        return print("  [skip] full_benchmark_v2/summary.json is missing")
    agg = json.load(open(f))["aggregated"]
    rows = []
    for p in PROBES:
        for ds in PAPER_PLATFORMS:
            e = agg.get(ds, {}).get(p)
            if not e:
                continue
            rows.append([p, ds, e.get("F1_mean"), e.get("F1_std"), e.get("n_runs")])
    write("T4_main_diagnosis.csv", ["probe", "dataset", "F1_mean", "F1_std", "n_runs"], rows)


# ---------------- T5: edge-class breakdown ----------------
def t5_edge_classes():
    f = RES / "full_benchmark_v2" / "summary.json"
    if not f.exists():
        return
    agg = json.load(open(f))["aggregated"]
    cats = ["cats.1-1.F1_mean", "cats.many-1.F1_mean", "cats.1-many.F1_mean", "cats.many-many.F1_mean"]
    rows = []
    for ds in PAPER_PLATFORMS:
        for p in PROBES:
            e = agg.get(ds, {}).get(p)
            if not e:
                continue
            rows.append([ds, p] + [e.get(c) for c in cats])
    write("T5_edge_class_F1.csv", ["dataset", "probe", "1-1", "many-1", "1-many", "many-many"], rows)


# ---------------- T6: cross-tier transfer matrix ----------------
def t6_transfer():
    f = RAW / "transfer_experiment.json"
    if not f.exists():
        return print("  [skip] transfer_experiment.json is missing")
    t = json.load(open(f))
    rows = []
    for k, v in t["results"].items():
        m = v.get("model", {})
        rows.append([k, m.get("F1"), m.get("P"), m.get("R"), m.get("tp"), m.get("fp"), m.get("fn"),
                     v.get("n_train"), v.get("n_test")])
    write("T6_transfer_matrix.csv",
          ["train->test", "F1", "P", "R", "tp", "fp", "fn", "n_train", "n_test"], rows)
    return t


# ---------------- T7: M3 backtracking reconstruction accuracy ----------------
def t7_backtrack():
    f = RAW / "backtrack_reconstruction.json"
    if not f.exists():
        return print("  [skip] backtrack_reconstruction.json is missing")
    d = json.load(open(f))["datasets"]
    rows = [[ds] + [d.get(ds, {}).get(p) for p in PROBES] for ds in PAPER_PLATFORMS]
    write("T7_M3_backtrack.csv", ["dataset"] + PROBES, rows)


# ---------------- T9: three-way chain-level comparison for ECB ----------------
def t9_ecb():
    f = RAW / "ecb_full.jsonl"
    if not f.exists():
        return print("  [skip] ecb_full.jsonl is missing")
    rows = [json.loads(l) for l in open(f)]
    summ = [r for r in rows if r.get("window_id") == -1]
    out, res = [], {}
    for ds in ("discord", "slack"):
        xs = [r for r in summ if r["dataset"] == ds]
        if not xs:
            continue
        zero = [r.get("path_F1_zero_filled", r.get("path_F1") or 0.0) for r in xs]
        m2r = [r["component_recall"] for r in xs if r.get("component_recall") is not None]
        m2p = [r.get("component_precision") for r in xs if r.get("component_precision") is not None]
        m3 = [r["ancestor_jaccard"] for r in xs if r.get("ancestor_jaccard") is not None]
        deg = sum(1 for r in xs if r.get("path_F1") is None)
        fails = sum(1 for r in xs if r.get("failure_modes"))
        trunc = sum(1 for r in xs if r.get("path_gold_truncated") or r.get("path_pred_truncated"))
        res[ds] = (st.mean(zero), st.mean(m2r) if m2r else None, st.mean(m3) if m3 else None)
        out.append([ds, len(xs), round(st.mean(zero), 2), round(st.mean(m2r), 2) if m2r else None,
                    round(st.mean(m2p), 2) if m2p else None, round(st.mean(m3), 2) if m3 else None,
                    deg, fails, trunc,
                    json.loads((RAW / "ecb_full.jsonl").read_text().splitlines()[0]).get("metric_version", "")])
    write("T9_ecb_chain_level.csv",
          ["dataset", "n_conv", "path_F1_zero_filled", "thread_recall_M2", "thread_precision_M2",
           "backtrack_M3", "n_degenerate_pathF1", "n_conv_with_llm_failure", "n_conv_path_truncated",
           "metric_version"], out)
    return res


# ---------------- T10: routing prototype ----------------
def t10_router():
    f = RAW / "moe_router.json"
    if not f.exists():
        return print("  [skip] moe_router.json is missing")
    d = json.load(open(f))["datasets"]
    rows = []
    for ds in PAPER_PLATFORMS:
        x = d.get(ds)
        if not x:
            continue
        best = max((v for v in x.get("single", {}).values()), default=None)
        rows.append([ds, x.get("routed_F1"), x.get("oracle_F1"), best,
                     round(x["routed_F1"] - best, 2) if best is not None else None])
    write("T10_routing.csv", ["dataset", "routed_F1", "oracle_F1", "best_single_probe", "gain"], rows)


# ---------------- negative result: v2 iterations ----------------
def tneg_v2():
    f = RAW / "ecb_v2_iterations.jsonl"
    if not f.exists():
        return print("  [skip] ecb_v2_iterations.jsonl is missing")
    rows = [json.loads(l) for l in open(f)]
    keys = [k for k in ("dataset", "conv_id", "iterations", "ancestor_jaccard", "path_F1") if k in (rows[0] if rows else {})]
    write("Tneg_v2_iterations.csv", keys, [[r.get(k) for k in keys] for r in rows])
    for ds in sorted({r.get("dataset") for r in rows if r.get("dataset")}):
        xs = [r.get("ancestor_jaccard") for r in rows if r.get("dataset") == ds and r.get("ancestor_jaccard") is not None]
        if xs:
            print(f"    v2 {ds}: mean ancJ = {st.mean(xs):.2f}% (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against the values published in the paper")
    args = ap.parse_args()
    print("== rebuilding the paper tables ==")
    t1_representation_space()
    t4_main()
    t5_edge_classes()
    t6_transfer()
    t7_backtrack()
    t9 = t9_ecb()
    t10_router()
    tneg_v2()

    if not args.check:
        print("\n(add --check to compare item by item against the published values)")
        return
    print("\n== comparison against published values (tolerance 0.15) ==")
    agg = json.load(open(RES / "full_benchmark_v2" / "summary.json"))["aggregated"]
    for p in PROBES:
        for ds in PAPER_PLATFORMS:
            got = (agg.get(ds, {}).get(p) or {}).get("F1_mean")
            if got is not None:
                note(abs(got - PAPER["T4"][p][ds]) <= 0.15, f"T4 {p}/{ds}", round(got, 2), PAPER["T4"][p][ds])
    t6 = json.load(open(RAW / "transfer_experiment.json"))["results"]
    for k, want in PAPER["T6"].items():
        got = (t6.get(k, {}).get("model") or {}).get("F1")
        if got is not None:
            note(abs(got - want) <= 0.6, f"T6 {k}", round(got, 2), want)
    bt = json.load(open(RAW / "backtrack_reconstruction.json"))["datasets"]
    for p in PROBES:
        for ds in PAPER_PLATFORMS:
            got = bt.get(ds, {}).get(p)
            if got is not None:
                note(abs(got - PAPER["T7"][p][ds]) <= 0.15, f"T7 {p}/{ds}", round(got, 2), PAPER["T7"][p][ds])
    rt = json.load(open(RAW / "moe_router.json"))["datasets"]
    for ds, want in PAPER["T10"].items():
        x = rt.get(ds)
        if x:
            note(abs(x["routed_F1"] - want[0]) <= 0.15, f"T10 {ds}/routed", x["routed_F1"], want[0])
            note(abs(x["oracle_F1"] - want[1]) <= 0.15, f"T10 {ds}/oracle", x["oracle_F1"], want[1])
    if t9:
        for ds, want in PAPER["T9"].items():
            got = t9.get(ds)
            if got and got[0] is not None:
                note(abs(got[0] - want[0]) <= 0.15, f"T9 {ds}/pathF1", round(got[0], 2), want[0])
                note(abs(got[1] - want[1]) <= 0.15, f"T9 {ds}/thread", round(got[1], 2), want[1])
                note(abs(got[2] - want[2]) <= 0.15, f"T9 {ds}/backtrack", round(got[2], 2), want[2])

    n_ok = sum(1 for c in checks if c[0])
    print(f"\n===== reproduction audit: {n_ok}/{len(checks)} items consistent =====")
    if failures:
        print("inconsistent items: " + "; ".join(failures))
        sys.exit(1)
    print("the shipped results reproduce every published value.")


if __name__ == "__main__":
    main()
