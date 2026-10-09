#!/usr/bin/env python3
"""Full benchmark v2: resume support + a per-combo on-disk trace + multi-seed averaging of the neural algorithms.

Differences from v1 (full_benchmark.py) (hardened on 2026-09-26, after two crashes):
1. Every (dataset, algorithm, seed) combination is written separately to
   results/full_benchmark_v2/runs/*.json
   -- a crash / restart at any moment loses no completed combination, and the restarted script
   skips them automatically (resume support)
2. PairMLP / MHA-Net each run SEEDS seeds, and summary.json reports mean+/-std
   (the heuristics prev1/saprev/simmax are deterministic, a single run with s0)
3. Memory safety (this machine has only 16GB of unified memory, do not repeat the 15:04 crash):
   - datasets are loaded one by one -> run -> release (v1 kept all 6 datasets resident at once)
   - the MSG_VEC_CACHE cap drops from 2,000,000 to 400,000 (vectors + text keys take about 300-400MB)
   - after each combo: del + torch.mps.empty_cache() + gc
   - memory guardrail: pause and wait while free memory is <15%, continue once it recovers
4. The data split is fixed at seed=42 (exactly the same as v1, so the numbers are directly comparable);
   the seed only changes the model initialization and the training order
"""
import os, sys, json, time, gc, random, argparse, subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
import torch.nn as nn
import transfer_experiment as TE
import full_benchmark as FB

RUNS_DIR = os.path.join(ROOT, "results", "full_benchmark_v2", "runs")
LOG_DIR = os.path.join(ROOT, "results", "logs")
SEEDS = [42, 43, 44, 45, 46]
HEURISTICS = ["prev1", "saprev", "simmax"]
NEURAL = ["PairMLP", "MHA-Net"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def now_min():
    return time.strftime("%Y-%m-%d %H:%M")

def wait_for_memory(min_free_pct=15, max_wait_s=1800):
    """Pause while the free memory is below the threshold (a disk-writing job can afford to wait, it cannot afford to crash)."""
    waited = 0
    while waited < max_wait_s:
        try:
            outp = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                                  text=True, timeout=30).stdout
            pct = int(outp.strip().split(":")[-1].strip().rstrip("%")
                      .replace("System-wide memory free percentage:", "").strip() or 0)
        except Exception:
            return
        if pct >= min_free_pct:
            return
        log(f"  [mem-guard] free={pct}% < {min_free_pct}%, sleep 60s (waited {waited}s)")
        time.sleep(60)
        waited += 60

def free_mem():
    try:
        outp = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                              text=True, timeout=30).stdout
        return int(outp.strip().rstrip("%").rsplit(":", 1)[-1].strip())
    except Exception:
        return 100

def reseed(seed):
    """Reset all random sources (the data split is unaffected: the split was already fixed to 42 when TE was imported)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    TE.rng = np.random.RandomState(seed)

def ckpt_path(ds, algo, seed):
    return os.path.join(RUNS_DIR, "{}__{}__s{}.json".format(ds, algo, seed))

def save_ckpt(path, payload):
    tmp = path + ".tmp"
    json.dump(payload, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)   # atomic write, a crash never leaves a half-written file

def flatten_metrics(m):
    """Metrics dict -> a flat key: value mapping (None skipped), for aggregation."""
    flat = {}
    for k, v in m.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if isinstance(v3, (int, float)):
                            flat["{}.{}.{}".format(k, k2, k3)] = v3
                elif isinstance(v2, (int, float)):
                    flat["{}.{}".format(k, k2)] = v2
        elif isinstance(v, (int, float)):
            flat[k] = v
    return flat

def update_summary(meta):
    """Scan the runs directory and aggregate all completed combos -> summary.json (incremental rewrite)."""
    agg = defaultdict(lambda: defaultdict(list))
    for fn in sorted(os.listdir(RUNS_DIR)):
        if not fn.endswith(".json"):
            continue
        ds, algo, s = fn[:-5].split("__")
        d = json.load(open(os.path.join(RUNS_DIR, fn)))
        for k, v in flatten_metrics(d["metrics"]).items():
            agg[ds][algo + ":" + k].append(v)
    out = {"completed_at": now_min(), "seeds": meta["seeds"],
           "config": meta["config"], "aggregated": {}}
    for ds, kvs in sorted(agg.items()):
        out["aggregated"][ds] = {}
        pairs = defaultdict(dict)
        for full_k, vals in kvs.items():
            algo, k = full_k.split(":", 1)
            pairs[algo][k] = vals
        for algo, kv in sorted(pairs.items()):
            entry = {"n_runs": max(len(v) for v in kv.values()) if kv else 0}
            seed_f1 = None
            for k, vals in sorted(kv.items()):
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                entry[k + "_mean"] = round(float(np.mean(vals)), 2)
                if len(vals) > 1:
                    entry[k + "_std"] = round(float(np.std(vals)), 2)
                if k == "link.F1":
                    seed_f1 = sorted(vals)
            if seed_f1 is not None:
                entry["link_F1_by_run"] = seed_f1
            out["aggregated"][ds][algo] = entry
    json.dump(out, open(os.path.join(ROOT, "results", "full_benchmark_v2",
                                     "summary.json"), "w"),
              ensure_ascii=False, indent=1)

def load_dataset(name):
    if name == "irc":        return TE.load_irc()
    if name == "molweni":    return TE.load_molweni()
    if name == "discord":    return TE.split_convs(TE.load_discord())
    if name == "slack":      return FB.load_slack()
    if name == "hn":         return FB.load_hn()
    if name == "reddit":     return FB.load_reddit()
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only-ds", default=None, help="comma separated, e.g. molweni,discord")
    ap.add_argument("--only-algos", default=None)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    ds_order = (args.only_ds.split(",") if args.only_ds
                else ["irc", "molweni", "discord", "slack", "hn", "reddit"])
    algos = (args.only_algos.split(",") if args.only_algos
             else HEURISTICS + NEURAL)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    meta = {"seeds": seeds, "algorithms": algos,
            "config": {"W": FB.W, "max_msgs": FB.MAX_MSGS, "epochs": FB.EPOCHS,
                       "patience": FB.PATIENCE, "split_seed": TE.SEED,
                       "model_seeds": seeds,
                       "protocol": "identical to full_benchmark v1; per-combo disk ckpt; neural = mean over seeds"}}
    json.dump(meta, open(os.path.join(ROOT, "results", "full_benchmark_v2",
                                      "meta.json"), "w"), ensure_ascii=False, indent=1)

    t0 = time.time()
    log(f"v2 start: ds={ds_order} algos={algos} seeds={seeds}")
    log(f"device={TE.DEVICE} free_mem={free_mem()}%")

    global msg_vec_global
    hasher = TE.Hasher(); hasher.fit(None)
    # v2: the vector cache cap is 400,000 for memory safety, v1 used 2,000,000
    cache = {}
    def msg_vec_global(m, _c=cache):
        t = m["text"]
        v = _c.get(t)
        if v is None:
            v = hasher.msg_vec(t)
            if len(_c) < 400_000:
                _c[t] = v
        return v
    FB.msg_vec_global = msg_vec_global   # used internally by FB.compute_metrics

    done = 0
    for name in ds_order:
        # skip the whole dataset: all combos already have a checkpoint
        combos = []
        for a in algos:
            if a in HEURISTICS:
                combos.append((a, 0))
            else:
                combos.extend((a, s) for s in seeds)
        missing = [c for c in combos if not os.path.exists(ckpt_path(name, c[0], c[1]))]
        if not missing:
            log(f"[skip] {name}: all {len(combos)} combos already on disk")
            continue

        log(f"=== dataset {name} === (missing {len(missing)}/{len(combos)} combos)")
        wait_for_memory()
        splits = load_dataset(name)
        test, dev, train = splits["test"], splits["dev"], splits["train"]
        log(f"  train={len(train)} dev={len(dev)} test={len(test)}")

        for algo, seed in combos:
            ck = ckpt_path(name, algo, seed)
            if os.path.exists(ck):
                continue
            wait_for_memory()
            tag = f"{name}-{algo}-s{seed}"
            if algo in HEURISTICS:
                maker = FB.make_heuristics()[algo]
                m = FB.compute_metrics(test, maker)
            elif algo == "PairMLP":
                reseed(seed)
                model = FB.train_pairmlp(train, dev, tag)
                m = FB.compute_metrics(test, FB.pairmlp_make(model))
                del model
            elif algo == "MHA-Net":
                reseed(seed)
                model = TE.train_model(train, dev, msg_vec_global, tag)
                m = FB.compute_metrics(test, FB.mhanet_make(model))
                del model
            else:
                continue
            payload = {"dataset": name, "algorithm": algo, "seed": seed,
                       "metrics": m, "saved_at": now_min(),
                       "runtime_seconds_total": round(time.time() - t0)}
            save_ckpt(ck, payload)
            done += 1
            log(f"  [ckpt] {tag}: link F1={m['link']['F1']} root F1={m['root']['F1']} "
                f"(total {round(time.time()-t0)}s)")
            update_summary(meta)
            if TE.DEVICE == "mps":
                torch.mps.empty_cache()
            gc.collect()

        # release at the dataset level (v1 keeping all datasets resident was the memory risk)
        del splits, test, dev, train
        cache.clear()
        gc.collect()
        if TE.DEVICE == "mps":
            torch.mps.empty_cache()
        log(f"=== dataset {name} done, mem released (free={free_mem()}%) ===")

    update_summary(meta)
    log(f"ALL_DONE in {round(time.time()-t0)}s, {done} new checkpoints")
    log(f"summary -> results/full_benchmark_v2/summary.json")

if __name__ == "__main__":
    main()
