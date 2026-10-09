#!/usr/bin/env python3
"""P1-8: a cross-tier transfer correction experiment with latency-quantile normalization (a controlled
validation of the J2 attribution).

Hypothesis: the main cause of the asymmetric J2 transfer is the cross-tier drift of the parent-child
latency distribution (1-2 orders of magnitude).
Test: replace the absolute latency tgap in pair_features with the **ECDF quantile rank of the
source-domain training set**,
and map the latencies of the target-domain test edges into the same quantile space through the
source-domain ECDF, then re-run both transfer directions:
  - discord->irc (baseline -38% degradation)
  - irc->discord (baseline -19% degradation)
If the degradation narrows significantly after normalization, then "latency distribution drift" is
promoted from a conjecture to a controlled, validated attribution.

The baseline numbers are read from results/transfer_experiment.json (do not re-run the baselines).
Memory safety: only 2 datasets stay resident; run under nice.
Output: results/transfer_norm.json
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import transfer_experiment as TE

RESULTS = os.path.join(ROOT, "results")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def train_latencies(convs):
    """The latencies (in seconds) of all edges of the training set that have a ts, used to build the
source-domain ECDF."""
    vals = []
    for conv in convs:
        msgs = conv["msgs"]
        id2t = {m["id"]: (m.get("ts") or 0) for m in msgs}
        for m in msgs:
            t = m.get("ts") or 0
            if not t:
                continue
            for p in m["parents"]:
                pt = id2t.get(p, 0)
                if pt and t > pt:
                    vals.append(t - pt)
    return np.sort(np.array(vals, dtype=np.float64)) if vals else np.array([0.0])

def make_norm_pair_features(ecdf):
    """Replaces TE.pair_features: tgap (absolute log latency) -> the source-domain ECDF quantile rank [0,1]."""
    def pair_features_norm(msgs, i, c, id2pos):
        m, cnd = msgs[i], msgs[c]
        same_author = 1.0 if m["author"] and m["author"] == cnd["author"] else 0.0
        gap = float(np.log1p(i - c))
        t_c, t_p = (m.get("ts") or 0), (cnd.get("ts") or 0)
        if t_c and t_p and t_c > t_p:
            lat = t_c - t_p
            tgap = float(np.searchsorted(ecdf, lat, side="right") / len(ecdf))
            has_ts = 1.0
        else:
            tgap, has_ts = 0.0, 0.0
        dl = float(np.log1p(abs(len(m["text"]) - len(cnd["text"]))))
        inter = float(np.log1p(i - c - 1))
        return [same_author, gap, tgap, has_ts, dl, inter]
    return pair_features_norm

def read_baseline():
    with open(os.path.join(RESULTS, "transfer_experiment.json")) as f:
        d = json.load(f)
    res = d.get("results", d)
    def model_f1(key):
        try:
            return float(res[key]["model"]["F1"])
        except Exception:
            return None
    out = {"discord->irc": model_f1("discord->irc"),
           "irc->discord": model_f1("irc->discord"),
           "irc->irc": model_f1("irc->irc"),
           "discord->discord": model_f1("discord->discord")}
    return out

def run_direction(src_name, tgt_name, src_load, tgt_load, ecdf_src_train):
    log(f"=== {src_name} -> {tgt_name} (latency quantile-normalized) ===")
    TE.pair_features = make_norm_pair_features(ecdf_src_train)
    src = src_load(); tgt = tgt_load()
    hasher = TE.Hasher(); hasher.fit(None)
    mvec = lambda m: hasher.msg_vec(m["text"])
    model = TE.train_model(src["train"], src["dev"], mvec,
                           f"{src_name}2{tgt_name}-norm")
    m = TE.eval_convs(model, tgt["test"], mvec)
    log(f"  {src_name}->{tgt_name} norm test F1={m['F1']}")
    del src, tgt, model
    if TE.DEVICE == "mps":
        import torch; torch.mps.empty_cache()
    return m

def main():
    t0 = time.time()
    base = read_baseline()
    log(f"baseline numbers found: {base}")

    discord = TE.split_convs(TE.load_discord())
    irc = TE.load_irc()
    ecdf_discord = train_latencies(discord["train"])
    ecdf_irc = train_latencies(irc["train"])
    log(f"ECDF sizes: discord={len(ecdf_discord)} irc={len(ecdf_irc)}")
    # release one copy, run the first direction
    del irc

    # direction 1: discord -> irc (using the discord ECDF)
    m1 = run_direction("discord", "irc",
                       lambda: TE.split_convs(TE.load_discord()), TE.load_irc,
                       ecdf_discord)

    # direction 2: irc -> discord (using the irc ECDF)
    m2 = run_direction("irc", "discord", TE.load_irc,
                       lambda: TE.split_convs(TE.load_discord()), ecdf_irc)

    out = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M"),
        "hypothesis": "J2 asymmetry caused by latency distribution shift; "
                      "fix = source-ECDF quantile normalization of tgap",
        "baseline_source": "results/transfer_experiment.json (single seed 42)",
        "discord2irc": {"norm_F1": m1["F1"], "baseline_F1": base.get("discord->irc") or base.get("discord2irc"),
                        "in_domain_irc_F1": 28.0},
        "irc2discord": {"norm_F1": m2["F1"], "baseline_F1": base.get("irc->discord") or base.get("irc2discord"),
                        "in_domain_discord_F1": 41.0},
        "runtime_seconds": round(time.time() - t0),
    }
    for k in ("discord2irc", "irc2discord"):
        b, n, ind = out[k]["baseline_F1"], out[k]["norm_F1"], out[k]["in_domain_irc_F1"] if k == "discord2irc" else out[k]["in_domain_discord_F1"]
        if b:
            out[k]["baseline_decay_pct"] = round(100 * (b - ind) / ind, 1)
            out[k]["norm_decay_pct"] = round(100 * (n - ind) / ind, 1)
    with open(os.path.join(RESULTS, "transfer_norm.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"saved -> results/transfer_norm.json | done in {round(time.time()-t0)}s")
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
