#!/usr/bin/env python
"""04 Compute the three metric layers -- step 4 of the DAG (L1 edge-level / L2 chain-level / L3 gap plus decay by chain length).

Outputs: results/metrics.json
"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] if "experiments" in p.parts else Path(__file__).resolve().parent))

from datasets.base import from_dict  # noqa: E402
from metrics import combine, length_stratified  # noqa: E402
from methods.registry import ALL  # noqa: E402
from utils import load_json, log, run_tag, save_json, run_tag  # noqa: E402


def main():
    cfg = load_json(ROOT / "config" / "config.json")
    inst_data = load_json(ROOT / "data" / "cache" / "instances.json")
    backend = cfg["llm"]["backend"]
    tag = run_tag(cfg)
    pred_dir = ROOT / "results" / "predictions"
    pk = tuple(cfg["eval"]["prefix_k"])
    buckets = [tuple(b) for b in cfg["eval"]["len_buckets"]]

    log("=" * 60)
    log("STEP 04 evaluate (L1 edge / L2 chain / L3 gap)")

    out = {}
    for M in ALL:
        for ds in cfg["datasets"]:
            f = pred_dir / ("%s__%s__%s.json" % (ds, M.name, tag))
            if not f.exists():
                log("[04][WARN] missing %s" % f.name)
                continue
            obj = load_json(f)
            insts = {i.instance_id: i for i in [from_dict(x) for x in inst_data[ds]]}
            rows = []
            for p in obj["preds"]:
                inst = insts[p["instance_id"]]
                pred = {int(k): v for k, v in p["pred_parent"].items()}
                rows.append(combine(inst, pred, prefix_k=pk))
            agg = {}
            for key in ("edge_precision", "edge_recall", "edge_f1", "edge_acc",
                        "chain_EM", "chain_f1", "gap"):
                agg[key] = round(sum(r[key] for r in rows) / len(rows), 4)
            agg["prefix_acc"] = {
                k: round(sum(r["prefix_acc"][k] for r in rows) / len(rows), 4) for k in pk
            }
            agg["len_stratified"] = length_stratified(rows, buckets)
            out["%s|%s" % (ds, M.name)] = {
                "dataset": ds, "method": M.name, "style": M.style,
                "paper": M.paper, **agg,
            }
            log(
                "[04] %-22s %-20s edgeF1=%.3f edgeAcc=%.3f chainEM=%.3f gap=%.3f"
                % (M.name, ds, agg["edge_f1"], agg["edge_acc"], agg["chain_EM"], agg["gap"])
            )

    # Two writes: (1) a tag-specific file so runs never overwrite each other and conditions can be compared,
    #          (2) metrics.json (the most recent run, kept for compatibility with 05)
    save_json(out, ROOT / "results" / ("metrics_%s.json" % tag))
    save_json(out, ROOT / "results" / "metrics.json")
    log("[04] DONE -> results/metrics_%s.json (%d rows)" % (tag, len(out)))


if __name__ == "__main__":
    main()
