#!/usr/bin/env python
"""01 Prepare the small-batch data (five datasets) -- step 1 of the DAG.

Outputs: data/small/<dataset>.json + data/small/manifest.json
Note: locally the pipeline is validated with synthetic data; real data is fetched on the server by datasets/fetch_scripts/
      and placed under data/raw/, after which the same channel in this script performs the split (see "Wiring in real data" in the README).
"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] if "experiments" in p.parts else Path(__file__).resolve().parent))

from datasets.registry import REGISTRY, gen_small  # noqa: E402
from utils import load_json, log, manifest, save_json  # noqa: E402


def main():
    cfg = load_json(ROOT / "config" / "config.json")
    out_dir = ROOT / "data" / "small"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("STEP 01 prepare small-batch data | seed=%s" % cfg["seed"])
    data = gen_small(cfg)

    stats = {}
    for ds, insts in data.items():
        save_json(insts, out_dir / ("%s.json" % ds))
        n_nodes = sum(len(x["items"]) for x in insts)
        lens = []
        for x in insts:
            gp = {int(k): v for k, v in x["gold_parent"].items()}
            for i in range(len(x["items"])):
                L, cur = 1, i
                while gp.get(cur, -1) != -1:
                    cur = gp[cur]
                    L += 1
                lens.append(L)
        stats[ds] = {
            "n_instances": len(insts),
            "n_nodes": n_nodes,
            "chain_len_min": min(lens),
            "chain_len_max": max(lens),
            "chain_len_mean": round(sum(lens) / len(lens), 2),
        }
        log(
            "[01] %-20s instances=%d nodes=%d chain_len(mean/min/max)=%.2f/%d/%d"
            % (ds, len(insts), n_nodes, stats[ds]["chain_len_mean"], min(lens), max(lens))
        )

    manifest(
        out_dir / "manifest.json",
        step="01_prepare_data",
        seed=cfg["seed"],
        source="synthetic(local validation)",
        datasets={d: REGISTRY[d]["full_data"] for d in cfg["datasets"]},
        stats=stats,
    )
    save_json(stats, ROOT / "results" / "step01_data_stats.json")
    log("[01] DONE -> %s" % out_dir)


if __name__ == "__main__":
    main()
