#!/usr/bin/env python
"""03 Run the 10 methods -- step 3 of the DAG (cache-first + concurrency + per-record persistence).

Outputs: results/predictions/<dataset>__<method>__<backend>.json
      Each record holds the instance items / gold_parent / pred_parent (consumed by 06 to emit per-item details)
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] if "experiments" in p.parts else Path(__file__).resolve().parent))

from datasets.base import from_dict  # noqa: E402
from llm.factory import get_llm  # noqa: E402
from methods.registry import ALL  # noqa: E402
from utils import load_json, log, run_tag, load_or_run, save_json, run_tag  # noqa: E402


def run_one(ds, M, mi, cfg, inst_data, tag, pred_dir):
    cache_file = pred_dir / ("%s__%s__%s.json" % (ds, M.name, tag))

    def compute():
        llm = get_llm(cfg)
        rng = random.Random(cfg["seed"] + mi * 17 + 3)
        res = []
        for x in inst_data[ds]:
            inst = from_dict(x)
            pred = M().predict(inst, llm, rng)
            res.append({
                "instance_id": inst.instance_id,
                "items": [{"idx": it.idx, "text": it.text} for it in inst.items],
                "gold_parent": {str(k): v for k, v in inst.gold_parent.items()},
                "pred_parent": {str(k): v for k, v in pred.items()},
            })
        return {"method": M.name, "style": M.style, "dataset": ds,
                "llm": llm.describe(), "preds": res}

    obj = load_or_run(cache_file, compute, "%s/%s" % (ds, M.name))
    return (ds, M.name, obj)


def main():
    cfg = load_json(ROOT / "config" / "config.json")
    inst_data = load_json(ROOT / "data" / "cache" / "instances.json")
    backend = cfg["llm"]["backend"]
    tag = run_tag(cfg)
    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    workers = cfg["llm"].get("workers", 8)

    log("=" * 60)
    log("STEP 03 run %d methods x %d datasets | backend=%s | workers=%d"
        % (len(ALL), len(cfg["datasets"]), tag, workers))

    tasks = [(ds, M, mi) for mi, M in enumerate(ALL) for ds in cfg["datasets"]]
    summary = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, ds, M, mi, cfg, inst_data, tag, pred_dir): (ds, M.name)
                for ds, M, mi in tasks}
        for f in as_completed(futs):
            ds, name = futs[f]
            try:
                d, m, obj = f.result()
                summary["%s|%s" % (d, m)] = {"style": obj["style"], "llm": obj["llm"]}
                log("[03] %-22s %-20s %s" % (m, d, obj["llm"]))
            except Exception as e:  # noqa: BLE001
                log("[03][ERROR] %s/%s -> %s" % (ds, name, e))

    save_json(summary, ROOT / "results" / "step03_run_summary.json")
    log("[03] DONE -> %s (%d tasks)" % (pred_dir, len(tasks)))


if __name__ == "__main__":
    main()
