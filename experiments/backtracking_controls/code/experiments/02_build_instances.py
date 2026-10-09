#!/usr/bin/env python
"""02 Build the unified chain instances -- step 2 of the DAG (split / gold derivation, manifest written).

Outputs: data/cache/instances.json + data/cache/instances_manifest.json
"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] if "experiments" in p.parts else Path(__file__).resolve().parent))

from datasets.base import from_dict  # noqa: E402
from utils import load_json, log, manifest, save_json  # noqa: E402


def build():
    cfg = load_json(ROOT / "config" / "config.json")
    out = {}
    for ds in cfg["datasets"]:
        raw = load_json(ROOT / "data" / "small" / ("%s.json" % ds))
        insts = [from_dict(x) for x in raw]
        out[ds] = [i.to_dict() for i in insts]  # normalize before writing to disk
    return out


def main():
    log("=" * 60)
    log("STEP 02 build unified chain instances (cache-first)")
    cache = ROOT / "data" / "cache" / "instances.json"
    mf = ROOT / "data" / "small" / "manifest.json"
    # cache invalidation rule: if the data manifest is newer than the cache, rebuild (otherwise stale instances are reused)
    stale = False
    if cache.exists():
        if mf.exists() and mf.stat().st_mtime > cache.stat().st_mtime:
            stale = True
            log("[02] CACHE STALE (data manifest newer) -> rebuild")
    data = (load_json(cache) if (cache.exists() and not stale) else build())
    save_json(data, cache)

    total_inst = total_node = 0
    for ds, insts in data.items():
        nodes = sum(len(x["items"]) for x in insts)
        total_inst += len(insts)
        total_node += nodes
        log("[02] %-20s instances=%d nodes=%d" % (ds, len(insts), nodes))
    manifest(
        ROOT / "data" / "cache" / "instances_manifest.json",
        step="02_build_instances",
        n_instances=total_inst,
        n_nodes=total_node,
        note="gold_parent = 1-to-1 gold edges; gold_chain = their transitive closure (full-chain gold)",
    )
    log("[02] DONE total instances=%d nodes=%d" % (total_inst, total_node))


if __name__ == "__main__":
    main()
