"""Shared utilities: logging (console plus on-disk, two-way traceability), JSON I/O, and cache-first operation (load_or_run).

Follows the Experiment Provenance Harness (Section 21):
  IR2 every step is written to disk and printed, giving two-way traceability
  IR3 cache-first; every cache hit is printed
"""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for _d in (CACHE, RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


def log(msg, also_print=True):
    """IR2: print to stdout and to logs/run.log simultaneously, ensuring two-way traceability."""
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    if also_print:
        print(line, flush=True)
    with open(LOGS / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_run(cache_file, compute_fn, desc=""):
    """IR3: on a cache hit, load directly and print the manifest; otherwise compute and write to disk."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        log("[CACHE HIT ] %s -> %s" % (desc or cache_file.name, cache_file))
        return load_json(cache_file)
    log("[CACHE MISS] %s -> computing..." % (desc or cache_file.name))
    obj = compute_fn()
    save_json(obj, cache_file)
    log("[CACHE SAVE] %s -> %s" % (desc or cache_file.name, cache_file))
    return obj


def tag_of(cfg):
    """Run tag (used to name prediction files and to isolate caches).

    It must include the temperature: at temperature 0 repeated calls return identical results, so "majority-vote / self-consistency" methods degenerate to a single call,
    and results from different temperatures must not be mixed, hence the tag.
    """
    b = cfg.get("llm", {}).get("backend", "mock")
    if b == "mock":
        return "mock-p%.2f" % cfg.get("llm", {}).get("mock_edge_accuracy", 0.9)
    t = cfg.get("llm", {}).get("temperature", 0)
    return "api-T%s" % str(t).replace(".", "")


def manifest(path, **kv):
    """Write a split/run manifest (IR2: anything written to disk must carry a manifest)."""
    rec = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), **kv}
    save_json(rec, path)
    log("[MANIFEST] %s : %s" % (Path(path).name, rec))
    return rec


def data_fingerprint():
    """Data fingerprint: a short hash of the contents of instances.json.

    It puts the "data version" into the prediction cache key -- otherwise, after changing the data (instance count / chain length),
    step 03 hits predictions computed on the old data and yields plausible-looking but wrong results (this project has hit that twice).
    """
    p = ROOT / "data" / "cache" / "instances.json"
    if not p.exists():
        return "nodata"
    import hashlib
    return hashlib.sha1(p.read_bytes()).hexdigest()[:8]


def run_tag(cfg):
    """The complete run tag = backend tag + data fingerprint (a change in either forces a recomputation)."""
    return "%s-%s" % (tag_of(cfg), data_fingerprint())
