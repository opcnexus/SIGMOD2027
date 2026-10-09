#!/usr/bin/env python
"""06 Write per-item results to disk (not aggregate statistics) -- step 6 of the DAG.

For every (method x dataset x instance x node) it emits:
  instance_id / node_idx / text / gold parent / predicted parent / gold chain / predicted chain / edge correct / chain exact match

Outputs:
  results/per_item/<method>__<dataset>.jsonl   per-item details (machine readable)
  results/per_item/<method>__<dataset>.md      per-item details (human readable table)
  results/per_item/ALL_per_item.csv            merged CSV of everything
"""
import csv
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] if "experiments" in p.parts else Path(__file__).resolve().parent))

from datasets.base import from_dict  # noqa: E402
from methods.base import chain_from  # noqa: E402
from methods.registry import ALL  # noqa: E402
from utils import load_json, log, run_tag  # noqa: E402


def fmt(chain):
    return "->".join(str(x) for x in chain) if chain else "(empty)"


def main():
    cfg = load_json(ROOT / "config" / "config.json")
    backend = cfg["llm"]["backend"]
    tag = run_tag(cfg)
    pred_dir = ROOT / "results" / "predictions"
    out_dir = ROOT / "results" / "per_item"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("STEP 06 dump per-item results (backend=%s)" % tag)

    all_rows = []
    for M in ALL:
        for ds in cfg["datasets"]:
            f = pred_dir / ("%s__%s__%s.json" % (ds, M.name, tag))
            if not f.exists():
                log("[06][WARN] missing %s" % f.name)
                continue
            obj = load_json(f)
            rows = []
            for rec in obj["preds"]:
                gp = {int(k): v for k, v in rec["gold_parent"].items()}
                pp = {int(k): v for k, v in rec["pred_parent"].items()}
                texts = {it["idx"]: it["text"] for it in rec["items"]}
                # prediction records only store idx/text, so add ts here (equal to idx, since idx order is time order)
                items_full = [{"idx": it["idx"], "text": it["text"], "ts": float(it["idx"])}
                              for it in rec["items"]]
                inst = from_dict({"dataset": ds, "instance_id": rec["instance_id"],
                                  "items": items_full, "gold_parent": rec["gold_parent"]})
                for i in range(inst.n):
                    g_parent = gp.get(i, -1)
                    p_parent = pp.get(i, -1)
                    g_chain = inst.gold_chain(i)
                    p_chain = chain_from(inst, pp, i)
                    rows.append({
                        "dataset": ds, "method": M.name, "style": M.style,
                        "instance_id": rec["instance_id"], "node_idx": i,
                        "text": texts.get(i, ""),
                        "gold_parent": g_parent, "pred_parent": p_parent,
                        "gold_chain": fmt(g_chain), "pred_chain": fmt(p_chain),
                        "edge_correct": int(g_parent == p_parent),
                        "chain_exact": int(p_chain == g_chain),
                        "chain_len": len(g_chain),
                    })
            all_rows.extend(rows)

            # per-item JSONL
            with open(out_dir / ("%s__%s.jsonl" % (M.name, ds)), "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")
            # per-item Markdown
            md = ["# Per-item results: %s x %s\n" % (M.name, ds),
                  "| Instance | Node | Text | Gold parent | Pred. parent | Edge OK | Gold chain | Pred. chain | Chain OK |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                md.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["instance_id"], r["node_idx"], r["text"][:48],
                    r["gold_parent"], r["pred_parent"], "✓" if r["edge_correct"] else "✗",
                    r["gold_chain"], r["pred_chain"], "✓" if r["chain_exact"] else "✗"))
            (out_dir / ("%s__%s.md" % (M.name, ds))).write_text("\n".join(md), encoding="utf-8")
            log("[06] %-22s %-20s rows=%d" % (M.name, ds, len(rows)))

    # merged CSV
    if all_rows:
        with open(out_dir / "ALL_per_item.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    log("[06] DONE -> %s | total rows=%d" % (out_dir, len(all_rows)))


if __name__ == "__main__":
    main()
