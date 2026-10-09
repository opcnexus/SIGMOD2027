#!/usr/bin/env python3
"""Structural statistics of the Molweni (HIT-SCIR, ACL 2020) DP data, compared with Ubuntu IRC.

Format: each dialog = {edus: [{text, speaker}], relations: [{x, y, type}]}
x = the party being replied to (earlier), y = the replying party; type = the discourse relation
(QAP/Elaboration/...)
root = an EDU without an incoming edge. Output results/molweni_structure.json
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOL = os.path.join(ROOT, "data", "molweni")


def main():
    out = {"completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "note": "EDU-level annotation (Molweni DP); x=parent y=child",
           "relation_type_hist": {}, "splits": {}}
    for f in ("train", "dev", "test"):
        path = os.path.join(MOL, "DP_{}.json".format(f))
        if not os.path.exists(path):
            out["splits"][f] = "FILE_MISSING"
            continue
        data = json.load(open(path))
        n_msg = n_root = n_single = n_multi = 0
        gap_hist = {}
        multi_ex = []
        for conv in data:
            n_edu = len(conv["edus"])
            parents = {k: [] for k in range(n_edu)}
            for rel in conv["relations"]:
                x, y = rel["x"], rel["y"]
                if 0 <= x < n_edu and 0 <= y < n_edu:
                    parents[y].append(x)
                    out["relation_type_hist"][rel["type"]] = \
                        out["relation_type_hist"].get(rel["type"], 0) + 1
            for k in range(n_edu):
                n_msg += 1
                ps = parents[k]
                if not ps:
                    n_root += 1
                elif len(ps) == 1:
                    n_single += 1
                else:
                    n_multi += 1
                    if len(multi_ex) < 5:
                        multi_ex.append({"dialog": conv.get("id"), "edu": k,
                                         "parents": ps})
                # the positional gap to the previous EDU (the structural distance approximates a time gap)
                if ps:
                    d = k - min(ps)
                    gap_hist[min(d, 20)] = gap_hist.get(min(d, 20), 0) + 1
        tot = max(1, n_root + n_single + n_multi)
        out["splits"][f] = {
            "n_dialogs": len(data), "n_edus": n_msg,
            "root": n_root, "single_parent": n_single, "multi_parent": n_multi,
            "root_pct": round(100 * n_root / tot, 2),
            "single_pct": round(100 * n_single / tot, 2),
            "multi_pct": round(100 * n_multi / tot, 2),
            "parent_distance_hist_capped20": gap_hist,
            "multi_examples": multi_ex,
        }
        print(f, "root={} single={} multi={} ({} edus)".format(
            n_root, n_single, n_multi, n_msg), flush=True)
    with open(os.path.join(ROOT, "results", "molweni_structure.json"), "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print("saved. completed_at:", out["completed_at"])


if __name__ == "__main__":
    main()
