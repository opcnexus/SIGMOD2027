#!/usr/bin/env python3
"""Generate the first-best predicted edges of the trained v2 DiGAT (GloVe) over all 80/20 files.

Purposes:
1. An honest evaluation of v3: the inference-time thread features must come from the predicted edges
   (reasoning over gold edges = label leakage, a confirmed bug).
2. Training the autoregressive thread construction: at training time the thread features come from the
   same prediction (train-inference consistency).

Note: the predictions on the train files come from a v2 that was trained on train -- this introduces an
optimistic bias,
which is acceptable as an edge source for thread features (an auxiliary feature); it is noted in the report.
Output: results/pred_links_v2.json {basename: [[src, tgt], ...]}
completed_at is precise to the minute.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_reproduction import DATA_ROOT, TEST_START, load_glove, short_name  # noqa
from digat import (RESULTS_DIR, load_split_paths, load_logs_for,
                   node_features, same_speaker_matrix, cand_index,
                   DiGAT, PairScorer, GRAPH_WINDOW, CAND_WINDOW)  # noqa


def predict_edges_all(model, scorer, log, emb, device, query_lo):
    """Emit the first-best edge for every message in [query_lo, n) (including self-links)."""
    model.eval(); scorer.eval()
    with torch.no_grad():
        x = node_features(log, emb)
        n = x.shape[0]
        x_t = torch.from_numpy(x).to(device)
        same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
        h = model(x_t, same_t)
        queries = list(range(query_lo, n))
        if not queries:
            return []
        qi, cj, valid = cand_index(n, queries)
        qi_t = torch.from_numpy(qi).to(device)
        cj_t = torch.from_numpy(cj).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        scores = scorer(h, qi_t, cj_t, same_t, n)
        scores = scores.masked_fill(~valid_t, -1e9)
        pred = scores.argmax(dim=1).cpu().numpy()
    return [(int(i), int(cj[r][pred[r]])) for r, i in enumerate(queries)]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    ckpt = torch.load(os.path.join(RESULTS_DIR, "digat_digat_glove.pt"),
                      map_location="cpu")
    in_dim, d_model = ckpt["in_dim"], ckpt["d_model"]
    graph = DiGAT(in_dim, d_model=d_model).to(device)
    graph.load_state_dict(ckpt["graph"])
    scorer = PairScorer(d_model=d_model).to(device)
    scorer.load_state_dict(ckpt["scorer"])
    print("v2 digat_glove loaded (in_dim={})".format(in_dim), flush=True)

    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))
    print("glove loaded", flush=True)

    split = load_split_paths()
    out = {}
    t0 = datetime.now()
    for name, logs, lo in (("train", load_logs_for(split["train"]), 1),
                           ("dev", load_logs_for(split["dev"]), TEST_START),
                           ("test", load_logs_for(split["test"]), TEST_START)):
        for k, log in enumerate(logs):
            edges = predict_edges_all(graph, scorer, log, emb, device, lo)
            out[short_name(log.name)] = edges
            if (k + 1) % 30 == 0:
                print("  {} {}/{}".format(name, k + 1, len(logs)), flush=True)
        print("{} done: {} files".format(name, len(logs)), flush=True)

    path = os.path.join(RESULTS_DIR, "pred_links_v2.json")
    meta = {
        "_meta": {
            "source": "v2 DiGAT(glove) first-best argmax edges",
            "caveat": "train-split predictions are optimistic (model trained on them); used only as auxiliary thread-feature edges",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_files": len(out),
        },
    }
    meta.update(out)
    with open(path, "w") as f:
        json.dump(meta, f)
    print("saved:", path, "completed_at:",
          datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
