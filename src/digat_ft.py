#!/usr/bin/env python3
"""End-to-end DistilBERT fine-tuning + DiGAT + pointer-based parent selection under
the official protocol (153/10/10).

Unlike static sentence vectors (bert_feats.npz), the BERT weights take part in
backpropagation (the top 2 layers are fine-tuned), so the numbers can be compared
directly with the literature (official test 10 files, link-level F1, evaluation from
line 1000).

Design:
- Node features = GloVe handcrafted features (55) + BERT mean-pool (768)
  (re-forwarded every epoch)
- BERT freezes embedding + layers 0-3 and fine-tunes layers 4-5 (lr=2e-5); the rest
  uses lr=1e-3
- Single-label CE (first-best = the highest-scoring gold parent) with argmax
  decoding (consistent with official FF)
- completed_at at minute precision
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import (DATA_ROOT, TEST_START, load_glove,
                             gold_edge_sets, link_prf, short_name)
from digat import (RESULTS_DIR, node_features, same_speaker_matrix,
                   cand_index, DiGAT, PairScorer, CAND_WINDOW)
from bert_encode import DistilBERT, load_weights, WordPiece, BERT_DIR, MAX_LEN


def now_min():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def encode_tokens(logs, tok):
    """Pre-tokenize: {basename: (n, MAX_LEN) int64 ids}"""
    out = {}
    for log in logs:
        ids_list = []
        for m in log.messages:
            ids = tok.encode(m.text)
            ids_list.append(ids + [0] * (MAX_LEN - len(ids)))
        out[short_name(log.name)] = np.array(ids_list, dtype=np.int64)
    return out


def bert_forward(bert, ids_np, device, frozen_cache=None):
    """frozen_cache: output cache of layers 0-3. Two forms:
    - str (path of the on-disk .npy file, fp16): read in chunks through memmap and
      converted to fp32 on the GPU on demand (never loaded into RAM as a whole)
    - tensor: used directly via .to(device)
    The mean-pool denominator is clamped (min=1): for an empty-text message all ids
    are 0 -> the mask sums to 0 and the division by zero yields NaN, which propagates
    forward -> argmax always picks candidate 0 -> F1=0.0 (the root cause of
    ep00 dev F1=0.0)."""
    ids = torch.from_numpy(ids_np).to(device)
    mask = (ids != 0).long()
    if frozen_cache is not None:
        if isinstance(frozen_cache, str):
            mm = np.load(frozen_cache, mmap_mode="r")
            h = torch.empty((ids.shape[0], mm.shape[1], mm.shape[2]),
                            dtype=torch.float32, device=device)
            for s in range(0, ids.shape[0], 4096):
                h[s:s + 4096] = torch.from_numpy(
                    np.asarray(mm[s:s + 4096], dtype=np.float32)).to(device)
        else:
            h = frozen_cache.to(device)
    else:
        h = bert(ids, mask, end=4)
    h = bert(ids, mask, start=4, h=h)
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # (n, 768)


FROZEN_CACHE_DIR = os.path.join(RESULTS_DIR, "frozen_cache")


def build_frozen_cache(bert, tok_by_file, device, split):
    """Outputs of the frozen layers (embedding+0-3): computed per file in chunks ->
    written to disk as fp16 -> read through memmap on demand during training.

    Root cause of the v1 cached version crashing (system reboot at 2026-09-26
    15:04): the (n,48,768) float32 arrays of 153 files
    all stayed resident in memory, about 20-60GB, far beyond the 16GB of unified
    memory -> swap exhausted -> system-wide crash.
    This version peaks below 1GB (CH=2048 rows per chunk) and resumes from already
    cached files after a restart (no recomputation)."""
    out_dir = os.path.join(FROZEN_CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    bert.eval()
    CH = 2048
    with torch.no_grad():
        for name, ids_np in tok_by_file.items():
            fp = os.path.join(out_dir, name.replace("/", "_") + ".npy")
            if os.path.exists(fp):
                paths[name] = fp
                print("  [cache hit] {} ({})".format(name, split), flush=True)
                continue
            n = ids_np.shape[0]
            fp_tmp = fp + ".tmp.npy"
            buf = np.lib.format.open_memmap(
                fp_tmp, mode="w+", dtype=np.float16,
                shape=(n, MAX_LEN, 768))
            for s in range(0, n, CH):
                ids = torch.from_numpy(ids_np[s:s + CH]).to(device)
                mask = (ids != 0).long()
                h = bert(ids, mask, end=4)
                buf[s:s + CH] = h.cpu().numpy().astype(np.float16)
                del ids, mask, h
            buf.flush()
            del buf
            os.replace(fp_tmp, fp)
            paths[name] = fp
            print("  [cached] {} n={} {:.1f}MB on disk [{}]".format(
                name, n, n * MAX_LEN * 768 * 2 / 1e6, split), flush=True)
    return paths


def labels_top1(log, queries, n):
    qi, cj, valid = cand_index(n, queries)
    Y = np.zeros_like(valid, dtype=np.float32)
    keep = np.zeros(len(queries), dtype=bool)
    for r, i in enumerate(queries):
        gold = set(t for t in log.parents.get(int(i), [int(i)]))
        hit = np.isin(cj[r], list(gold))
        Y[r] = hit.astype(np.float32)
        keep[r] = hit.any()
    return qi[keep], cj[keep], valid[keep], Y[keep]


def file_loss(bert, graph, scorer, log, tok_ids, emb, device, frozen_cache=None):
    x = node_features(log, emb)
    bv = bert_forward(bert, tok_ids, device, frozen_cache).detach().cpu().numpy()
    x = np.concatenate([x, bv], axis=1)
    n = x.shape[0]
    queries = list(range(TEST_START, n))
    if not queries:
        return None
    qi, cj, valid, Y = labels_top1(log, queries, n)
    if len(qi) == 0:
        return None
    x_t = torch.from_numpy(x).to(device)
    same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
    h = graph(x_t, same_t)
    scores = scorer(h, torch.from_numpy(qi).to(device),
                    torch.from_numpy(cj).to(device),
                    same_t, n)
    valid_t = torch.from_numpy(valid).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    s_m = scores.masked_fill(~valid_t, -1e9)
    gold_scores = s_m * Y_t + (-1e9) * (1 - Y_t)
    top1 = gold_scores.argmax(dim=1)
    return nn.functional.cross_entropy(s_m, top1)


def predict(model_pack, log, tok_ids, emb, device, frozen_cache=None):
    bert, graph, scorer = model_pack
    bert.eval(); graph.eval(); scorer.eval()
    with torch.no_grad():
        x = node_features(log, emb)
        bv = bert_forward(bert, tok_ids, device, frozen_cache).detach().cpu().numpy()
        x = np.concatenate([x, bv], axis=1)
        n = x.shape[0]
        x_t = torch.from_numpy(x).to(device)
        same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
        h = graph(x_t, same_t)
        queries = list(range(TEST_START, n))
        if not queries:
            return set()
        qi, cj, valid = cand_index(n, queries)
        scores = scorer(h, torch.from_numpy(qi).to(device),
                        torch.from_numpy(cj).to(device), same_t, n)
        scores = scores.masked_fill(~torch.from_numpy(valid).to(device), -1e9)
        pred = scores.argmax(dim=1).cpu().numpy()
    return set((int(i), int(cj[r][pred[r]])) for r, i in enumerate(queries))


def evaluate(pack, logs, tok, emb, device, caches=None):
    gold = gold_edge_sets(logs)
    auto = {}
    for log in logs:
        nm = short_name(log.name)
        fc = caches[nm] if caches else None
        auto[nm] = predict(pack, log, tok[nm], emb, device, fc)
    return link_prf(gold, auto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bert-lr", type=float, default=2e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--tag", default="bert_ft_official")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        print("WARNING: MPS unavailable, falling back to CPU (more than 10x slower,"
              " not recommended)", flush=True)
    print("device:", device, flush=True)

    # memory / disk guard (the system crashed twice on a 16GB unified-memory
    # machine, so check hard limits up front)
    import shutil
    free_gb = shutil.disk_usage(RESULTS_DIR).free / 1e9
    print("disk free: {:.1f} GB (the cache needs about 10-12 GB)".format(free_gb),
          flush=True)
    if free_gb < 15:
        raise SystemExit("less than 15GB of disk left, please clean results/ or the"
                         " system cache first")

    train_logs = load_split("train", DATA_ROOT)
    dev_logs = load_split("dev", DATA_ROOT)
    test_logs = load_split("test", DATA_ROOT)
    print("official split: train={} dev={} test={}".format(
        len(train_logs), len(dev_logs), len(test_logs)), flush=True)

    print("tokenizing ...", flush=True)
    tok = WordPiece(os.path.join(BERT_DIR, "vocab.txt"))
    tok_train = encode_tokens(train_logs, tok)
    tok_dev = encode_tokens(dev_logs, tok)
    tok_test = encode_tokens(test_logs, tok)

    print("loading GloVe ...", flush=True)
    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))

    with open(os.path.join(BERT_DIR, "config.json")) as f:
        cfg = json.load(f)
    bert = DistilBERT(cfg).to(device)
    load_weights(bert, os.path.join(BERT_DIR, "model.safetensors"))
    # freeze: embedding + layers 0-3
    for p in (list(bert.word_emb.parameters()) +
              list(bert.pos_emb.parameters()) +
              list(bert.emb_ln.parameters())):
        p.requires_grad_(False)
    for i in range(4):
        for p in bert.layers[i].parameters():
            p.requires_grad_(False)
    print("BERT loaded; layers 4-5 trainable", flush=True)
    print("building frozen-layer caches (one-time, disk-streamed fp16) ...", flush=True)
    fc_train = build_frozen_cache(bert, tok_train, device, "train")
    fc_dev = build_frozen_cache(bert, tok_dev, device, "dev")
    fc_test = build_frozen_cache(bert, tok_test, device, "test")
    print("frozen caches ready (paths on disk, loaded per-file via memmap)", flush=True)

    in_dim = 55 + 768
    graph = DiGAT(in_dim, d_model=args.d_model).to(device)
    scorer = PairScorer(d_model=args.d_model).to(device)

    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": bert_params, "lr": args.bert_lr},
        {"params": list(graph.parameters()) + list(scorer.parameters()),
         "lr": args.lr, "weight_decay": 1e-6},
    ])

    best_dev, best_state, bad = -1.0, None, 0
    t0 = time.time()
    ep = -1
    for ep in range(args.epochs):
        bert.train(); graph.train(); scorer.train()
        rng = np.random.RandomState(3000 + ep)
        order = rng.permutation(len(train_logs))
        tot, nb = 0.0, 0
        for bi in order:
            log = train_logs[bi]
            loss = file_loss(bert, graph, scorer, log,
                             tok_train[short_name(log.name)], emb, device,
                             fc_train[short_name(log.name)])
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_params, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        m_dev = evaluate((bert, graph, scorer), dev_logs, tok_dev, emb, device, fc_dev)
        print("ep{:02d} loss={:.4f} dev F1={:.1f} ({:.0f}s)".format(
            ep, tot / max(1, nb), m_dev["F1"], time.time() - t0), flush=True)
        if m_dev["F1"] > best_dev:
            best_dev = m_dev["F1"]; bad = 0
            best_state = {
                "bert": {k: v.detach().cpu().clone()
                         for k, v in bert.state_dict().items()},
                "graph": {k: v.detach().cpu().clone()
                          for k, v in graph.state_dict().items()},
                "scorer": {k: v.detach().cpu().clone()
                           for k, v in scorer.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= 2:
                print("early stop at ep", ep, flush=True)
                break

    if best_state:
        bert.load_state_dict(best_state["bert"])
        graph.load_state_dict(best_state["graph"])
        scorer.load_state_dict(best_state["scorer"])

    m_test = evaluate((bert, graph, scorer), test_logs, tok_test, emb, device, fc_test)
    print("BEST dev F1={:.1f}".format(best_dev))
    print("test:", m_test)

    out = {
        "model": "distilbert+digat end-to-end", "tag": args.tag,
        "protocol": "OFFICIAL 153/10/10 (comparable to literature)",
        "best_dev_F1": best_dev, "test": m_test,
        "epochs_run": ep + 1, "train_seconds": round(time.time() - t0),
        "completed_at": now_min(),
    }
    with open(os.path.join(RESULTS_DIR, "{}.json".format(args.tag)), "w") as f:
        json.dump(out, f, indent=2)
    print("saved. completed_at:", out["completed_at"])


if __name__ == "__main__":
    main()
