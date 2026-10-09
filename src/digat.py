"""GTS-DiGAT v2: directed graph attention + pointer-based parent selection for
conversation disentanglement (a system-level upgrade of GTS-Backtrack).

Design inspirations (all sources verified, see report section 2):
- TC-DAG (Li et al., arXiv:2605.01717, 2026): edges may only point from earlier to
  later messages (time-forward DAG),
  with separate relation parameters for the same and different speaker
  (thread-constrained + speaker-aware).
- DialogueGCN (Ghosal et al., EMNLP-IJCNLP 2019, D19-1015): sliding-window graph plus
  speaker relation edges.
- HeterMPC (Gu et al., ACL 2022, 2203.08500): sentence embeddings from a pretrained
  LM (BERT) as node features.
- Pointer-based parent selection + softmax CE: consistent with the FF protocol
  (Kummerfeld et al., ACL 2019).

Data protocol (user-specified): all 173 labelled files are re-split 80/20
(seed=42).
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_chatlog
from data_loader import ChatLog  # noqa: F401 (type annotations)
from ff_reproduction import (DATA_ROOT, TEST_START, load_glove,
                             gold_edge_sets, link_prf, short_name)

RESULTS_DIR = "./results"
SPLIT_PATH = os.path.join(RESULTS_DIR, "split_8020.json")
GRAPH_WINDOW = 30      # DiGAT time-forward window (DAG constraint)
CAND_WINDOW = 101      # parent candidate window (same as FF: self + previous 100)


# ---------------- 80/20 re-split ----------------

def build_split(seed: int = 42) -> dict:
    """Pool all 173 labelled files -> 138 train (the last 21 serve as an internal
    dev) / 35 test"""
    names = []
    for split in ("train", "dev", "test"):
        d = os.path.join(DATA_ROOT, split)
        for f in sorted(os.listdir(d)):
            if f.endswith(".tok.txt"):
                names.append(os.path.join(split, f[:-len(".tok.txt")]))
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    n_test = int(round(len(names) * 0.20))
    test, train = names[:n_test], names[n_test:]
    n_dev = int(round(len(train) * 0.15))
    dev, train = train[-n_dev:], train[:-n_dev]
    out = {"train": train, "dev": dev, "test": test, "seed": seed,
           "protocol": "80/20 pooled split (NOT comparable to official 153/10/10 numbers)"}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out


def load_split_paths() -> dict:
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            return json.load(f)
    return build_split()


def load_logs_for(names) -> list:
    out = []
    for name in names:  # name looks like "train/2015-xx_yy"
        split, base = name.split("/", 1)
        prefix = os.path.join(DATA_ROOT, split, base)
        log = load_chatlog(prefix)
        if log is not None:
            out.append(log)
    return out


# ---------------- Node features ----------------

def node_features(log: ChatLog, emb: dict) -> np.ndarray:
    """GloVe mean (50) + 5-dim structural features -> (n, 55)"""
    n = len(log.messages)
    feats = np.zeros((n, 55), dtype=np.float32)
    for k, m in enumerate(log.messages):
        toks = getattr(m, "tokens", None) or m.text.replace("<s>", "").replace("</s>", "").split()
        vecs = [emb[t] for t in toks if t in emb]
        if vecs:
            feats[k, :50] = np.mean(vecs, axis=0)
        feats[k, 50] = math.log1p(len(toks))
        feats[k, 51] = k / max(1, n - 1)
        feats[k, 52] = 1.0 if "?" in m.text else 0.0
        feats[k, 53] = 1.0 if "!" in m.text else 0.0
        feats[k, 54] = 1.0 if m.text.strip().startswith("@") else 0.0
    return feats


def same_speaker_matrix(log: ChatLog) -> np.ndarray:
    spk = np.array([m.speaker for m in log.messages])
    return (spk[:, None] == spk[None, :])


# ---------------- Models ----------------

class _Proj(nn.Module):
    """PairMLP mode: linearly projects the node features to d_model (the speaker
    matrix argument is ignored)"""

    def __init__(self, in_dim, d):
        super().__init__()
        self.lin = nn.Linear(in_dim, d)

    def forward(self, x, same):
        return self.lin(x)


class _Identity(nn.Module):
    """PairMLP mode: no graph layer, the node features go straight into the scorer"""

    def forward(self, x, same):
        return x

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return

    def parameters(self):
        return iter([])


class DiGAT(nn.Module):
    """Directed graph attention: node i attends only to historical nodes in
    [i-W, i] (DAG mask), with three edge relation types: self / same speaker /
    different speaker (each with its own K/V projection)."""

    def __init__(self, in_dim, d_model=128, heads=4, layers=2,
                 window=GRAPH_WINDOW, dropout=0.2):
        super().__init__()
        self.d = d_model
        self.heads = heads
        self.window = window
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "q": nn.Linear(d_model, d_model),
                "k_self": nn.Linear(d_model, d_model),
                "v_self": nn.Linear(d_model, d_model),
                "k_same": nn.Linear(d_model, d_model),
                "v_same": nn.Linear(d_model, d_model),
                "k_diff": nn.Linear(d_model, d_model),
                "v_diff": nn.Linear(d_model, d_model),
                "norm": nn.LayerNorm(d_model),
            }))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, same_spk):
        """x: (n, in_dim)  same_spk: (n, n) bool (j has the same speaker as i,
        diagonal included)"""
        n = x.shape[0]
        h = self.proj(x)
        idx = torch.arange(n, device=x.device)
        dag_mask = (idx.view(1, -1) >= idx.view(-1, 1) - self.window) & \
                   (idx.view(1, -1) <= idx.view(-1, 1))          # (n, n) j<=i and j>=i-W
        ident = idx.view(1, -1) == idx.view(-1, 1)
        for layer in self.layers:
            q = layer["q"](h).view(n, self.heads, -1).transpose(0, 1)      # (H, n, dh)
            outs = []
            for rel, km, vm in (("self", "k_self", "v_self"),
                                ("same", "k_same", "v_same"),
                                ("diff", "k_diff", "v_diff")):
                k = layer[km](h).view(n, self.heads, -1).transpose(0, 1)
                v = layer[vm](h).view(n, self.heads, -1).transpose(0, 1)
                e = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d // self.heads)
                if rel == "self":
                    rmask = dag_mask & ident
                elif rel == "same":
                    rmask = dag_mask & same_spk & ~ident
                else:
                    rmask = dag_mask & ~same_spk
                e = e.masked_fill(~rmask.unsqueeze(0), -1e9)
                outs.append(torch.matmul(e.softmax(dim=-1), v))
            agg = sum(outs) / len(outs)                                    # (H, n, dh)
            agg = agg.transpose(0, 1).reshape(n, self.d)
            h = layer["norm"](h + self.dropout(agg))
        return h


class PairScorer(nn.Module):
    """score(i,j) = MLP([h_i, h_j, h_i*h_j, |h_i-h_j|, pair scalar features])"""

    N_PAIR = 5  # log distance, 1/distance, same speaker, query position, cand position

    def __init__(self, d_model=128, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 4 + PairScorer.N_PAIR, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h, qi, cj, same_spk, n):
        """h: (n,d); qi: (Q,) query indices; cj: (Q, W) candidate indices"""
        hi = h[qi]                                        # (Q, d)
        hj = h[cj]                                        # (Q, W, d)
        W = cj.shape[1]
        hi_ = hi.unsqueeze(1).expand(-1, W, -1)
        scal = torch.stack([
            torch.log1p((qi.view(-1, 1) - cj).float().clamp(min=0)),
            1.0 / (qi.view(-1, 1) - cj).float().clamp(min=1),
            same_spk[qi.unsqueeze(1).expand(-1, W), cj].float(),
            (qi.float() / max(1, n - 1)).view(-1, 1).expand(-1, W),
            cj.float() / max(1, n - 1),
        ], dim=-1)                                        # (Q, W, 5)
        feats = torch.cat([hi_, hj, hi_ * hj, (hi_ - hj).abs(),
                           scal], dim=-1)
        return self.net(feats).squeeze(-1)                # (Q, W)


# ---------------- Training / evaluation ----------------

def cand_index(n, queries):
    """(Q, W) candidate indices + (Q, W) validity mask: cand = i-100+k,
    0<=cand<=i"""
    W = CAND_WINDOW
    qi = np.array(queries, dtype=np.int64)
    off = np.arange(W, dtype=np.int64)
    cj = qi.reshape(-1, 1) - (W - 1) + off.reshape(1, -1)
    valid = (cj >= 0) & (cj <= qi.reshape(-1, 1))
    cj = np.clip(cj, 0, n - 1)
    return qi, cj, valid


def train_file_forward(model, scorer, x_t, same_t, queries, log, device):
    """Returns (scores, pos, valid_loss mask); queries whose gold parent falls
    outside the candidate window are dropped"""
    h = model(x_t, same_t)
    n = x_t.shape[0]
    if not queries:
        return None, None, None
    qi, cj, valid = cand_index(n, queries)
    qi_t = torch.from_numpy(qi).to(device)
    cj_t = torch.from_numpy(cj).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    scores = scorer(h, qi_t, cj_t, same_t, n)
    scores = scores.masked_fill(~valid_t, -1e9)
    gold = []
    for i in qi:
        ts = log.parents.get(int(i), [])
        gold.append(ts[0] if ts else int(i))
    gold_t = torch.tensor(gold, dtype=torch.long, device=device)
    hit = (cj_t == gold_t.view(-1, 1))
    ok = hit.any(dim=1)                                  # only count the loss when the gold is in the window
    if not ok.any():
        return None, None, None
    pos = hit.float().argmax(dim=1)[ok]
    return scores[ok], pos, ok


def predict_edges(model, scorer, log, emb, device, bert_cache=None):
    """Pick the parent of every message with line>=TEST_START by argmax
    (self-links included) and return the edge set"""
    model.eval(); scorer.eval()
    with torch.no_grad():
        x = node_features(log, emb)
        if bert_cache is not None:
            x = np.concatenate([x, bert_cache[short_name(log.name)]], axis=1)
        same_t = torch.from_numpy(same_speaker_matrix(log)).to(log and "cpu")
        x_t = torch.from_numpy(x)
        device = x_t.device if x_t.is_mps else torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        x_t, same_t = x_t.to(device), same_t.to(device)
        h = model(x_t, same_t)
        n = x.shape[0]
        queries = list(range(TEST_START, n))
        if not queries:
            return set()
        qi, cj, valid = cand_index(n, queries)
        qi_t = torch.from_numpy(qi).to(device)
        cj_t = torch.from_numpy(cj).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        scores = scorer(h, qi_t, cj_t, same_t, n)
        scores = scores.masked_fill(~valid_t, -1e9)
        pred = scores.argmax(dim=1).cpu().numpy()
    edges = set()
    for r, i in enumerate(queries):
        edges.add((i, int(cj[r][pred[r]])))
    return edges


def evaluate(model, scorer, logs, emb, device, bert_cache=None):
    gold = gold_edge_sets(logs)
    auto = {}
    for log in logs:
        auto[short_name(log.name)] = predict_edges(
            model, scorer, log, emb, device, bert_cache)
    return link_prf(gold, auto), auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pairmlp", "digat"], default="digat")
    ap.add_argument("--bert", default=None, help="Path to the BERT feature npz (optional)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    split = load_split_paths()
    print("split sizes: train={} dev={} test={}".format(
        len(split["train"]), len(split["dev"]), len(split["test"])))
    train_logs = load_logs_for(split["train"])
    dev_logs = load_logs_for(split["dev"])
    test_logs = load_logs_for(split["test"])

    print("loading GloVe ...", flush=True)
    emb = load_glove(os.path.join(DATA_ROOT, "glove-ubuntu.txt"))

    bert_cache = None
    if args.bert and os.path.exists(args.bert):
        z = np.load(args.bert)
        # npz keys come from the basename (may carry suffixes such as .train-c),
        # normalised to a short name here
        bert_cache = {k.split(".")[0]: z[k].astype(np.float32) for k in z.files}
        print("bert feats loaded:", len(bert_cache), "files", flush=True)

    in_dim = 55 + (768 if bert_cache is not None else 0)
    graph = (DiGAT(in_dim, d_model=args.d_model).to(device)
             if args.model == "digat"
             else _Proj(in_dim, args.d_model).to(device))  # PairMLP: score directly after the projection
    scorer = PairScorer(d_model=args.d_model).to(device)
    params = list(scorer.parameters()) + list(graph.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-6)

    best_dev, best_state, bad = -1.0, None, 0
    t0 = time.time()
    ep = -1
    for ep in range(args.epochs):
        graph.train(); scorer.train()
        rng = np.random.RandomState(1000 + ep)
        order = rng.permutation(len(train_logs))
        tot_loss, n_batches = 0.0, 0
        for bi in order:
            log = train_logs[bi]
            x = node_features(log, emb)
            if bert_cache is not None:
                x = np.concatenate([x, bert_cache[short_name(log.name)]], axis=1)
            queries = sorted(i for i in log.parents if 1 <= i < x.shape[0])
            if not queries:
                continue
            x_t = torch.from_numpy(x).to(device)
            same_t = torch.from_numpy(same_speaker_matrix(log)).to(device)
            scores, pos, _ = train_file_forward(graph, scorer, x_t, same_t,
                                                queries, log, device)
            if scores is None:
                continue
            loss = nn.functional.cross_entropy(scores, pos)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item(); n_batches += 1
        m_dev, _ = evaluate(graph, scorer, dev_logs, emb, device, bert_cache)
        print("ep{:02d} loss={:.4f} dev F1={:.1f} ({:.0f}s)".format(
            ep, tot_loss / max(1, n_batches), m_dev["F1"], time.time() - t0),
            flush=True)
        if m_dev["F1"] > best_dev:
            best_dev = m_dev["F1"]
            bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in list(graph.state_dict().items()) +
                          list(scorer.state_dict().items())}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at ep", ep, flush=True)
                break

    if best_state:
        g_keys = set(graph.state_dict().keys())
        graph.load_state_dict({k: v for k, v in best_state.items() if k in g_keys})
        scorer.load_state_dict({k: v for k, v in best_state.items()
                                if k not in g_keys})

    m_test, auto = evaluate(graph, scorer, test_logs, emb, device, bert_cache)
    print("BEST dev F1={:.1f}".format(best_dev))
    print("test:", m_test)

    tag = args.tag or args.model
    with open(os.path.join(RESULTS_DIR, "digat_results_{}.json".format(tag)), "w") as f:
        json.dump({"model": args.model, "bert": bool(bert_cache),
                   "best_dev_F1": best_dev, "test": m_test,
                   "epochs_run": ep + 1}, f, indent=2)
    torch.save({"graph": graph.state_dict(), "scorer": scorer.state_dict(),
                "in_dim": in_dim, "d_model": args.d_model,
                "model": args.model},
               os.path.join(RESULTS_DIR, "digat_{}.pt".format(tag)))
    print("saved:", tag)


if __name__ == "__main__":
    main()
