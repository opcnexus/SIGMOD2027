"""A pure-torch implementation of DistilBERT inference, encoding every message as a 768-dimensional sentence vector.

It does not depend on the transformers library (the sandbox file proxy intercepts its module import),
only safetensors is used to read the weights.
Output: data/bert_feats.npz  {short_name: (n, 768) float16}
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_split
from ff_reproduction import DATA_ROOT
from digat import load_split_paths

BERT_DIR = "./data/bert"
OUT_PATH = "./data/bert_feats.npz"
MAX_LEN = 48       # max tokens per message (IRC messages are short)
BATCH = 128


class DistilBERT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, L, H = cfg["dim"], cfg["n_layers"], cfg["n_heads"]
        self.d = d
        self.word_emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos_emb = nn.Embedding(cfg["max_position_embeddings"], d)
        self.emb_ln = nn.LayerNorm(d, eps=cfg.get("layer_norm_eps", 1e-12))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q": nn.Linear(d, d), "k": nn.Linear(d, d),
                "v": nn.Linear(d, d), "o": nn.Linear(d, d),
                "ln1": nn.LayerNorm(d, eps=cfg.get("layer_norm_eps", 1e-12)),
                "f1": nn.Linear(d, cfg["hidden_dim"]),
                "f2": nn.Linear(cfg["hidden_dim"], d),
                "ln2": nn.LayerNorm(d, eps=cfg.get("layer_norm_eps", 1e-12)),
            }) for _ in range(L)])
        self.L, self.H = L, H
        self.act = nn.GELU()

    def forward(self, ids, mask, start=0, end=None, h=None):
        n, s = ids.shape
        pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(n, -1)
        if h is None:
            h = self.emb_ln(self.word_emb(ids) + self.pos_emb(pos))
        dh = self.d // self.H
        layers = self.layers[start:end] if end is not None else self.layers[start:]
        for lyr in layers:
            q = lyr["q"](h).view(n, s, self.H, dh).transpose(1, 2)
            k = lyr["k"](h).view(n, s, self.H, dh).transpose(1, 2)
            v = lyr["v"](h).view(n, s, self.H, dh).transpose(1, 2)
            att = torch.matmul(q, k.transpose(-1, -2)) / (dh ** 0.5)
            att = att.masked_fill(mask[:, None, None, :] == 0, -1e9)
            ctx = torch.matmul(att.softmax(-1), v)
            ctx = ctx.transpose(1, 2).reshape(n, s, self.d)
            h = lyr["ln1"](h + lyr["o"](ctx))
            h = lyr["ln2"](h + lyr["f2"](self.act(lyr["f1"](h))))
        return h


def load_weights(model, path):
    from safetensors.torch import load_file
    sd = load_file(path)
    # strip the "distilbert." prefix
    sd = {k[len("distilbert."):]: v for k, v in sd.items()
          if k.startswith("distilbert.")}
    out = {}
    out["word_emb.weight"] = sd["embeddings.word_embeddings.weight"]
    out["pos_emb.weight"] = sd["embeddings.position_embeddings.weight"]
    out["emb_ln.weight"] = sd["embeddings.LayerNorm.weight"]
    out["emb_ln.bias"] = sd["embeddings.LayerNorm.bias"]
    for i in range(model.L):
        p = "layers.{}.".format(i)
        s = "transformer.layer.{}.".format(i)
        for a, b in (("q", "attention.q_lin"), ("k", "attention.k_lin"),
                     ("v", "attention.v_lin"), ("o", "attention.out_lin")):
            out[p + a + ".weight"] = sd[s + b + ".weight"]
            out[p + a + ".bias"] = sd[s + b + ".bias"]
        out[p + "ln1.weight"] = sd[s + "sa_layer_norm.weight"]
        out[p + "ln1.bias"] = sd[s + "sa_layer_norm.bias"]
        out[p + "f1.weight"] = sd[s + "ffn.lin1.weight"]
        out[p + "f1.bias"] = sd[s + "ffn.lin1.bias"]
        out[p + "f2.weight"] = sd[s + "ffn.lin2.weight"]
        out[p + "f2.bias"] = sd[s + "ffn.lin2.bias"]
        out[p + "ln2.weight"] = sd[s + "output_layer_norm.weight"]
        out[p + "ln2.bias"] = sd[s + "output_layer_norm.bias"]
    missing, unexpected = model.load_state_dict(out, strict=True), None
    return model


def basic_tokenize(text):
    import re
    text = text.replace("<s>", "").replace("</s>", " ").lower()
    toks = re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text)
    return toks


class WordPiece:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, w in enumerate(f):
                self.vocab[w.rstrip("\n")] = i

    def encode(self, text):
        out = ["[CLS]"]
        for w in basic_tokenize(text):
            if len(out) >= MAX_LEN - 1:
                break
            if w in self.vocab:
                out.append(w)
                continue
            cur, pieces = w, []
            ok = True
            while cur:
                if pieces:
                    cand = [p for p in (cur[:j] for j in range(len(cur), 0, -1))
                            if "##" + p in self.vocab]
                    if not cand:
                        ok = False
                        break
                    p = cand[0]
                    pieces.append("##" + p)
                    cur = cur[len(p):]
                else:
                    cand = [p for p in (cur[:j] for j in range(len(cur), 0, -1))
                            if p in self.vocab]
                    if not cand:
                        ok = False
                        break
                    p = cand[0]
                    pieces.append(p)
                    cur = cur[len(p):]
            if ok:
                out.extend(pieces)
            else:
                out.append("[UNK]")
        out.append("[SEP]")
        out = out[:MAX_LEN]      # long words split into several pieces and can go out of bounds, force truncation
        return [self.vocab.get(t, self.vocab["[UNK]"]) for t in out]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(os.path.join(BERT_DIR, "config.json")) as f:
        cfg = json.load(f)
    model = DistilBERT(cfg).to(device).eval()
    load_weights(model, os.path.join(BERT_DIR, "model.safetensors"))
    print("weights loaded", flush=True)
    # sanity check: the sentence vectors must not collapse (a too small std means the forward implementation is wrong)
    with torch.no_grad():
        ids = torch.randint(1000, 5000, (8, 20), device=device)
        mask = torch.ones(8, 20, dtype=torch.long, device=device)
        probe = model(ids, mask)
        pm = mask.unsqueeze(-1).float()
        pv = (probe * pm).sum(1) / pm.sum(1)
    print("sanity: mean={:.4f} std={:.4f} (expect std>0.3)".format(
        pv.mean().item(), pv.std().item()), flush=True)
    assert pv.std().item() > 0.1, "forward output collapsed — check weight mapping"
    tok = WordPiece(os.path.join(BERT_DIR, "vocab.txt"))

    logs = []
    for split in ("train", "dev", "test"):
        logs.extend(load_split(split, DATA_ROOT))

    texts, keys = [], []
    for log in logs:
        name = os.path.basename(log.name)
        for k, m in enumerate(log.messages):
            texts.append(m.text)
            keys.append((name, k))
    print("messages:", len(texts), flush=True)

    out = {}
    buf_ids, buf_mask, buf_keys = [], [], []
    t0 = time.time()

    def flush():
        if not buf_ids:
            return
        ids = torch.tensor(buf_ids, dtype=torch.long, device=device)
        mask = torch.tensor(buf_mask, dtype=torch.long, device=device)
        with torch.no_grad():
            h = model(ids, mask)
            m = mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
        v = v.cpu().numpy().astype(np.float16)
        for (name, k), vec in zip(buf_keys, v):
            out.setdefault(name, []).append((k, vec))
        buf_ids.clear(); buf_mask.clear(); buf_keys.clear()

    maxlen = 0
    for (name, k), text in zip(keys, texts):
        ids = tok.encode(text)
        maxlen = max(maxlen, len(ids))
        buf_ids.append(ids + [0] * (MAX_LEN - len(ids)))
        buf_mask.append([1] * len(ids) + [0] * (MAX_LEN - len(ids)))
        buf_keys.append((name, k))
        if len(buf_ids) >= BATCH:
            flush()
    flush()
    print("encoded in {:.0f}s, max len {}".format(time.time() - t0, maxlen),
          flush=True)

    final = {}
    for name, lst in out.items():
        lst.sort(key=lambda x: x[0])
        final[name] = np.stack([v for _, v in lst])
    np.savez_compressed(OUT_PATH, **final)
    print("saved:", OUT_PATH, len(final), "files")


if __name__ == "__main__":
    main()
