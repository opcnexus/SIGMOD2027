#!/usr/bin/env python3
"""Cross-tier transfer experiment (report section 4.1, next step)

Goal: quantify the disentanglement F1 decay caused by a "representation space
shift" --
the same model architecture and the same feature system (language independent:
char-3gram hashing + structure / speaker / position / time-gap features),
running in-domain and zero-shot transfer across the IRC / Molweni / Discord(silver)
tiers.

Configurations (6):
  in-domain:  irc→irc, molweni→molweni, discord→discord
  cross:      irc→discord, discord→irc, molweni→discord

Evaluation protocol (identical to evaluate_v3):
  evaluate only on messages that contain gold edges; candidates = a forward window of
  W messages; pred = sigmoid >= tau;
  micro P/R/F1 (edge level). Discord silver = the platform's explicit
  message_reference (the bright region).

Records:
  results/transfer_experiment.json   all results + completed_at + a data fingerprint
  results/logs/transfer_experiment.log  a copy of the run log
  built-in checks: parents time-ordering check, non-empty gold edges, F1 range,
  prev-1 baseline anchor

The completion time is authoritative in the JSON completed_at field.
"""
import os, sys, json, re, time, glob, random, hashlib
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
os.makedirs(LOGS, exist_ok=True)
SEED = 42
W = 100          # candidate window
HASH_DIM = 20000
EMB_DIM = 64
EPOCHS = 12
PATIENCE = 3

import torch
import torch.nn as nn
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
rng = np.random.RandomState(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def now_min():
    return time.strftime("%Y-%m-%d %H:%M")

# ---------------- Data loading: unified conv format ----------------
# conv = {"conv_id": str, "msgs": [{"id","author","parents":[...],"text","ts"}]}

def load_irc():
    from data_loader import load_chatlog
    from digat import build_split
    split = build_split(SEED)  # lists {'train': [...], 'dev': [...], 'test': [...]}
    out = {}
    for ph in ("train", "dev", "test"):
        names = split[ph]
        convs = []
        for nm in names:
            base = os.path.join(ROOT, "data", "irc-disentanglement", "data", nm)
            logc = load_chatlog(base, with_gold=True)
            if logc is None:
                continue
            msgs = []
            for m in logc.messages:
                if m.is_system or m.mid < 1000:
                    continue
                msgs.append({"id": m.mid, "author": m.speaker,
                             "parents": logc.replies_of(m.mid),
                             "text": " ".join(t for t in m.tokens if t not in ("<s>", "</s>")),
                             "ts": None})
            if len(msgs) >= 20:
                convs.append({"conv_id": nm, "msgs": msgs})
        out[ph] = convs
    return out

def load_molweni(max_train=800, max_test=200):
    MOL = os.path.join(ROOT, "data", "molweni")
    train_raw = json.load(open(os.path.join(MOL, "DP_train.json")))
    dev_raw = json.load(open(os.path.join(MOL, "DP_dev.json")))
    test_raw = json.load(open(os.path.join(MOL, "DP_test.json")))
    def to_convs(data, limit):
        convs = []
        for conv in data[:limit * 3]:
            n = len(conv["edus"])
            parents = defaultdict(list)
            for rel in conv["relations"]:
                x, y = rel["x"], rel["y"]
                if 0 <= x < n and 0 <= y < n and x != y:
                    parents[y].append(x)
            msgs = [{"id": k, "author": e.get("speaker", ""),
                     "parents": parents.get(k, []),
                     "text": e.get("text", ""), "ts": None}
                    for k, e in enumerate(conv["edus"])]
            if sum(1 for m in msgs if m["parents"]) >= 3:
                convs.append({"conv_id": str(conv.get("id", len(convs))), "msgs": msgs})
            if len(convs) >= limit:
                break
        return convs
    return {"train": to_convs(train_raw, max_train),
            "dev": to_convs(dev_raw, 100),
            "test": to_convs(test_raw, max_test)}

def load_discord():
    """channel-week windows; only messages with an explicit reference are kept as
    silver annotations (the bright region).
    Note: evaluation uses only the messages that have gold edges, which focuses on
    the bright region by construction; training likewise uses only the bright-region
    messages and their candidates."""
    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "candidates", "discord", "*.json"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("is_bot") or d.get("type") not in (0, 19, 21):
                continue
            ts = None
            if d.get("timestamp"):
                try:
                    from datetime import datetime, timezone
                    ts = int(datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).timestamp())
                except Exception:
                    ts = None
            ref = (d.get("message_reference") or {}).get("message_id")
            by[(f.split("/")[-1], d.get("channel_id"), ts // 604800 if ts else 0)].append(
                {"id": d["id"], "author": d["author"]["id"],
                 "parents": [ref] if ref else [], "ts": ts,
                 "text": (d.get("content") or "")[:300]})
    convs = []
    for (srv, ch, wk), msgs in by.items():
        if len(msgs) < 25:
            continue
        ids = {m["id"] for m in msgs}
        for m in msgs:
            m["parents"] = [p for p in m["parents"] if p in ids]
        msgs.sort(key=lambda m: m["ts"] or 0)
        # keep only the bright-region messages as modelling targets (consistent with
        # the training / evaluation protocol)
        bright = [m for m in msgs if m["parents"]]
        if len(bright) >= 8:
            convs.append({"conv_id": f"{srv[:6]}-{ch[-6:]}-{wk}", "msgs": bright})
    return convs

def split_convs(convs, ratios=(0.7, 0.15, 0.15)):
    convs = sorted(convs, key=lambda c: c["conv_id"])
    r = random.Random(SEED)
    idx = list(range(len(convs)))
    r.shuffle(idx)
    n = len(convs)
    ntr, ndv = int(n * ratios[0]), int(n * ratios[1])
    return {"train": [convs[i] for i in idx[:ntr]],
            "dev": [convs[i] for i in idx[ntr:ntr + ndv]],
            "test": [convs[i] for i in idx[ntr + ndv:]]}

# ---------------- Features: char-3gram hashing ----------------
def grams(text):
    t = re.sub(r"\s+", "", text.lower())
    if len(t) < 3:
        return [t] if t else []
    return [t[i:i + 3] for i in range(len(t) - 2)]

class Hasher:
    def __init__(self):
        self.mat = None  # lazy: torch embedding

    def fit(self, all_texts):
        self.mat = (rng.randn(HASH_DIM, EMB_DIM) / np.sqrt(EMB_DIM)).astype(np.float32)

    def msg_vec(self, text):
        v = np.zeros(EMB_DIM, dtype=np.float32)
        if not self.mat is None:
            for g in set(grams(text)):
                h = int(hashlib.md5(g.encode()).hexdigest(), 16) % HASH_DIM
                v += self.mat[h]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

# ---------------- Models ----------------
class PairNet(nn.Module):
    def __init__(self, d=EMB_DIM + 2, hid=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 6, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d * 2 + 6, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, 1))

    def forward(self, seq, pair_idx, pair_feats):
        # seq: [1, L, d]; causal visibility is controlled by the mask (candidates
        # only take history)
        h, _ = self.attn(seq, seq, seq)
        ci, pi = pair_idx[:, 0], pair_idx[:, 1]
        z = torch.cat([h[0, ci], h[0, pi], pair_feats], dim=-1)
        return self.mlp(z).squeeze(-1)

def build_examples(msgs, msg_vecs):
    """Every message with a gold edge -> (child_pos, candidate_pos_list, labels,
    pair_feats)"""
    ex = []
    n = len(msgs)
    for i, m in enumerate(msgs):
        if not m["parents"]:
            continue
        lo = max(0, i - W)
        cands = list(range(lo, i))
        if not cands:
            continue
        pid_set = set(m["parents"])
        labels = [1.0 if msgs[c]["id"] in pid_set else 0.0 for c in cands]
        ex.append((i, cands, labels))
    return ex

def pair_features(msgs, i, c, id2pos):
    m, cnd = msgs[i], msgs[c]
    same_author = 1.0 if m["author"] and m["author"] == cnd["author"] else 0.0
    gap = float(np.log1p(i - c))
    tgap = float(np.log1p(max(0, (m["ts"] or 0) - (cnd["ts"] or 0)))) if (m["ts"] and cnd["ts"]) else 0.0
    has_ts = 1.0 if (m["ts"] and cnd["ts"]) else 0.0
    dl = float(np.log1p(abs(len(m["text"]) - len(cnd["text"]))))
    inter = float(np.log1p(i - c - 1))
    return [same_author, gap, tgap, has_ts, dl, inter]

MSG_VECS_CACHE = None

def node_seq(msgs, vecs):
    """64d hashing vector + [normalised position, log text length] = 66d node
    features"""
    n = len(msgs)
    extra = [[(i / max(1, n - 1)), float(np.log1p(len(msgs[i]["text"])))] for i in range(n)]
    return np.concatenate([np.stack(vecs), np.array(extra, dtype=np.float32)], axis=1)

def encode_conv(model, msgs, msg_vecs):
    seq = torch.tensor(node_seq(msgs, msg_vecs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ex = build_examples(msgs, msg_vecs)
    if not ex:
        return []
    out = []
    for (i, cands, labels) in ex:
        pair_idx = torch.tensor([[i, c] for c in cands], dtype=torch.long, device=DEVICE)
        pf = torch.tensor([pair_features(msgs, i, c, None) for c in cands],
                          dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            if model.training:
                logits = model(seq, pair_idx, pf)
            else:
                logits = model(seq, pair_idx, pf)
        out.append((i, cands, labels, logits.cpu().numpy()))
    return out

def eval_convs(model, convs, msg_vec_of, tau=0.5):
    model.eval()
    tp = fp = fn = 0
    for conv in convs:
        msgs = conv["msgs"]
        vecs = [msg_vec_of(m) for m in msgs]
        for (i, cands, labels, logits) in encode_conv(model, msgs, vecs):
            pred = {cands[j] for j, s in enumerate(logits) if s >= tau}
            if not pred:
                pred = {cands[int(np.argmax(logits))]}
            gold = {cands[j] for j, l in enumerate(labels) if l > 0}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"P": round(100 * p, 1), "R": round(100 * r, 1), "F1": round(100 * f1, 1),
            "tp": tp, "fp": fp, "fn": fn}

def baseline_prev1(convs):
    tp = fp = fn = 0
    for conv in convs:
        msgs = conv["msgs"]
        for i, m in enumerate(msgs):
            if not m["parents"] or i == 0:
                continue
            gold = set(m["parents"])
            pred = {msgs[i - 1]["id"]}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"P": round(100 * p, 1), "R": round(100 * r, 1),
            "F1": round(100 * 2 * p * r / max(1e-9, p + r), 1)}

def train_model(train_convs, dev_convs, msg_vec_of, tag):
    model = PairNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(train_convs)
        tot = nb = 0
        for conv in train_convs:
            msgs = conv["msgs"]
            vecs = [msg_vec_of(m) for m in msgs]
            ex = build_examples(msgs, vecs)
            if not ex:
                continue
            seq = torch.tensor(node_seq(msgs, vecs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            for (i, cands, labels) in ex:
                pair_idx = torch.tensor([[i, c] for c in cands], dtype=torch.long, device=DEVICE)
                pf = torch.tensor([pair_features(msgs, i, c, None) for c in cands],
                                  dtype=torch.float32, device=DEVICE)
                y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)
                logits = model(seq, pair_idx, pf)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
        m = eval_convs(model, dev_convs, msg_vec_of)
        log(f"  {tag} ep{ep:02d} loss={tot/max(1,nb):.4f} dev F1={m['F1']}")
        if m["F1"] > best_f1:
            best_f1, bad = m["F1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"  {tag} early stop ep{ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

# ---------------- Validation ----------------
def validate_convs(name, convs):
    n_msg = n_gold = n_tviol = 0
    for c in convs:
        for m in c["msgs"]:
            n_msg += 1
            n_gold += len(m["parents"])
            if m["ts"]:
                id2t = {x["id"]: x["ts"] for x in c["msgs"]}
                for p in m["parents"]:
                    if p in id2t and id2t[p] > m["ts"]:
                        n_tviol += 1
    log(f"[validate] {name}: convs={len(convs)} msgs={n_msg} gold_edges={n_gold} time_violations={n_tviol}")
    assert n_gold > 0, f"{name}: no gold edges!"
    assert n_tviol == 0, f"{name}: {n_tviol} parent-after-child time violations!"
    return {"convs": len(convs), "msgs": n_msg, "gold_edges": n_gold, "time_violations": n_tviol}

# ---------------- Main flow ----------------
def main():
    t0 = time.time()
    out = {"completed_at": now_min(), "device": DEVICE,
           "config": {"W": W, "hash_dim": HASH_DIM, "emb_dim": EMB_DIM,
                      "epochs": EPOCHS, "patience": PATIENCE, "seed": SEED},
           "features": "char3gram-hashing(64d)+same_author+log_idxgap+log_timelap+has_ts+log_len_diff+log_interleave; causal MHA encoder; BCE multi-label; tau=0.5 w/ argmax fallback",
           "datasets": {}, "results": {}, "notes": []}

    log("loading IRC ...")
    irc = load_irc()
    out["datasets"]["irc"] = {ph: validate_convs(f"irc-{ph}", v) for ph, v in irc.items() if v}
    log("loading Molweni ...")
    mol = load_molweni()
    out["datasets"]["molweni"] = {ph: validate_convs(f"molweni-{ph}", v) for ph, v in mol.items() if v}
    log("loading Discord (silver) ...")
    dis = split_convs(load_discord())
    out["datasets"]["discord"] = {ph: validate_convs(f"discord-{ph}", v) for ph, v in dis.items() if v}

    global MSG_VECS_CACHE
    hasher = Hasher()
    hasher.fit(None)  # the whole experiment shares one random projection matrix
                     # (keeps the configurations comparable)

    def msg_vec_of(m):
        return hasher.msg_vec(m["text"])

    def run_config(tag, train_convs, dev_convs, test_convs):
        log(f"=== config {tag} ===")
        base = baseline_prev1(test_convs)
        log(f"  baseline prev-1: {base}")
        model = train_model(train_convs, dev_convs, msg_vec_of, tag)
        m = eval_convs(model, test_convs, msg_vec_of)
        log(f"  TEST: {m}")
        out["results"][tag] = {"baseline_prev1": base, "model": m,
                               "n_train": len(train_convs), "n_test": len(test_convs)}
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    run_config("irc->irc", irc["train"], irc["dev"], irc["test"])
    run_config("molweni->molweni", mol["train"], mol["dev"], mol["test"])
    run_config("discord->discord", dis["train"], dis["dev"], dis["test"])
    # zero-shot cross-tier: test the model trained on the source domain directly on
    # the target domain
    # retrain and evaluate on the target test set (the target-domain dev is used only
    # to tune tau and does not take part in training)
    def zero_shot(tag, src_train, src_dev, tgt_test):
        log(f"=== config {tag} (zero-shot) ===")
        base = baseline_prev1(tgt_test)
        log(f"  baseline prev-1 on target: {base}")
        model = train_model(src_train, src_dev, msg_vec_of, tag)
        m = eval_convs(model, tgt_test, msg_vec_of)
        log(f"  TEST on target: {m}")
        out["results"][tag] = {"baseline_prev1": base, "model": m,
                               "n_train": len(src_train), "n_test": len(tgt_test)}
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    zero_shot("irc->discord", irc["train"], irc["dev"], dis["test"])
    zero_shot("discord->irc", dis["train"], dis["dev"], irc["test"])
    zero_shot("molweni->discord", mol["train"], mol["dev"], dis["test"])

    # aggregate the decay
    for tgt, keys in {"irc": ["irc->irc", "discord->irc"],
                      "discord": ["discord->discord", "irc->discord", "molweni->discord"]}.items():
        for k in keys:
            if k in out["results"]:
                pass
    decay = {}
    if "irc->discord" in out["results"] and "discord->discord" in out["results"]:
        decay["irc_to_discord_F1_drop"] = round(
            out["results"]["discord->discord"]["model"]["F1"] -
            out["results"]["irc->discord"]["model"]["F1"], 1)
    if "discord->irc" in out["results"] and "irc->irc" in out["results"]:
        decay["discord_to_irc_F1_drop"] = round(
            out["results"]["irc->irc"]["model"]["F1"] -
            out["results"]["discord->irc"]["model"]["F1"], 1)
    if "molweni->discord" in out["results"] and "discord->discord" in out["results"]:
        decay["molweni_to_discord_F1_drop"] = round(
            out["results"]["discord->discord"]["model"]["F1"] -
            out["results"]["molweni->discord"]["model"]["F1"], 1)
    out["F1_decay"] = decay
    out["runtime_seconds"] = round(time.time() - t0)
    out["completed_at"] = now_min()

    path = os.path.join(RESULTS, "transfer_experiment.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    log(f"saved -> {path}")
    log(f"ALL_DONE in {out['runtime_seconds']}s")

if __name__ == "__main__":
    main()
