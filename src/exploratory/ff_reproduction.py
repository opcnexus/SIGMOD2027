#!/usr/bin/env python3
"""Faithful reproduction of the FF model of Kummerfeld et al. (ACL 2019) (pure Python + PyTorch)

- Features: the official get_features() 77 dims (replicated line by line from src/disentangle.py, incl. year/msg-rate/query/link/both)
- Model: input(77) -> hidden(512, softsign) -> hidden -> 1 logit, softmax over the candidate window
- Training: every query message scores all candidates in its 101-wide window, cross-entropy (gold is multi-label -> normalized multiclass)
- Evaluation: official link-level P/R/F1, test starts at line>=1000
"""
from __future__ import annotations

import math
import os
import string
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ChatLog, load_split

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = "./results"
TEST_START = 1000
MAX_DIST = 101

# the official reserved_words list is too large; take the necessary subset here (bot names, etc.)
BOTS = {"ubottu", "ubotu"}
COMMON_SHORT_NAMES = set()  # the official code has a list; the empty set has negligible effect (only affects short-name mention detection)


# ---------------- replication of the official features ----------------

def update_users_from_ascii(lines: list, users: set):
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        user = parts[1]
        if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
                    "Window", "Server:", "Screen:", "Geometry", "CO,",
                    "Current", "Query", "Prompt:", "Second", "Split",
                    "Logging", "Logfile", "Notification", "Hold", "Window",
                    "Lastlog", "Notify", "netjoined:"]:
            continue
        if parts[0].endswith("==="):
            seg = " ".join(parts).split("is now known as")
            if len(seg) == 2 and parts[-1] == seg[-1].strip():
                user = parts[-1]
        elif parts[0].endswith("]") and user.startswith("<"):
            user = user[1:-1]
        user = user.lower()
        if user in BOTS:
            users.add(user)
            continue
        if user.isdigit():
            continue
        users.add(user)


def get_targets_official(line_parts: list, users: set) -> set:
    targets = set()
    for token in line_parts[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = list(token)
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = "".join(core)
                if nword in users and (len(core) > 2 or nword in COMMON_SHORT_NAMES):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = "".join(core)
                    if nword in users and (len(core) > 2 or nword in COMMON_SHORT_NAMES):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets


def lines_to_info(text_ascii: list):
    """Replication of the official lines_to_info -> (info, target_info)"""
    users = set()
    update_users_from_ascii(text_ascii, users)

    chour, cmin = 12, 0
    info, target_info, nexts = [], {}, {}
    for line_no, line in enumerate(text_ascii):
        parts = line.split()
        if len(parts) < 2:
            parts = parts + ["", ""]
        if parts[0].startswith("["):
            user = parts[1][1:-1]
            nexts.setdefault(user, []).append(line_no)

    prev = {}
    for line_no, line in enumerate(text_ascii):
        parts = line.split()
        if len(parts) < 2:
            parts = parts + ["", ""]
        user = parts[1]
        system = True
        if parts[0].startswith("["):
            chour = int(parts[0][1:3])
            cmin = int(parts[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = user in BOTS
        targets = get_targets_official(parts, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]
        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, parts, next_from_user))
    return info, target_info


def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        a, b = b, a
    ahour, amin = info[a][2], info[a][3]
    bhour, bmin = info[b][2], info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60 * (bhour - ahour - 1)


def get_features_official(name: str, query_no: int, link_no: int, text_ascii: list,
                          info, target_info) -> list:
    """The 77-dim official features (no word vectors)"""
    f = []
    quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user = info[query_no]
    luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user = info[link_no]

    for i in range(2004, 2018):
        f.append(str(i) in name)
    # msg per minute
    start = end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i].split() and text_ascii[i].split()[0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i].split() and text_ascii[-i].split()[0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(s <= msg_per_min < e)

    # Query (10)
    f.append(qsystem)
    f.append(qhour / 24)
    f.append(len(qtargets) > 0)
    f.append(qlast_from_user is not None)
    f.append(False if qlast_from_user is None else len(info[qlast_from_user][1]) > 0)
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(s <= dist < e)
    time_ = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(s <= time_ < e)
    f.append(qis_bot)

    # Link (13)
    f.append(lsystem)
    f.append(lhour / 24)
    f.append(link_no != query_no and len(ltargets) > 0)
    f.append(link_no != query_no and llast_from_user is not None)
    f.append(False if (link_no == query_no or llast_from_user is None) else len(info[llast_from_user][1]) > 0)
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(link_no != query_no and s <= dist < e)
    time_ = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(link_no != query_no and s <= time_ < e)
    f.append(lis_bot)
    f.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    f.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    # Both (22)
    f.append(link_no == query_no)
    dist = query_no - link_no
    f.append(min(100, dist) / 100)
    f.append(dist > 1)
    time_ = get_time_diff(info, link_no, query_no)
    f.append(min(100, time_) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for s, e in zip(cutoffs, cutoffs[1:]):
        f.append(s <= time_ < e)
    f.append(quser.lower() in ltargets)
    f.append(luser.lower() in qtargets)
    f.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    f.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    if link_no != query_no and (quser, luser) in target_info:
        f.append(min(target_info[quser, luser]) < link_no)
        f.append(max(target_info[quser, luser]) > query_no)
        f.append(any(query_no > n > link_no for n in target_info[quser, luser]))
    else:
        f += [False, False, False]
    if link_no != query_no and (luser, quser) in target_info:
        f.append(min(target_info[luser, quser]) < link_no)
        f.append(max(target_info[luser, quser]) > query_no)
        f.append(any(query_no > n > link_no for n in target_info[luser, quser]))
    else:
        f += [False, False, False]
    f.append(luser == quser)
    f.append(link_no != query_no and len(ltargets & qtargets) > 0)
    ltokens = set(text_ascii[link_no].split())
    qtokens = set(text_ascii[query_no].split())
    common = len(ltokens & qtokens)
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        f.append(common / len(ltokens))
        f.append(common / len(qtokens))
    else:
        f += [False, False]
    f.append(link_no != query_no and common == 0)
    f.append(link_no != query_no and common == 1)
    f.append(link_no != query_no and common > 1)
    f.append(link_no != query_no and common > 5)

    return [1.0 if x is True else (0.0 if x is False else float(x)) for x in f]


# ---------------- data preparation ----------------

class LogPrep:
    """Precomputes info/target_info/text_ascii; optionally attaches GloVe word vectors
    With cache_feats=False the feature lists are not cached (required to avoid OOM during large-scale training/evaluation)"""
    def __init__(self, log: ChatLog, emb: dict = None, cache_feats: bool = True):
        self.log = log
        self.text_ascii = [m.raw for m in log.messages]
        self.info, self.target_info = lines_to_info(self.text_ascii)
        self.cand_cache = {}
        self.feat_cache = {} if cache_feats else None
        self.cache_feats = cache_feats
        self.emb = emb
        self.utt_vec_cache = {}
        if emb is not None:
            unka = emb.get("<unka>")
            self._unka = unka if unka is not None else np.zeros(50, dtype=np.float32)

    def candidates(self, i: int) -> list:
        """Window: self (=root) + the previous 100 messages"""
        if i in self.cand_cache:
            return self.cand_cache[i]
        lo = max(0, i - MAX_DIST + 1)
        cands = list(range(lo, i)) + [i]  # order: history messages + self
        self.cand_cache[i] = cands
        return cands

    def utt_vec(self, i: int, dim: int = 50) -> np.ndarray:
        """Official semantics: per-token lookup (miss -> <unka>), emax + average concatenated (100 dims)"""
        if i in self.utt_vec_cache:
            return self.utt_vec_cache[i]
        toks = self.log.messages[i].tokens
        vecs = []
        for w in toks:
            v = self.emb.get(w)
            if v is None:
                v = self.emb.get(w.lower())
            if v is None:
                v = self._unka
            vecs.append(v)
        m = np.stack(vecs)
        out = np.concatenate([m.max(0), m.mean(0)]).astype(np.float32)
        self.utt_vec_cache[i] = out
        return out

    def features(self, i: int, j: int) -> list:
        if self.feat_cache is not None:
            key = (i, j)
            if key in self.feat_cache:
                return self.feat_cache[key]
        base = get_features_official(self.log.name, i, j, self.text_ascii,
                                     self.info, self.target_info)
        if self.emb is not None:
            # official concatenation order: [features, qvec_max, qvec_mean, ovec_max, ovec_mean]
            qv = self.utt_vec(i)
            ov = self.utt_vec(j)
            base = list(base) + list(qv[:50]) + list(qv[50:]) + list(ov[:50]) + list(ov[50:])
        if self.feat_cache is not None:
            self.feat_cache[key] = base
        return base


def build_instances(logs: list, test_start: int = TEST_START, is_test=False, emb: dict = None):
    """Returns [(log_idx, query_no, cand_ids, feat_matrix, gold_idx_set)]; with a non-empty emb the features are 277-dim
    In emb mode (277 dims) LogPrep does not cache feature lists (avoids OOM); features are kept only as a float32 matrix
    Note: the full train split (62K instances x 101 x 277 x 4B ~= 7GB) hits the sandbox memory limit,
    so use build_train_memmap() for training; this function is only for small-scale (dev/test) evaluation"""
    preps = []
    instances = []
    for li, log in enumerate(logs):
        p = LogPrep(log, emb=emb, cache_feats=(emb is None))
        preps.append(p)
        if is_test:
            query_ids = range(test_start, len(log.messages))
        else:
            query_ids = sorted(set(k for k in log.parents if k >= 1 and k < len(log.messages)))
        for qi in query_ids:
            cands = p.candidates(qi)
            gold = set()
            for t in log.parents.get(qi, []):
                if t in cands:
                    gold.add(cands.index(t))
                # official semantics: a self-link (t == qi) marks the root, and cands contains qi itself
            if not is_test:
                ann = log.parents.get(qi)
                if ann is None:
                    continue  # queries without an annotation record are not trained
                # if the annotation contains a self-link, the root was already added to gold via cands.index(qi) above
                if not gold:
                    # all targets fall outside the window: the official code skips (do_instance: len(gold)==0 and train)
                    continue
            feats = np.empty((len(cands), 77 + (200 if emb is not None else 0)), dtype=np.float32)
            for k_, c in enumerate(cands):
                feats[k_] = p.features(qi, c)
            instances.append((li, qi, cands, feats, gold))
    return preps, instances


def build_train_memmap(logs: list, emb: dict, out_dir: str, test_start: int = TEST_START):
    """[Deprecated: memmap dirty pages count against the sandbox memory quota and get the process killed] kept only for small-scale experiments"""
    raise RuntimeError("use train_ff_stream() instead (sandbox memory limit kills memmap builds)")


# ---------------- model ----------------

class FFNet(nn.Module):
    def __init__(self, in_dim=277, hidden=512, layers=2, nonlin="softsign"):
        super().__init__()
        nl = {"softsign": nn.Softsign, "tanh": nn.Tanh, "relu": nn.ReLU}[nonlin]
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nl()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):  # x: (B, C, F)
        return self.net(x).squeeze(-1)  # (B, C)


def load_glove(path: str):
    """Official glove-ubuntu.txt: word + 50 dims"""
    emb = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 51:
                continue
            emb[parts[0].lower()] = np.array([float(v) for v in parts[1:]], dtype=np.float32)
    return emb


def glove_utt(emb, tokens, dim=50):
    """Official: emax + average over the tok sequence of query/option (4 segments concatenated)"""
    vecs = [emb.get(w.lower(), emb.get("<unka>", None)) for w in tokens]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        z = np.zeros(dim, dtype=np.float32)
        return np.concatenate([z, z])
    m = np.stack(vecs)
    return np.concatenate([m.max(0), m.mean(0)])


def stream_file_instances(log: ChatLog, emb: dict):
    """Generates training instances file by file (features computed and released on the fly). Peak memory = one file's feature block."""
    p = LogPrep(log, emb=emb, cache_feats=False)
    query_ids = sorted(set(k for k in log.parents if 1 <= k < len(log.messages)))
    for qi in query_ids:
        cands = p.candidates(qi)
        gold = set()
        for t in log.parents.get(qi, []):
            if t in cands:
                gold.add(cands.index(t))
        if not gold:
            continue
        feats = np.empty((len(cands), 277), dtype=np.float32)
        for k_, c in enumerate(cands):
            feats[k_] = p.features(qi, c)
        yield (log, qi, cands, feats, gold)


def train_ff_stream(train_logs, dev_logs, emb, epochs=8, lr=0.05, report_every=5000):
    """Zero-disk streaming training: rescans the train files every epoch, computing features on the fly (~65K instances/epoch).
    Works around the sandbox memory limit: at any moment only one file's feature block plus the precomputed query order are held."""
    import random
    model = FFNet(in_dim=277)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    step = 0
    for ep in range(epochs):
        cur_lr = lr / (1 + 0.103 * ep)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        order = list(range(len(train_logs)))
        random.Random(10 + ep).shuffle(order)
        total_loss, n_inst = 0.0, 0
        t0 = time.time()
        for li in order:
            log = train_logs[li]
            for (lg, qi, cands, feats, gold) in stream_file_instances(log, emb):
                logits = model(torch.from_numpy(feats).unsqueeze(0)).squeeze(0)
                target = torch.zeros_like(logits)
                for g_ in gold:
                    target[g_] = 1.0
                target = target / target.sum()
                loss = -(target * torch.log_softmax(logits, dim=0)).sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.74)
                opt.step()
                total_loss += loss.item()
                n_inst += 1
                step += 1
                if step % report_every == 0:
                    print("  ep{} step{} lr={:.4f} loss={:.4f} ({:.0f}s)".format(
                        ep, step, cur_lr, total_loss / report_every, time.time() - t0), flush=True)
                    total_loss = 0.0
        print("  ep{} done: {} instances ({:.0f}s)".format(ep, n_inst, time.time() - t0), flush=True)
    return model


def train_ff(train_inst, dev_logs, epochs=8, lr=0.05, report_every=5000):
    model = FFNet()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    packed = []
    for li, qi, cands, feats, gold in train_inst:
        if gold:
            packed.append((torch.from_numpy(feats), gold))
    print("train queries: {}".format(len(packed)))
    step = 0
    for ep in range(epochs):
        import random
        # official lr decay: lr / (1 + decay*epoch)
        cur_lr = lr / (1 + 0.103 * ep)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        random.Random(10 + ep).shuffle(packed)
        total_loss = 0.0
        t0 = time.time()
        for feats, gold in packed:
            logits = model(feats.unsqueeze(0)).squeeze(0)
            target = torch.zeros_like(logits)
            for g in gold:
                target[g] = 1.0
            target = target / target.sum()
            loss = -(target * torch.log_softmax(logits, dim=0)).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.74)
            opt.step()
            total_loss += loss.item()
            step += 1
            if step % report_every == 0:
                print("  ep{} step{} lr={:.4f} loss={:.4f} ({:.0f}s)".format(
                    ep, step, cur_lr, total_loss / report_every, time.time() - t0))
                total_loss = 0.0
    return model


def predict_links(model, preps, logs, test_start=TEST_START, threshold=0.5):
    """Outputs the predicted edge set for every query (a (src,tgt) set aligned with the official evaluate)"""
    model.eval()
    all_edges = {}
    with torch.no_grad():
        for li, log in enumerate(logs):
            p = preps[li]
            edges = set()
            for qi in range(test_start, len(log.messages)):
                cands = p.candidates(qi)
                if not cands:
                    continue
                feats = np.array([p.features(qi, c) for c in cands], dtype=np.float32)
                logits = model(torch.from_numpy(feats))
                probs = torch.softmax(logits, dim=0).numpy()
                # greedy: keep only top-1 (the official default single-link greedy); if top1 is self it is the root
                top = int(probs.argmax())
                tgt = cands[top]
                edges.add((qi, tgt))
            all_edges[short_name(log.name)] = edges
    return all_edges


def link_prf(gold_sets, auto_sets):
    tg = ta = m = 0
    for name in gold_sets:
        g = gold_sets[name]
        a = auto_sets.get(name, set())
        tg += len(g)
        ta += len(a)
        m += len(g & a)
    p = 100 * m / ta if ta else 0
    r = 100 * m / tg if tg else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return {"P": round(p, 1), "R": round(r, 1), "F1": round(f, 1), "gold": tg, "auto": ta, "matched": m}


def short_name(name: str) -> str:
    """gold files use the short name (dropping the .train-a/.train-c suffix), used for alignment"""
    return name.split(".")[0]


def gold_edge_sets(logs, test_start=TEST_START):
    out = {}
    for log in logs:
        s = set()
        for src, tgts in log.parents.items():
            if src < test_start:
                continue
            for t in tgts:
                s.add((src, t))
        out[short_name(log.name)] = s
    return out


def main():
    print("loading data ...")
    train_logs = load_split("train", DATA_ROOT)
    dev_logs = load_split("dev", DATA_ROOT)
    test_logs = load_split("test", DATA_ROOT)
    print("files:", len(train_logs), len(dev_logs), len(test_logs))

    glove_path = os.path.join(DATA_ROOT, "glove-ubuntu.txt")
    emb = None
    if os.path.exists(glove_path) and os.path.getsize(glove_path) > 29_000_000:
        print("loading GloVe word vectors ...")
        emb = load_glove(glove_path)
        print("  vocab:", len(emb))
    else:
        print("GloVe unavailable, using 77-d features only")

    print("skipping upfront instance build (stream training)")

    print("training FF model (stream) ...")
    model = train_ff_stream(train_logs, dev_logs, emb, epochs=8, lr=0.05)

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "ff_model_glove.pt"))
    print("model saved")

    # evaluate dev + test
    for split, logs in [("dev", dev_logs), ("test", test_logs)]:
        preps, _ = build_instances(logs, is_test=True, emb=emb)
        auto = predict_links(model, preps, logs)
        gold = gold_edge_sets(logs)
        print(split, link_prf(gold, auto))

    # save the model and the predictions
    import json
    with open(os.path.join(CKPT_DIR, "ff_predictions_test_glove.json"), "w") as f:
        json.dump({k: [list(e) for e in sorted(v)] for k, v in auto.items()}, f)
    print("saved predictions")


if __name__ == "__main__":
    main()
