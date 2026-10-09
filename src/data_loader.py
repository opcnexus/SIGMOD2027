#!/usr/bin/env python3
"""Data loader for the official irc-disentanglement format (semantics calibrated
against the official disentangle.py / graph-eval.py)

Format essentials (confirmed against the official code):
1. .tok.txt holds one message per line: `<s> [prefix words...] body </s>`; the line
   number (0-based) is the message id.
2. .annotation.txt holds `id1 id2 -` per line: max(id) is the replying message
   (source), min(id) is the message being replied to (target).
   - A self-link (1002,1002) means message 1002 starts a new conversation (root).
   - The official evaluation only starts at id>=1000 (the first 1000 lines are context).
3. .ascii.txt holds per line: `[HH:MM] <speaker> text`, or `=== ... ===` for a
   system message.
4. gold.*.graphs.txt: `filename:src tgt -`, the globally merged version with the
   same semantics.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

FEATURE_NEVER = 0  # placeholder


@dataclass
class Message:
    mid: int              # 0-based line number = message id
    speaker: str          # '' for system messages
    text: str             # tokenized text (no <s></s>)
    raw: str              # ascii line
    is_system: bool
    time: str = ""
    tokens: list = field(default_factory=list)  # official format: ["<s>", ...] without </s>


@dataclass
class ChatLog:
    name: str             # e.g. 2016-02-22_17
    messages: List[Message] = field(default_factory=list)
    # source -> set of targets; includes the self-link root
    parents: Dict[int, List[int]] = field(default_factory=lambda: {})

    def replies_of(self, mid: int) -> List[int]:
        """Given a message mid, return all earlier messages it replies to (self-links
        excluded, since a self-link means root)"""
        return [t for t in self.parents.get(mid, []) if t != mid]

    def is_root(self, mid: int) -> bool:
        return mid in self.parents.get(mid, [])


_TIME_RE = re.compile(r"^\[(\d{2}:\d{2})\]")


def parse_ascii_line(line: str) -> Tuple[str, str, bool]:
    """Returns (speaker, text, is_system)"""
    line = line.rstrip("\n")
    m = _TIME_RE.match(line)
    time = m.group(1) if m else ""
    body = line[m.end():].strip() if m else line
    if body.startswith("==="):
        return "", body, True
    sm = re.match(r"<([^>]+)>\s*(.*)", body)
    if sm:
        return sm.group(1), sm.group(2), False
    # system line with no speaker
    return "", body, True


def load_chatlog(name_prefix: str, with_gold: bool = True) -> Optional[ChatLog]:
    """name_prefix carries no suffix, e.g. .../dev/2004-11-15_03"""
    tok_path = name_prefix + ".tok.txt"
    if not os.path.exists(tok_path):
        return None
    log = ChatLog(name=os.path.basename(name_prefix))
    # tok text
    tok_lines = [l.rstrip("\n") for l in open(tok_path, encoding="utf-8", errors="replace")]
    # ascii provides the speaker
    ascii_path = name_prefix + ".ascii.txt"
    ascii_lines = []
    if os.path.exists(ascii_path):
        ascii_lines = [l.rstrip("\n") for l in open(ascii_path, encoding="utf-8", errors="replace")]
    for i, tl in enumerate(tok_lines):
        toks = tl.strip().split()
        if toks and toks[-1] == "</s>":
            toks = toks[:-1]
        # official read_data semantics: make sure the first token is <s> (kept, for
        # use with word vectors)
        if not toks or toks[0] != "<s>":
            toks = ["<s>"] + toks
        body = toks[1:]  # text excludes <s>
        text = " ".join(body)
        speaker, _, is_sys = ("", "", True)
        if i < len(ascii_lines):
            speaker, _, is_sys = parse_ascii_line(ascii_lines[i])
        log.messages.append(Message(mid=i, speaker=speaker, text=text,
                                    raw=ascii_lines[i] if i < len(ascii_lines) else "",
                                    is_system=is_sys, tokens=toks))
    # annotations
    if with_gold:
        ann_path = name_prefix + ".annotation.txt"
        if os.path.exists(ann_path):
            for line in open(ann_path, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                nums = [int(v) for v in line.split() if v != "-"]
                if len(nums) < 2:
                    continue
                src = max(nums)
                for t in nums:
                    if t != src:
                        log.parents.setdefault(src, []).append(t)
                if len(nums) == 2 and nums[0] == nums[1]:
                    log.parents.setdefault(src, []).append(src)
    return log


def load_split(split: str, data_root: str) -> List[ChatLog]:
    """Load every ChatLog under the train/dev/test directory (tok + ascii +
    annotation).
    Note: train file names carry a .train-a/.train-c suffix, and ChatLog.name keeps
    the full base name"""
    d = os.path.join(data_root, split)
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".tok.txt"):
            prefix = os.path.join(d, f[:-len(".tok.txt")])
            log = load_chatlog(prefix)
            if log is not None:
                out.append(log)
    return out


def gold_graph_pairs(split: str, data_root: str) -> Dict[str, set]:
    """Read gold.{split}.graphs.txt -> {filename: set((src,tgt))}, aligned with the
    official graph-eval.py"""
    path = os.path.join(data_root, "gold.{}.graphs.txt".format(split))
    storage: Dict[str, set] = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fname = ""
        if ":" in line.split(" ")[0]:
            fname, _, line = line.partition(":")
        nums = [int(n) for n in line.split() if n != "-"]
        src = max(nums)
        nums.remove(src)
        if len(nums) == 0 or (nums[0] == src and len(nums) == 1):
            storage.setdefault(fname, set()).add((src, src))
        else:
            for n in nums:
                storage.setdefault(fname, set()).add((src, n))
    return storage


def stats(logs: List[ChatLog]) -> str:
    n_msg = sum(len(l.messages) for l in logs)
    n_edges = sum(sum(len(v) for v in l.parents.values()) for l in logs)
    n_self = sum(1 for l in logs for src, ts in l.parents.items() for t in ts if t == src)
    return "files={} msgs={} edges={} self-links={}".format(len(logs), n_msg, n_edges, n_self)


if __name__ == "__main__":
    import sys
    DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for split in ["dev", "test", "train"]:
        logs = load_split(split, DATA_ROOT)
        if logs:
            print(split, stats(logs))
            # spot-check one reply relation
            for log in logs:
                for src, tgts in log.parents.items():
                    real = [t for t in tgts if t != src]
                    if real:
                        m1 = log.messages[src] if src < len(log.messages) else None
                        m2 = log.messages[real[0]] if real[0] < len(log.messages) else None
                        if m1 and m2 and not m1.is_system and not m2.is_system:
                            print("  sample: [{}] {} -> [{}] {}".format(
                                src, m1.text[:50], real[0], m2.text[:50]))
                            break
                else:
                    continue
                break
