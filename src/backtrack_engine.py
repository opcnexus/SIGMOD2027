#!/usr/bin/env python3
"""repspace engine: given one utterance, look backwards for all the historical
utterances related to it.

Usage 1 (as a library):
    from backtrack_engine import BacktrackEngine
    eng = BacktrackEngine()
    result = eng.trace("2016-02-22_17", 1043, max_hops=5)
    -> returns a directed chain/tree plus relatedness scores

Usage 2 (CLI):
    python3 backtrack_engine.py --file 2016-02-22_17 --id 1043
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ChatLog, load_chatlog, load_split, parse_ascii_line
from baselines import TFIDF, CombinedLinear, norm_speaker, mentioned_users, token_overlap

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TraceNode:
    mid: int
    speaker: str
    text: str
    score: float          # estimated relatedness to the query message
    depth: int            # 0 = the query message itself
    relation: str         # "direct-parent" | "thread-context"


@dataclass
class TraceResult:
    query_mid: int
    file: str
    query_speaker: str
    query_text: str
    chain: List[TraceNode] = field(default_factory=list)      # direct parent chain (recursive reply-to)
    context: List[TraceNode] = field(default_factory=list)     # sibling / related messages of the same thread
    thread_members: List[int] = field(default_factory=list)    # inferred members of the whole thread

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class BacktrackEngine:
    """Forward backtracking engine built around the CombinedLinear scorer"""

    def __init__(self, model: Optional[CombinedLinear] = None, logs: Optional[List[ChatLog]] = None):
        self.logs = logs or []
        self.by_name: Dict[str, ChatLog] = {l.name: l for l in self.logs}
        if model is None:
            tf = TFIDF()
            tf.fit([m.text for l in self.logs for m in l.messages])
            model = CombinedLinear(tf)
        self.model = model

    # ----- core tracing -----

    def trace(self, fname: str, mid: int, max_hops: int = 10) -> TraceResult:
        log = self.by_name[fname]
        q = log.messages[mid]
        res = TraceResult(query_mid=mid, file=fname, query_speaker=q.speaker, query_text=q.text)

        # 1) recursively trace the parent chain upwards
        cur = mid
        visited = {mid}
        for _ in range(max_hops):
            parents = self.model.predict_parents(log, cur)
            real = [p for p in parents if p != cur and p not in visited]
            if not real:
                break
            for p in real:
                pm = log.messages[p]
                res.chain.append(TraceNode(
                    mid=p, speaker=pm.speaker, text=pm.text,
                    score=float(self.model.score(log, cur, p)), depth=len(res.chain) + 1,
                    relation="direct-parent"))
                visited.add(p)
            cur = real[0]

        # 2) thread context: messages that are strongly linked to the query message
        #    or to a member of its parent chain
        thread = set(visited)
        frontier = list(visited)
        for _ in range(2):  # two rounds of expansion
            new_frontier = []
            for m in frontier:
                # find who replied to m (looking forward) and whom m replied to
                # (looking backward)
                for j in range(max(0, m - 100), min(len(log.messages), m + 100)):
                    if j in thread or log.messages[j].is_system:
                        continue
                    parents_j = self.model.predict_parents(log, j)
                    if m in parents_j:
                        thread.add(j)
                        new_frontier.append(j)
            frontier = new_frontier
            if not frontier:
                break

        for m in sorted(thread - visited):
            mm = log.messages[m]
            if mm.is_system:
                continue
            res.context.append(TraceNode(
                mid=m, speaker=mm.speaker, text=mm.text,
                score=float(self.model.score(log, mid, m)) if m != mid else 1.0,
                depth=-1, relation="thread-context"))
        res.thread_members = sorted(thread)
        res.context.sort(key=lambda n: -n.score)
        return res

    # ----- demo: start a backtrack from arbitrary text (locate the most similar
    # message first) -----

    def locate(self, text: str) -> List[Tuple[str, int, float]]:
        """Locate the K messages most similar to text in the whole corpus"""
        tf = self.model.tfidf
        vq = tf.vec(text)
        hits = []
        for log in self.logs:
            for m in log.messages:
                if m.is_system:
                    continue
                s = TFIDF.cosine(vq, tf.vec(m.text))
                if s > 0.15:
                    hits.append((log.name, m.mid, s))
        hits.sort(key=lambda x: -x[2])
        return hits[:10]


def build_default_engine() -> BacktrackEngine:
    print("loading logs ...", file=sys.stderr)
    logs = load_split("test", DATA_ROOT) + load_split("dev", DATA_ROOT)
    print("building engine ...", file=sys.stderr)
    return BacktrackEngine(logs=logs)


def _print_trace(res: TraceResult, show_context: int = 8):
    print("=" * 72)
    print("query: [{:>5}] {}: {}".format(res.query_mid, res.query_speaker, res.query_text))
    print("=" * 72)
    if res.chain:
        print("\n-- upward trace (reply-to parent chain) --")
        for n in res.chain:
            print("  d{} [{:>5}] ({:+.2f}) {}: {}".format(n.depth, n.mid, n.score, n.speaker, n.text))
    else:
        print("\n-- this message is the thread root (start of a new topic) --")
    if res.context:
        print("\n-- same-thread context (by relatedness, top {}) --".format(show_context))
        for n in res.context[:show_context]:
            print("  [{:>5}] ({:+.2f}) {}: {}".format(n.mid, n.score, n.speaker, n.text))
    print("\n{} messages in the thread in total".format(len(res.thread_members)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="repspace engine CLI")
    ap.add_argument("--file", help="Log file name (e.g. 2016-02-22_17)")
    ap.add_argument("--id", type=int, help="Query message id (0-based line number)")
    ap.add_argument("--text", help="Locate the query message by text")
    ap.add_argument("--max-hops", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    eng = build_default_engine()
    if args.text:
        hits = eng.locate(args.text)
        if not hits:
            print("no match found"); sys.exit(1)
        fname, mid, s = hits[0]
        print("located: {}:{} score={:.3f}".format(fname, mid, s))
    else:
        if not args.file or args.id is None:
            ap.error("need --file + --id, or --text")
        fname, mid = args.file, args.id

    res = eng.trace(fname, mid, max_hops=args.max_hops)
    if args.json:
        print(res.to_json())
    else:
        _print_trace(res)
