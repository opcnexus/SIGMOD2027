"""The three-layer metric system (matching the unified evaluation protocol of reports/datasets_5_gold.md).

L1 edge level (1-to-1 backtracking): Precision / Recall / F1 / Accuracy
L2 chain level (full-chain backtracking): Exact Match (EM) / prefix@k / chain-F1
L3 the gap: Gap = edge_acc - chain_EM, plus a "chain length L -> accuracy" decay curve
"""
from methods.base import chain_from


def edge_metrics(inst, pred_parent):
    """L1: set-level P/R/F1 between predicted and gold edge sets, plus per-node accuracy."""
    gold_edges, pred_edges, correct = set(), set(), 0
    nodes = list(range(1, inst.n))
    for i in nodes:
        g = inst.gold_parent.get(i, -1)
        p = pred_parent.get(i, -1)
        if g != -1:
            gold_edges.add((i, g))
        if p != -1:
            pred_edges.add((i, p))
        if g == p:
            correct += 1
    inter = gold_edges & pred_edges
    prec = len(inter) / len(pred_edges) if pred_edges else 0.0
    rec = len(inter) / len(gold_edges) if gold_edges else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = correct / len(nodes) if nodes else 0.0
    return {
        "edge_precision": prec,
        "edge_recall": rec,
        "edge_f1": f1,
        "edge_acc": acc,
        "n_edges_gold": len(gold_edges),
        "n_edges_pred": len(pred_edges),
    }


def chain_metrics(inst, pred_parent, prefix_k=(1, 2, 3)):
    """L2: rebuild the full chain for each node and compare it with the gold chain."""
    em, f1s, hits = 0, [], 0
    prefix_hit = {k: 0 for k in prefix_k}
    prefix_tot = {k: 0 for k in prefix_k}
    by_len = {}
    nodes = list(range(inst.n))
    for i in nodes:
        g = inst.gold_chain(i)
        p = chain_from(inst, pred_parent, i)
        if p == g:
            em += 1
            hits += 1
        gs, ps = set(g), set(p)
        inter = len(gs & ps)
        f1s.append(2 * inter / (len(gs) + len(ps)) if (gs or ps) else 0.0)
        for k in prefix_k:
            if len(g) >= k:
                prefix_tot[k] += 1
                if p[:k] == g[:k]:
                    prefix_hit[k] += 1
        L = len(g)
        by_len.setdefault(L, [0, 0])
        by_len[L][0] += 1
        if p == g:
            by_len[L][1] += 1
    n = len(nodes) or 1
    return {
        "chain_EM": em / n,
        "chain_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "prefix_acc": {
            k: (prefix_hit[k] / prefix_tot[k] if prefix_tot[k] else 0.0) for k in prefix_k
        },
        "by_len": {str(L): {"n": v[0], "em": v[1] / v[0] if v[0] else 0.0} for L, v in sorted(by_len.items())},
    }


def length_stratified(rows, buckets=((2, 3), (4, 5), (6, 8))):
    """L3: EM bucketed by gold chain length, to observe the exponential p^L decay."""
    out = []
    for lo, hi in buckets:
        tot, hit = 0, 0
        for r in rows:
            for L, v in r["by_len"].items():
                L = int(L)
                if lo <= L <= hi:
                    tot += v["n"]
                    hit += v["em"] * v["n"]
        out.append({"bucket": "%d-%d" % (lo, hi), "n": tot, "em": (hit / tot) if tot else None})
    return out


def combine(inst, pred_parent, prefix_k=(1, 2, 3)):
    e = edge_metrics(inst, pred_parent)
    c = chain_metrics(inst, pred_parent, prefix_k)
    return {
        **e,
        **c,
        "gap": e["edge_acc"] - c["chain_EM"],  # the L3 gap
    }
